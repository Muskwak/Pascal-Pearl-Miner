#pragma once

// Pascal (sm_61) host orchestration for tensor_hash.
//
// Same three-stage pipeline as the SM90 host (tensor_hash_host.hpp), but stage 1
// uses MerkleTreeRootsKernelSm61 (direct global loads) instead of the Hopper
// TMA/warpgroup kernel. Stages 2 and 3 (ComputeBlakeMTKernel, ReduceRootsKernel)
// and the commitment-hash kernel are the stock CuTe kernels, reused unchanged.
// The `num_stages` argument is accepted for API compatibility but ignored (it
// only configured the SM90 TMA pipeline).

#include <assert.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#include "blake3/blake3_constants.hpp"
#include "commitment_hash_from_merkle_roots_kernel.hpp"
#include "compute_blake_mt_kernel.hpp"
#include "gemm/error_check.hpp"
#include "merkle_tree_roots_kernel_sm61.hpp"
#include "reduce_roots_kernel.h"
#include "tensor_hash_constants.cuh"

#include <cutlass/cutlass.h>
#include "cutlass/device_kernel.h"

using u8 = uint8_t;
using u32 = uint32_t;

namespace {
inline void th_set_key(const uint8_t d_key[blake3::KEY_SIZE]) {
  gpuErrchk(cudaMemcpyToSymbol(c_key, d_key, blake3::KEY_SIZE, 0,
                               cudaMemcpyDeviceToDevice));
}
}  // namespace

template <int kNumConsumerThreads, int kLeavesPerMTBlock>
void tensor_hash_impl_sm61(const uint8_t* data, uint32_t data_size,
                           uint8_t* out, const uint8_t key[blake3::KEY_SIZE],
                           uint32_t /*num_blocks*/, uint8_t* roots,
                           cudaStream_t stream) {
  th_set_key(key);
  const u32 data_len = data_size;

  // ---- Stage 1: per-chunk BLAKE3 leaf + per-CTA Merkle root (Pascal) ----
  using RootsKernel = pearl::MerkleTreeRootsKernelSm61<kNumConsumerThreads>;
  typename RootsKernel::Arguments args{data, data_len, roots};
  typename RootsKernel::Params kernel_params =
      RootsKernel::to_underlying_arguments(args);
  auto roots_kernel = cutlass::device_kernel<RootsKernel>;
  dim3 grid = RootsKernel::get_grid_shape(kernel_params);
  dim3 block = RootsKernel::get_block_shape();
  constexpr static int roots_smem_size = RootsKernel::SharedStorageSize;
  if (roots_smem_size >= 48 * 1024) {
    gpuErrchk(cudaFuncSetAttribute(reinterpret_cast<const void*>(roots_kernel),
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   roots_smem_size));
  }
  roots_kernel<<<grid, block, roots_smem_size, stream>>>(kernel_params);
  gpuErrchk(cudaGetLastError());

  // Number of stage-1 roots = number of grid blocks.
  const u32 num_chunks =
      (data_len + blake3::CHUNK_SIZE - 1) / blake3::CHUNK_SIZE;
  const u32 num_roots =
      (num_chunks + kNumConsumerThreads - 1) / kNumConsumerThreads;

  // ---- Stage 2: Merkle tree over the stage-1 roots ----
  const int num_blocks_for_mt =
      (num_roots + kLeavesPerMTBlock - 1) / kLeavesPerMTBlock;
  const bool is_single_block = (num_blocks_for_mt == 1);

  if (is_single_block) {
    using K = pearl::ComputeBlakeMTKernel<kLeavesPerMTBlock, true>;
    typename K::Arguments a{reinterpret_cast<uint32_t*>(roots), num_roots};
    typename K::Params p = K::to_underlying_arguments(a);
    auto k = cutlass::device_kernel<K>;
    constexpr static int smem = K::SharedStorageSize;
    if (smem >= 48 * 1024) {
      gpuErrchk(cudaFuncSetAttribute(reinterpret_cast<const void*>(k),
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     smem));
    }
    k<<<K::get_grid_shape(p), K::get_block_shape(), smem, stream>>>(p);
    gpuErrchk(cudaGetLastError());
  } else {
    using K = pearl::ComputeBlakeMTKernel<kLeavesPerMTBlock, false>;
    typename K::Arguments a{reinterpret_cast<uint32_t*>(roots), num_roots};
    typename K::Params p = K::to_underlying_arguments(a);
    auto k = cutlass::device_kernel<K>;
    constexpr static int smem = K::SharedStorageSize;
    if (smem >= 48 * 1024) {
      gpuErrchk(cudaFuncSetAttribute(reinterpret_cast<const void*>(k),
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     smem));
    }
    k<<<K::get_grid_shape(p), K::get_block_shape(), smem, stream>>>(p);
    gpuErrchk(cudaGetLastError());

    // ---- Stage 3: reduce the per-MT-block roots into the final root ----
    using R = pearl::ReduceRootsKernel<kNumConsumerThreads>;
    typename R::Arguments a3{reinterpret_cast<uint32_t*>(roots),
                             static_cast<uint32_t>(num_blocks_for_mt)};
    typename R::Params p3 = R::to_underlying_arguments(a3);
    auto k3 = cutlass::device_kernel<R>;
    constexpr static int smem3 = R::SharedStorageSize;
    if (smem3 >= 48 * 1024) {
      gpuErrchk(cudaFuncSetAttribute(reinterpret_cast<const void*>(k3),
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     smem3));
    }
    k3<<<R::get_grid_shape(p3), R::get_block_shape(), smem3, stream>>>(p3);
    gpuErrchk(cudaGetLastError());
  }

  gpuErrchk(cudaMemcpyAsync(out, roots, blake3::CHAINING_VALUE_SIZE,
                            cudaMemcpyDeviceToDevice, stream));
}

template <int kNumConsumerThreads>
static void dispatch_leaves_sm61(uint32_t leaves_per_mt_block,
                                 const uint8_t* data, uint32_t data_size,
                                 uint8_t* out,
                                 const uint8_t key[blake3::KEY_SIZE],
                                 uint32_t num_blocks, uint8_t* roots,
                                 cudaStream_t stream) {
  switch (leaves_per_mt_block) {
    case 256:
      tensor_hash_impl_sm61<kNumConsumerThreads, 256>(
          data, data_size, out, key, num_blocks, roots, stream);
      break;
    case 512:
      tensor_hash_impl_sm61<kNumConsumerThreads, 512>(
          data, data_size, out, key, num_blocks, roots, stream);
      break;
    case 1024:
      tensor_hash_impl_sm61<kNumConsumerThreads, 1024>(
          data, data_size, out, key, num_blocks, roots, stream);
      break;
    default:
      throw std::runtime_error("Unsupported leaves_per_mt_block: " +
                               std::to_string(leaves_per_mt_block) +
                               ". Supported values are: 256, 512, 1024");
  }
}

void tensor_hash(const uint8_t* data, uint32_t data_size, uint8_t* out,
                 const uint8_t key[32], uint32_t num_blocks,
                 uint32_t threads_per_block, uint32_t /*num_stages*/,
                 uint32_t leaves_per_mt_block, uint8_t* roots,
                 cudaDeviceProp& /*deviceProp*/, cudaStream_t stream) {
  switch (threads_per_block) {
    case 128:
      dispatch_leaves_sm61<128>(leaves_per_mt_block, data, data_size, out, key,
                                num_blocks, roots, stream);
      break;
    case 256:
      dispatch_leaves_sm61<256>(leaves_per_mt_block, data, data_size, out, key,
                                num_blocks, roots, stream);
      break;
    case 512:
      dispatch_leaves_sm61<512>(leaves_per_mt_block, data, data_size, out, key,
                                num_blocks, roots, stream);
      break;
    default:
      throw std::runtime_error("Unsupported threads_per_block: " +
                               std::to_string(threads_per_block) +
                               ". Supported values are: 128, 256, 512");
  }
}

void commitment_hash_from_merkle_roots(
    const uint8_t* A_merkle_root, const uint8_t* B_merkle_root,
    const uint8_t* key, uint8_t* A_commitment_hash, uint8_t* B_commitment_hash,
    cudaDeviceProp& /*deviceProp*/, cudaStream_t stream) {
  using K = pearl::CommitmentHashFromMerkleRootsKernel;
  typename K::Arguments args{A_merkle_root, B_merkle_root, key,
                             A_commitment_hash, B_commitment_hash};
  typename K::Params p = K::to_underlying_arguments(args);
  auto k = cutlass::device_kernel<K>;
  constexpr static int smem = K::SharedStorageSize;
  k<<<dim3(1), dim3(1), smem, stream>>>(p);
  gpuErrchk(cudaGetLastError());
}
