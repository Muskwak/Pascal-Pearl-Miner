#pragma once

// Pascal (sm_61) replacement for the SM90 MerkleTreeRootsKernel.
//
// Identical algorithm — one BLAKE3 keyed-hash leaf per 1024-byte chunk, then a
// per-CTA Merkle reduction over kNumConsumerThreads leaves — but it loads chunk
// data directly from global memory instead of via TMA bulk copies and warpgroup
// producer/consumer pipelines (which only exist on Hopper). Each thread owns one
// chunk; the downstream ComputeBlakeMTKernel / ReduceRootsKernel stages are the
// stock CuTe kernels and are reused unchanged.

#include "blake3/blake3.cuh"
#include "cute/tensor.hpp"
#include "merkle_tree_utils.hpp"
#include "tensor_hash_constants.cuh"

#include <cutlass/cutlass.h>

namespace pearl {

using namespace cute;

template <int kNumConsumerThreads>
class MerkleTreeRootsKernelSm61 {
 public:
  using Element = uint8_t;
  static constexpr uint32_t MaxThreadsPerBlock = kNumConsumerThreads;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  static constexpr int kNumBlocksPerChunk =
      blake3::CHUNK_SIZE / blake3::MSG_BLOCK_SIZE;  // 16
  static constexpr int kNumWordsPerBlock =
      blake3::MSG_BLOCK_SIZE / sizeof(uint32_t);    // 16

  // Leaves staged in shared memory as (8 words, kNumConsumerThreads leaves);
  // element (word, leaf) at word*kNumConsumerThreads + leaf.
  using SmemLayoutLeaves =
      Layout<Shape<Int<blake3::CHAINING_VALUE_SIZE_U32>,
                   Int<kNumConsumerThreads>>,
             Stride<Int<kNumConsumerThreads>, Int<1>>>;

  static constexpr int SharedStorageSize =
      kNumConsumerThreads * blake3::CHAINING_VALUE_SIZE;  // bytes

  struct Arguments {
    const Element* ptr_data;
    uint32_t data_len;
    Element* ptr_roots;
  };
  struct Params {
    const Element* ptr_data;
    uint32_t data_len;
    Element* ptr_roots;
  };

  static Params to_underlying_arguments(Arguments const& a) {
    return Params{a.ptr_data, a.data_len, a.ptr_roots};
  }

  static dim3 get_grid_shape(Params const& p) {
    const uint32_t num_chunks =
        (p.data_len + blake3::CHUNK_SIZE - 1) / blake3::CHUNK_SIZE;
    return dim3((num_chunks + kNumConsumerThreads - 1) / kNumConsumerThreads);
  }
  static dim3 get_block_shape() { return dim3(kNumConsumerThreads); }

  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
    const int tid = threadIdx.x;
    Tensor sLeaves = make_tensor(
        make_smem_ptr(reinterpret_cast<uint32_t*>(smem_buf)),
        SmemLayoutLeaves{});

    const uint32_t num_chunks =
        (params.data_len + blake3::CHUNK_SIZE - 1) / blake3::CHUNK_SIZE;
    const uint32_t num_grid_blocks =
        (num_chunks + kNumConsumerThreads - 1) / kNumConsumerThreads;
    const uint32_t global_chunk_idx = blockIdx.x * kNumConsumerThreads + tid;

    // Leaf = BLAKE3 keyed-hash of the chunk. Chaining value starts at the key.
    Tensor rCV = make_tensor<uint32_t>(
        Layout<Shape<Int<blake3::CHAINING_VALUE_SIZE_U32>>>{});
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) rCV(i) = c_key[i];

    if (global_chunk_idx < num_chunks) {
      const uint32_t chunk_start = global_chunk_idx * blake3::CHUNK_SIZE;
      const uint32_t chunk_len =
          min((uint32_t)blake3::CHUNK_SIZE, params.data_len - chunk_start);

      Tensor rBlock =
          make_tensor<uint32_t>(Layout<Shape<Int<kNumWordsPerBlock>>>{});
      for (int b = 0; b < kNumBlocksPerChunk; ++b) {
        const uint32_t block_start = b * blake3::MSG_BLOCK_SIZE;  // within chunk
        // Assemble 16 little-endian words (64 bytes), zero-padding past the
        // valid chunk length (matches the SM90 kernel: block_len stays 64).
        CUTLASS_PRAGMA_UNROLL
        for (int w = 0; w < kNumWordsPerBlock; ++w) {
          const uint32_t byte_off = block_start + w * 4u;  // within chunk
          uint32_t word = 0;
          CUTLASS_PRAGMA_UNROLL
          for (int by = 0; by < 4; ++by) {
            const uint32_t cb = byte_off + by;
            if (cb < chunk_len) {
              word |= (uint32_t)params.ptr_data[chunk_start + cb] << (by * 8);
            }
          }
          rBlock(w) = word;
        }

        blake3::CompressParams cp{.counter = global_chunk_idx,
                                  .block_len = blake3::MSG_BLOCK_SIZE,
                                  .flags = blake3::KEYED_HASH};
        if (b == 0) cp.flags |= blake3::CHUNK_START;
        if (b == kNumBlocksPerChunk - 1) cp.flags |= blake3::CHUNK_END;
        blake3::compress_msg_block_u32(rBlock, rCV, cp);
      }
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i)
      sLeaves(i, tid) = rCV(i);
    __syncthreads();

    // Number of valid leaves in this CTA (mirrors the SM90 kernel exactly).
    const bool is_last_block = (blockIdx.x == num_grid_blocks - 1);
    uint32_t num_leaves;
    if (!is_last_block) {
      num_leaves = kNumConsumerThreads;
    } else {
      const uint32_t chunks_in_block = num_chunks % kNumConsumerThreads;
      const uint32_t actual_chunks =
          (chunks_in_block == 0) ? kNumConsumerThreads : chunks_in_block;
      const uint32_t remainder_bytes = params.data_len % blake3::CHUNK_SIZE;
      const bool last_chunk_too_small =
          (remainder_bytes > 0) && (remainder_bytes < blake3::MSG_BLOCK_SIZE);
      num_leaves = last_chunk_too_small
                       ? (actual_chunks > 0 ? actual_chunks - 1 : 0)
                       : actual_chunks;
    }

    // Per-CTA Merkle reduction (root ends up at sLeaves(_, 0)).
    if (!is_last_block) {
      merkle_tree_utils::compute_perfect_mt<false>(sLeaves, kNumConsumerThreads);
    } else if ((num_leaves & (num_leaves - 1)) == 0) {
      merkle_tree_utils::compute_perfect_mt<false>(sLeaves, num_leaves);
    } else {
      merkle_tree_utils::compute_blake_mt<false>(sLeaves, num_leaves);
    }

    // Write this CTA's root: roots laid out (8, num_grid_blocks), stride (1, 8).
    if (tid < blake3::CHAINING_VALUE_SIZE_U32) {
      uint32_t* roots = reinterpret_cast<uint32_t*>(params.ptr_roots);
      roots[blockIdx.x * blake3::CHAINING_VALUE_SIZE_U32 + tid] =
          sLeaves(tid, 0);
    }
  }
};

}  // namespace pearl
