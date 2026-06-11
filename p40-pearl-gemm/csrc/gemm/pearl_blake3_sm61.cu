// Pascal (sm_61) BLAKE3-only Pearl kernel — consumes transcript buffer from
// pearl_gemm_only_sm61.cu and produces digests / found-flag.
//
// One thread per hash tile: reads 16 uint32 from the global transcript buffer,
// computes keyed BLAKE3 (single block, exactly like the BLAKE3 tail of
// pearl_pow_fused_sm61.cu), writes digest to out_digests (if non-null), and
// compares <= target to set found_flag/found_coord.
//
// Sub-millisecond even for 1M tiles.
//
// Bit-exact: the BLAKE3 call and comparison are identical to the fused kernel.

#include <cuda_runtime.h>
#include <cstdint>

#include <cute/tensor.hpp>
#include "blake3/blake3.cuh"

using namespace cute;

static constexpr int HT_B3 = 16;
static constexpr int TRANSCRIPT_U32_B3 = 16;

__global__ void pearl_blake3_kernel(
    const uint32_t* __restrict__ transcript_buffer,  // [num_tiles, 16] input
    int num_tiles,
    int n,
    const uint32_t* __restrict__ pow_key,     // 8 words
    const uint32_t* __restrict__ pow_target,  // 8 words, little-endian
    uint8_t* __restrict__ out_digests,        // [num_tiles, 32] or nullptr
    int* __restrict__ found_flag,             // atomic, may be nullptr
    int* __restrict__ found_coord) {          // [2] or nullptr

  const int tile_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (tile_id >= num_tiles) return;

  // Load transcript from global buffer
  uint32_t transcript[TRANSCRIPT_U32_B3];
  const uint32_t* tb = &transcript_buffer[(size_t)tile_id * TRANSCRIPT_U32_B3];
#pragma unroll
  for (int i = 0; i < TRANSCRIPT_U32_B3; ++i) transcript[i] = tb[i];

  Tensor block = make_tensor<uint32_t>(Int<TRANSCRIPT_U32_B3>{});
#pragma unroll
  for (int i = 0; i < TRANSCRIPT_U32_B3; ++i) block(i) = transcript[i];
  Tensor cv = make_tensor<uint32_t>(Int<blake3::CHAINING_VALUE_SIZE_U32>{});
#pragma unroll
  for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) cv(i) = pow_key[i];

  blake3::compress_msg_block_u32(block, cv,
                                 blake3::COMPRESS_PARAMS_SINGLE_BLOCK_KEYED);

  if (out_digests) {
    uint32_t* od = reinterpret_cast<uint32_t*>(out_digests + (size_t)tile_id * 32);
#pragma unroll
    for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) od[i] = cv(i);
  }

  if (found_flag) {
    bool le = true;
#pragma unroll
    for (int i = blake3::CHAINING_VALUE_SIZE_U32 - 1; i >= 0; --i) {
      uint32_t h = cv(i), tg = pow_target[i];
      if (h > tg) { le = false; break; }
      if (h < tg) break;
    }
    if (le && atomicCAS(found_flag, 0, 1) == 0 && found_coord) {
      const int tiles_w = n / HT_B3;
      const int gi = (tile_id / tiles_w) * HT_B3;
      const int gj = (tile_id % tiles_w) * HT_B3;
      found_coord[0] = gi;
      found_coord[1] = gj;
    }
  }
}

void launch_pearl_blake3(
    const uint32_t* transcript_buffer, int num_tiles, int n,
    const uint32_t* pow_key, const uint32_t* pow_target,
    uint8_t* out_digests, int* found_flag, int* found_coord,
    cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (num_tiles + threads - 1) / threads;
  pearl_blake3_kernel<<<blocks, threads, 0, stream>>>(
      transcript_buffer, num_tiles, n,
      pow_key, pow_target, out_digests, found_flag, found_coord);
}
