// Pascal (sm_61) Pearl PoW kernel.
//
// Reproduces the proof-of-work core of the reference miner (miner-base
// noisy_gemm.py `_tiled_matmul` + `_check_pow_target`) for Pascal GPUs, using
// DP4A for the int8 contraction instead of Hopper tensor cores.
//
// For the *noised* operands A (m x k) and B^T (n x k), both int8, the PoW is
// computed independently per 16x16 output ("hash") tile:
//   - Tile the k dimension by R = noise_rank. For each full k-tile t = 0..k/R-1:
//       Csum += A[tile, t*R:(t+1)*R] @ Bt[tile, t*R:(t+1)*R]^T   (cumulative int32)
//       h    = XOR over the 256 int32 of the *cumulative* Csum (as uint32)
//       transcript[t % 16] = rotl32(transcript[t % 16], 13) ^ h
//   - digest = BLAKE3(transcript[16 x u32, little-endian], key = pow_key)
//   - the tile "wins" if digest <= pow_target (uint256, little-endian).
//
// XOR is associative/commutative so the reduction order is irrelevant; this is
// bit-exact with the reference's lop3 XOR tree. The keyed BLAKE3 of the 64-byte
// transcript is a single keyed block (CHUNK_START|CHUNK_END|ROOT) — identical to
// `blake3.blake3(transcript_bytes, key=pow_key)` and to pow_utils::check_pow_target.
//
// Assumes m % 16 == 0, n % 16 == 0, k % R == 0 (the 128-aligned shapes the miner
// uses). Partial edge tiles / partial k-tiles do not contribute to the PoW in the
// reference and are out of scope here.

#include <cuda_runtime.h>
#include <cstdint>

#include <cute/tensor.hpp>
#include "blake3/blake3.cuh"

using namespace cute;

static __device__ __forceinline__ int dp4a_pow(int a, int b, int c) {
  int r;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
  asm volatile("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else
  r = c;
  for (int i = 0; i < 4; ++i)
    r += int((int8_t)((a >> (i * 8)) & 0xFF)) * int((int8_t)((b >> (i * 8)) & 0xFF));
#endif
  return r;
}

static __device__ __forceinline__ uint32_t rotl32_dev(uint32_t x, int n) {
  return (x << n) | (x >> (32 - n));
}

static constexpr int HT = 16;             // hash tile dimension
static constexpr int HASH_ROT = 13;       // HASH_ACCUMULATE_ROTATION
static constexpr int TRANSCRIPT_U32 = 16;

template <int R>
__global__ void pearl_pow_kernel(
    const int8_t* __restrict__ A,    // [m, k] noised
    const int8_t* __restrict__ Bt,   // [n, k] noised (B transposed)
    int n, int k,
    const uint32_t* __restrict__ pow_key,     // 8 words
    const uint32_t* __restrict__ pow_target,  // 8 words, little-endian (word0 = LSW)
    uint8_t* __restrict__ out_digests,        // [num_tiles, 32] or nullptr
    int* __restrict__ found_flag,             // atomic, may be nullptr
    int* __restrict__ found_coord) {          // [2] = (row, col) of winning tile, or nullptr

  const int tiles_w = n / HT;
  const int tile_id = blockIdx.x;
  const int i0 = (tile_id / tiles_w) * HT;
  const int j0 = (tile_id % tiles_w) * HT;

  const int tid = threadIdx.x;       // 0..255
  const int di = tid / HT;
  const int dj = tid % HT;

  __shared__ int8_t sA[HT * R];
  __shared__ int8_t sB[HT * R];
  __shared__ uint32_t sRed[HT * HT];
  __shared__ uint32_t sTranscript[TRANSCRIPT_U32];

  if (tid < TRANSCRIPT_U32) sTranscript[tid] = 0u;

  int csum = 0;
  const int T = k / R;
  for (int t = 0; t < T; ++t) {
    const int p = t * R;
    __syncthreads();
    for (int idx = tid; idx < HT * R; idx += blockDim.x) {
      const int r = idx / R;
      const int c = idx % R;
      sA[idx] = A[(i0 + r) * k + (p + c)];
      sB[idx] = Bt[(j0 + r) * k + (p + c)];
    }
    __syncthreads();

    int part = 0;
#pragma unroll
    for (int kk = 0; kk < R; kk += 4) {
      int a4 = *reinterpret_cast<const int*>(&sA[di * R + kk]);
      int b4 = *reinterpret_cast<const int*>(&sB[dj * R + kk]);
      part = dp4a_pow(a4, b4, part);
    }
    csum += part;

    // XOR-reduce the cumulative 16x16 tile across all 256 threads.
    sRed[tid] = (uint32_t)csum;
    __syncthreads();
#pragma unroll
    for (int s = (HT * HT) / 2; s > 0; s >>= 1) {
      if (tid < s) sRed[tid] ^= sRed[tid + s];
      __syncthreads();
    }
    if (tid == 0) {
      const int idx = t % TRANSCRIPT_U32;
      sTranscript[idx] = rotl32_dev(sTranscript[idx], HASH_ROT) ^ sRed[0];
    }
    __syncthreads();
  }

  if (tid == 0) {
    Tensor block = make_tensor<uint32_t>(Int<TRANSCRIPT_U32>{});
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < TRANSCRIPT_U32; ++i) block(i) = sTranscript[i];
    Tensor cv = make_tensor<uint32_t>(Int<blake3::CHAINING_VALUE_SIZE_U32>{});
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) cv(i) = pow_key[i];

    blake3::compress_msg_block_u32(block, cv,
                                   blake3::COMPRESS_PARAMS_SINGLE_BLOCK_KEYED);

    if (out_digests) {
      uint32_t* od = reinterpret_cast<uint32_t*>(out_digests + (size_t)tile_id * 32);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) od[i] = cv(i);
    }

    if (found_flag) {
      bool le = true;  // digest <= target ?
      CUTLASS_PRAGMA_UNROLL
      for (int i = blake3::CHAINING_VALUE_SIZE_U32 - 1; i >= 0; --i) {
        uint32_t h = cv(i), tg = pow_target[i];
        if (h > tg) { le = false; break; }
        if (h < tg) break;
      }
      if (le && atomicCAS(found_flag, 0, 1) == 0 && found_coord) {
        found_coord[0] = i0;
        found_coord[1] = j0;
      }
    }
  }
}

void launch_pearl_pow(
    const int8_t* A, const int8_t* Bt, int m, int n, int k, int R,
    const uint32_t* pow_key, const uint32_t* pow_target,
    uint8_t* out_digests, int* found_flag, int* found_coord,
    cudaStream_t stream) {
  const int num_tiles = (m / HT) * (n / HT);
  dim3 grid(num_tiles);
  dim3 block(HT * HT);  // 256
  switch (R) {
    case 256:
      pearl_pow_kernel<256><<<grid, block, 0, stream>>>(
          A, Bt, n, k, pow_key, pow_target, out_digests, found_flag, found_coord);
      break;
    case 128:
      pearl_pow_kernel<128><<<grid, block, 0, stream>>>(
          A, Bt, n, k, pow_key, pow_target, out_digests, found_flag, found_coord);
      break;
    case 64:
      pearl_pow_kernel<64><<<grid, block, 0, stream>>>(
          A, Bt, n, k, pow_key, pow_target, out_digests, found_flag, found_coord);
      break;
    default:
      break;  // unsupported R; caller validates
  }
}
