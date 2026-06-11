// Pascal (sm_61) GEMM-ONLY Pearl kernel — high-throughput, no BLAKE3.
//
// Same fused warp/tile structure as pearl_pow_fused_sm61.cu (shared-mem operand
// reuse, register-blocked micro-tiles, __shfl_xor reduction, rotl-xor transcript
// accumulation) but DOES NOT compute BLAKE3. Instead it writes the 16-word
// transcript per hash tile to a global buffer (transcript_buffer[num_tiles, 16]).
//
// Without the keyed-BLAKE3 ~60 registers the thread register footprint drops to
// ~40, letting us raise occupancy from MINB=2 (50 %) to MINB=3 (75 %).
//
// A companion kernel (pearl_blake3_sm61.cu) consumes the transcript buffer.
//
// Bit-exact transcript: the GEMM loop, XOR reduction and rotl-xor accumulation
// are identical to pearl_pow_fused_sm61.cu, so the 16-word transcript written
// to global memory is byte-identical.

#include <cuda_runtime.h>
#include <cstdint>

#include <cute/tensor.hpp>
#include "blake3/blake3.cuh"

using namespace cute;

static __device__ __forceinline__ int dp4a_go(int a, int b, int c) {
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

static __device__ __forceinline__ uint32_t rotl32_go(uint32_t x, int n) {
  return (x << n) | (x >> (32 - n));
}

static constexpr int HT_GO = 16;
static constexpr int HASH_ROT_GO = 13;
static constexpr int TRANSCRIPT_U32_GO = 16;
static constexpr int ELT_PER_LANE_GO = (HT_GO * HT_GO) / 32;  // 8

template <int R, int WM, int WN, int MINB>
__global__ void __launch_bounds__(WM* WN * 32, MINB) pearl_gemm_only_kernel(
    const int8_t* __restrict__ A,     // [m, k] noised
    const int8_t* __restrict__ Bt,    // [n, k] noised (B transposed)
    int n, int k,
    uint32_t* __restrict__ transcript_buffer)  // [num_tiles, 16] output

{
  constexpr int ROWS_A = HT_GO * WM;
  constexpr int ROWS_B = HT_GO * WN;
  constexpr int RW = R / 4;
  // Conflict-free odd stride (coprime to 32). Matches the fused kernel; the
  // int4-aligned stride (RW+4)&~3 == 68 has 68%32==4, reintroducing bank
  // conflicts that net-regress at the real k=4096 config.
  constexpr int SAW = RW + 1;

  const int tiles_w = n / HT_GO;
  const int blocks_n = tiles_w / WN;
  const int block_row = blockIdx.x / blocks_n;
  const int block_col = blockIdx.x % blocks_n;
  const int row_base = block_row * ROWS_A;
  const int col_base = block_col * ROWS_B;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int wm = warp / WN;
  const int wn = warp % WN;
  const int aRow0 = wm * HT_GO;
  const int bRow0 = wn * HT_GO;

  int acc[ELT_PER_LANE_GO];
#pragma unroll
  for (int e = 0; e < ELT_PER_LANE_GO; ++e) acc[e] = 0;
  uint32_t transcript[TRANSCRIPT_U32_GO];
#pragma unroll
  for (int e = 0; e < TRANSCRIPT_U32_GO; ++e) transcript[e] = 0u;

  __shared__ int sAi[ROWS_A * SAW];
  __shared__ int sBi[ROWS_B * SAW];
  const int* Ai = reinterpret_cast<const int*>(A);
  const int* Bi = reinterpret_cast<const int*>(Bt);

  const int T = k / R;
  for (int t = 0; t < T; ++t) {
    const int koff4 = (t * R) / 4;
    __syncthreads();
    for (int i = tid; i < ROWS_A * RW; i += blockDim.x) {
      const int r = i / RW, c4 = i % RW;
      sAi[r * SAW + c4] = Ai[(size_t)(row_base + r) * (k / 4) + koff4 + c4];
    }
    for (int i = tid; i < ROWS_B * RW; i += blockDim.x) {
      const int r = i / RW, c4 = i % RW;
      sBi[r * SAW + c4] = Bi[(size_t)(col_base + r) * (k / 4) + koff4 + c4];
    }
    __syncthreads();

    constexpr int RM = 4, RN = 2;
    const int mtr = lane >> 3;
    const int mtc = lane & 7;
    const int* ar[RM];
    const int* br[RN];
#pragma unroll
    for (int i = 0; i < RM; ++i) ar[i] = &sAi[(aRow0 + mtr * RM + i) * SAW];
#pragma unroll
    for (int j = 0; j < RN; ++j) br[j] = &sBi[(bRow0 + mtc * RN + j) * SAW];
#pragma unroll
    for (int kk = 0; kk < RW; ++kk) {
      int a[RM], b[RN];
#pragma unroll
      for (int i = 0; i < RM; ++i) a[i] = ar[i][kk];
#pragma unroll
      for (int j = 0; j < RN; ++j) b[j] = br[j][kk];
#pragma unroll
      for (int i = 0; i < RM; ++i)
#pragma unroll
        for (int j = 0; j < RN; ++j)
          acc[i * RN + j] = dp4a_go(a[i], b[j], acc[i * RN + j]);
    }
    uint32_t lx = 0u;
#pragma unroll
    for (int e = 0; e < ELT_PER_LANE_GO; ++e) lx ^= (uint32_t)acc[e];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      lx ^= __shfl_xor_sync(0xffffffffu, lx, off);
    if (lane == 0) {
      const int idx = t % TRANSCRIPT_U32_GO;
      transcript[idx] = rotl32_go(transcript[idx], HASH_ROT_GO) ^ lx;
    }
  }

  if (lane != 0) return;

  const int gi = row_base + aRow0;
  const int gj = col_base + bRow0;
  const int tile_id = (gi / HT_GO) * tiles_w + (gj / HT_GO);
  uint32_t* tb = &transcript_buffer[(size_t)tile_id * TRANSCRIPT_U32_GO];
#pragma unroll
  for (int i = 0; i < TRANSCRIPT_U32_GO; ++i) tb[i] = transcript[i];
}

template <int R, int WM, int WN, int MINB>
static void launch_go_cfg(const int8_t* A, const int8_t* Bt, int m, int n, int k,
                          uint32_t* transcript_buffer, cudaStream_t stream) {
  const int num_block_tiles = (m / (HT_GO * WM)) * (n / (HT_GO * WN));
  dim3 grid(num_block_tiles);
  dim3 block(WM * WN * 32);
  pearl_gemm_only_kernel<R, WM, WN, MINB><<<grid, block, 0, stream>>>(
      A, Bt, n, k, transcript_buffer);
}

// Variant dispatch for the GEMM-only kernel.
//   v=0 -> 4×4 MINB3  (75 % occupancy, recommended for gemm-only)
//   v=1 -> 4×4 MINB2  (50 %)
//   v=2 -> 4×4 MINB1  (25 %)
//   v=3 -> 2×2 MINB4  (100 % occupancy, less reuse)
//   v=4 -> 2×4 MINB3
void launch_pearl_gemm_only(
    const int8_t* A, const int8_t* Bt, int m, int n, int k, int R,
    uint32_t* transcript_buffer, int variant, cudaStream_t stream) {
  if (R == 256) {
    switch (variant) {
      case 0:
        launch_go_cfg<256, 4, 4, 3>(A, Bt, m, n, k, transcript_buffer, stream); break;
      case 1:
        launch_go_cfg<256, 4, 4, 2>(A, Bt, m, n, k, transcript_buffer, stream); break;
      case 2:
        launch_go_cfg<256, 4, 4, 1>(A, Bt, m, n, k, transcript_buffer, stream); break;
      case 3:
        launch_go_cfg<256, 2, 2, 4>(A, Bt, m, n, k, transcript_buffer, stream); break;
      case 4:
        launch_go_cfg<256, 2, 4, 3>(A, Bt, m, n, k, transcript_buffer, stream); break;
      default:
        launch_go_cfg<256, 4, 4, 3>(A, Bt, m, n, k, transcript_buffer, stream); break;
    }
  } else if (R == 128) {
    launch_go_cfg<128, 4, 4, 3>(A, Bt, m, n, k, transcript_buffer, stream);
  } else if (R == 64) {
    launch_go_cfg<64, 4, 4, 3>(A, Bt, m, n, k, transcript_buffer, stream);
  }
}
