#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Pascal (sm_61) INT8 GEMM using the DP4A intrinsic.
//
// Computes  C[m, n] = (sum_k A[m, k] * B[n, k]) * A_scales[m] * B_scales[n]
// with A row-major [M, K] (int8), B row-major [N, K] (int8, i.e. the transposed
// operand), and C row-major [M, N] (fp16).  The K dimension is contraction;
// DP4A consumes 4 contiguous-K int8 values per call.
//
// Tiling: each block computes a 64x64 output tile.  The block has 16x16 = 256
// threads and each thread owns a 4x4 micro-tile, so the 256 threads cover the
// full 64x64 tile (16*4 = 64 rows, 16*4 = 64 cols).

#define DP4A_TILE_M 64
#define DP4A_TILE_N 64
#define DP4A_TILE_K 64
#define DP4A_THREAD_M 4   // output rows per thread
#define DP4A_THREAD_N 4   // output cols per thread
#define DP4A_BLOCK_DIM 16 // 16x16 threads = 256

__device__ __forceinline__ int dp4a(int a, int b, int c) {
  int result;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
  asm volatile("dp4a.s32.s32 %0, %1, %2, %3;"
               : "=r"(result)
               : "r"(a), "r"(b), "r"(c));
#else
  result = c;
  for (int i = 0; i < 4; ++i) {
    int8_t ba = (a >> (i * 8)) & 0xFF;
    int8_t bb = (b >> (i * 8)) & 0xFF;
    result += int(ba) * int(bb);
  }
#endif
  return result;
}

__global__ void dp4a_gemm_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    const float* __restrict__ A_scales,
    const float* __restrict__ B_scales,
    half* __restrict__ C,
    int M, int N, int K) {

  // 16-byte alignment lets us read 4 contiguous int8 as a single 32-bit word.
  __shared__ __align__(16) int8_t smem_A[DP4A_TILE_M * DP4A_TILE_K];
  __shared__ __align__(16) int8_t smem_B[DP4A_TILE_N * DP4A_TILE_K];

  const int m_start = blockIdx.x * DP4A_TILE_M;
  const int n_start = blockIdx.y * DP4A_TILE_N;

  const int tx = threadIdx.x;  // 0..15  -> column group
  const int ty = threadIdx.y;  // 0..15  -> row group
  const int tid = ty * DP4A_BLOCK_DIM + tx;
  const int nthreads = DP4A_BLOCK_DIM * DP4A_BLOCK_DIM;  // 256

  int acc[DP4A_THREAD_M][DP4A_THREAD_N] = {0};

  for (int k_tile = 0; k_tile < K; k_tile += DP4A_TILE_K) {
    // Stage the A tile (64x64), zero-padding out-of-range elements so the
    // contraction loop can run over the full tile unconditionally.
    for (int i = tid; i < DP4A_TILE_M * DP4A_TILE_K; i += nthreads) {
      int mi = i / DP4A_TILE_K;
      int ki = i % DP4A_TILE_K;
      int gm = m_start + mi;
      int gk = k_tile + ki;
      smem_A[i] = (gm < M && gk < K) ? A[gm * K + gk] : (int8_t)0;
    }
    // Stage the B tile (64x64).  B is [N, K] row-major.
    for (int i = tid; i < DP4A_TILE_N * DP4A_TILE_K; i += nthreads) {
      int ni = i / DP4A_TILE_K;
      int ki = i % DP4A_TILE_K;
      int gn = n_start + ni;
      int gk = k_tile + ki;
      smem_B[i] = (gn < N && gk < K) ? B[gn * K + gk] : (int8_t)0;
    }
    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < DP4A_TILE_K; kk += 4) {
      int a_vals[DP4A_THREAD_M];
      int b_vals[DP4A_THREAD_N];

      #pragma unroll
      for (int i = 0; i < DP4A_THREAD_M; ++i) {
        int row = ty * DP4A_THREAD_M + i;
        a_vals[i] = *reinterpret_cast<const int*>(
            &smem_A[row * DP4A_TILE_K + kk]);
      }
      #pragma unroll
      for (int j = 0; j < DP4A_THREAD_N; ++j) {
        int col = tx * DP4A_THREAD_N + j;
        b_vals[j] = *reinterpret_cast<const int*>(
            &smem_B[col * DP4A_TILE_K + kk]);
      }

      #pragma unroll
      for (int i = 0; i < DP4A_THREAD_M; ++i) {
        #pragma unroll
        for (int j = 0; j < DP4A_THREAD_N; ++j) {
          acc[i][j] = dp4a(a_vals[i], b_vals[j], acc[i][j]);
        }
      }
    }
    __syncthreads();
  }

  #pragma unroll
  for (int i = 0; i < DP4A_THREAD_M; ++i) {
    int gm = m_start + ty * DP4A_THREAD_M + i;
    if (gm >= M) continue;
    float a_scale = A_scales[gm];
    #pragma unroll
    for (int j = 0; j < DP4A_THREAD_N; ++j) {
      int gn = n_start + tx * DP4A_THREAD_N + j;
      if (gn >= N) continue;
      float scale = a_scale * B_scales[gn];
      float result_f32 = acc[i][j] * scale;
      result_f32 = fmaxf(-65504.0f, fminf(65504.0f, result_f32));
      C[gm * N + gn] = __float2half(result_f32);
    }
  }
}

void launch_dp4a_gemm(
    const int8_t* A, const int8_t* B,
    const float* A_scales, const float* B_scales,
    half* C, int M, int N, int K,
    cudaStream_t stream) {

  dim3 block(DP4A_BLOCK_DIM, DP4A_BLOCK_DIM);
  dim3 grid(
      (M + DP4A_TILE_M - 1) / DP4A_TILE_M,
      (N + DP4A_TILE_N - 1) / DP4A_TILE_N);

  dp4a_gemm_kernel<<<grid, block, 0, stream>>>(
      A, B, A_scales, B_scales, C, M, N, K);
}
