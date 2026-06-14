#include <cuda_runtime.h>
#include <cstdint>

// Pascal (sm_61) noising kernels for the Pearl PoW pipeline.
//
// The miner adds structured low-rank noise to the (7-bit quantized) activation
// matrix A and weight matrix B, then computes the noised GEMM ApEA @ BpEB^T.
// Two int32 correction terms (AxEBL, EARxBpEB) are produced so the noise can be
// removed afterwards (the denoise step). The noise factors are produced by the
// BLAKE3 noise-generation kernel:
//   EAL  [M,R]  dense int8, values in [-32, 32)
//   EBR  [N,R]  dense int8, values in [-32, 32)
//   EAR, EBL    sparse int8 (exactly one +1 and one -1 per K position), provided
//               in both R-major [K,R] and K-major [R,K] layouts.
//
// noise_A consumes EAR_R_major [K,R] and EBL_K_major [R,K]:
//   ApEA[m,k]  = A[m,k] + sum_r EAL[m,r] * EAR_Rmaj[k,r]   (int8; fits because
//                A in [-63,63] and the noise term is in (-64,64))
//   AxEBL[m,r] = sum_k A[m,k] * EBL_Kmaj[r,k]              (int32, uses clean A)
//
// noise_B consumes EBL_R_major [K,R] and EAR_K_major [R,K]:
//   BpEB[n,k]     = B[n,k] + sum_r EBR[n,r] * EBL_Rmaj[k,r]      (int8)
//   EARxBpEB[n,r] = sum_k BpEB[n,k] * EAR_Kmaj[r,k]              (int32, noised B)
//
// One block processes one row (m or n). The row of A/B and its dense noise row
// are staged in dynamic shared memory; correctness is the priority over peak
// throughput here.

namespace {

inline __device__ int8_t clamp_to_int8(int v) {
  if (v > 127) v = 127;
  if (v < -128) v = -128;
  return (int8_t)v;
}

}  // namespace

// EAR is EAR_R_major [K, R]; EBL is EBL_K_major [R, K].
__global__ void noise_A_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ EAL,
    const int8_t* __restrict__ EAR,
    const int8_t* __restrict__ EBL,
    int8_t* __restrict__ ApEA,
    int32_t* __restrict__ AxEBL,
    int M, int K, int R) {

  const int m = blockIdx.x;
  if (m >= M) return;

  extern __shared__ int8_t smem[];
  int8_t* sA = smem;        // K bytes: A[m, :]
  int8_t* sEAL = smem + K;  // R bytes: EAL[m, :]

  for (int k = threadIdx.x; k < K; k += blockDim.x) sA[k] = A[m * K + k];
  for (int r = threadIdx.x; r < R; r += blockDim.x) sEAL[r] = EAL[m * R + r];
  __syncthreads();

  // ApEA[m,k] = A[m,k] + sum_r EAL[m,r] * EAR_Rmaj[k,r]
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    int acc = 0;
    const int8_t* ear_row = &EAR[k * R];  // EAR_R_major: row k, R cols
    for (int r = 0; r < R; ++r) acc += (int)sEAL[r] * (int)ear_row[r];
    ApEA[m * K + k] = clamp_to_int8((int)sA[k] + acc);
  }

  // AxEBL[m,r] = sum_k A[m,k] * EBL_Kmaj[r,k]   (clean A)
  for (int r = threadIdx.x; r < R; r += blockDim.x) {
    int acc = 0;
    const int8_t* ebl_row = &EBL[r * K];  // EBL_K_major: row r, K cols
    for (int k = 0; k < K; ++k) acc += (int)sA[k] * (int)ebl_row[k];
    AxEBL[m * R + r] = acc;
  }
}

// EBL is EBL_R_major [K, R]; EAR is EAR_K_major [R, K].
__global__ void noise_B_kernel(
    const int8_t* __restrict__ B,
    const int8_t* __restrict__ EBR,
    const int8_t* __restrict__ EAR,
    const int8_t* __restrict__ EBL,
    int8_t* __restrict__ BpEB,
    int32_t* __restrict__ EARxBpEB,
    int N, int K, int R) {

  const int n = blockIdx.x;
  if (n >= N) return;

  extern __shared__ int8_t smem[];
  int8_t* sBpEB = smem;       // K bytes: B[n, :], overwritten with BpEB[n, :]
  int8_t* sEBR = smem + K;    // R bytes: EBR[n, :]

  for (int k = threadIdx.x; k < K; k += blockDim.x) sBpEB[k] = B[n * K + k];
  for (int r = threadIdx.x; r < R; r += blockDim.x) sEBR[r] = EBR[n * R + r];
  __syncthreads();

  // BpEB[n,k] = B[n,k] + sum_r EBR[n,r] * EBL_Rmaj[k,r]
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    int acc = 0;
    const int8_t* ebl_row = &EBL[k * R];  // EBL_R_major: row k, R cols
    for (int r = 0; r < R; ++r) acc += (int)sEBR[r] * (int)ebl_row[r];
    int8_t v = clamp_to_int8((int)sBpEB[k] + acc);
    sBpEB[k] = v;              // reuse shared row as BpEB for the next stage
    BpEB[n * K + k] = v;
  }
  __syncthreads();

  // EARxBpEB[n,r] = sum_k BpEB[n,k] * EAR_Kmaj[r,k]   (noised B)
  for (int r = threadIdx.x; r < R; r += blockDim.x) {
    int acc = 0;
    const int8_t* ear_row = &EAR[r * K];  // EAR_K_major: row r, K cols
    for (int k = 0; k < K; ++k) acc += (int)sBpEB[k] * (int)ear_row[k];
    EARxBpEB[n * R + r] = acc;
  }
}

static int noise_block_threads(int K, int R) {
  int want = K > R ? K : R;
  int t = 256;
  while (t < want && t < 1024) t <<= 1;
  return t;
}

void launch_noise_A(
    const int8_t* A, const int8_t* EAL,
    const int8_t* EAR, const int8_t* EBL,
    int8_t* ApEA, int32_t* AxEBL,
    int M, int K, int R,
    cudaStream_t stream) {

  dim3 block(noise_block_threads(K, R));
  dim3 grid(M);
  size_t smem = (size_t)(K + R) * sizeof(int8_t);
  noise_A_kernel<<<grid, block, smem, stream>>>(
      A, EAL, EAR, EBL, ApEA, AxEBL, M, K, R);
}

void launch_noise_B(
    const int8_t* B, const int8_t* EBR,
    const int8_t* EAR, const int8_t* EBL,
    int8_t* BpEB, int32_t* EARxBpEB,
    int N, int K, int R,
    cudaStream_t stream) {

  dim3 block(noise_block_threads(K, R));
  dim3 grid(N);
  size_t smem = (size_t)(K + R) * sizeof(int8_t);
  noise_B_kernel<<<grid, block, smem, stream>>>(
      B, EBR, EAR, EBL, BpEB, EARxBpEB, N, K, R);
}
