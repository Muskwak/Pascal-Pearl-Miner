// Standalone correctness test for the Pascal noising kernels.
// Validates noise_A / noise_B against a CPU reference of the exact integer math.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

void launch_noise_A(const int8_t* A, const int8_t* EAL, const int8_t* EAR,
                    const int8_t* EBL, int8_t* ApEA, int32_t* AxEBL,
                    int M, int K, int R, cudaStream_t stream);
void launch_noise_B(const int8_t* B, const int8_t* EBR, const int8_t* EAR,
                    const int8_t* EBL, int8_t* BpEB, int32_t* EARxBpEB,
                    int N, int K, int R, cudaStream_t stream);

static int8_t q7() { return (int8_t)((rand() % 127) - 63); }          // [-63,63]
static int8_t dense() { return (int8_t)((rand() % 64) - 32); }        // [-32,32)

// Build a sparse ±1 matrix with one +1 and one -1 per K position, returning
// both R-major [K,R] and K-major [R,K] layouts of the SAME matrix.
static void make_sparse(int K, int R, std::vector<int8_t>& rmaj,
                        std::vector<int8_t>& kmaj) {
  rmaj.assign((size_t)K * R, 0);
  kmaj.assign((size_t)R * K, 0);
  for (int k = 0; k < K; ++k) {
    int r0 = rand() % R;
    int r1 = rand() % R;
    if (r1 == r0) r1 = (r0 + 1) % R;
    rmaj[(size_t)k * R + r0] = 1;  rmaj[(size_t)k * R + r1] = -1;
    kmaj[(size_t)r0 * K + k] = 1;  kmaj[(size_t)r1 * K + k] = -1;
  }
}

template <typename T>
static T* to_dev(const std::vector<T>& h) {
  T* d; cudaMalloc(&d, h.size() * sizeof(T));
  cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}

static bool check_A(int M, int K, int R) {
  std::vector<int8_t> A(M * K), EAL(M * R);
  for (auto& x : A) x = q7();
  for (auto& x : EAL) x = dense();
  std::vector<int8_t> EAR_r, EAR_k, EBL_r, EBL_k;
  make_sparse(K, R, EAR_r, EAR_k);  // noise_A uses EAR_R_major
  make_sparse(K, R, EBL_r, EBL_k);  // noise_A uses EBL_K_major

  int8_t *dA = to_dev(A), *dEAL = to_dev(EAL);
  int8_t *dEAR = to_dev(EAR_r), *dEBL = to_dev(EBL_k);
  int8_t* dApEA; cudaMalloc(&dApEA, (size_t)M * K);
  int32_t* dAxEBL; cudaMalloc(&dAxEBL, (size_t)M * R * sizeof(int32_t));

  launch_noise_A(dA, dEAL, dEAR, dEBL, dApEA, dAxEBL, M, K, R, 0);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) { printf("  CUDA err: %s\n", cudaGetErrorString(err)); return false; }

  std::vector<int8_t> ApEA(M * K); std::vector<int32_t> AxEBL(M * R);
  cudaMemcpy(ApEA.data(), dApEA, ApEA.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(AxEBL.data(), dAxEBL, AxEBL.size() * 4, cudaMemcpyDeviceToHost);

  int bad = 0;
  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K; ++k) {
      int acc = 0;
      for (int r = 0; r < R; ++r) acc += (int)EAL[m * R + r] * (int)EAR_r[(size_t)k * R + r];
      int ref = (int)A[m * K + k] + acc;
      if (ref > 127) ref = 127; if (ref < -128) ref = -128;
      if ((int)ApEA[m * K + k] != ref) ++bad;
    }
    for (int r = 0; r < R; ++r) {
      long acc = 0;
      for (int k = 0; k < K; ++k) acc += (int)A[m * K + k] * (int)EBL_k[(size_t)r * K + k];
      if (AxEBL[m * R + r] != (int32_t)acc) ++bad;
    }
  }
  cudaFree(dA); cudaFree(dEAL); cudaFree(dEAR); cudaFree(dEBL); cudaFree(dApEA); cudaFree(dAxEBL);
  printf("  noise_A M=%d K=%d R=%d  bad=%d  -> %s\n", M, K, R, bad, bad ? "FAIL" : "PASS");
  return bad == 0;
}

static bool check_B(int N, int K, int R) {
  std::vector<int8_t> B(N * K), EBR(N * R);
  for (auto& x : B) x = q7();
  for (auto& x : EBR) x = dense();
  std::vector<int8_t> EBL_r, EBL_k, EAR_r, EAR_k;
  make_sparse(K, R, EBL_r, EBL_k);  // noise_B uses EBL_R_major
  make_sparse(K, R, EAR_r, EAR_k);  // noise_B uses EAR_K_major

  int8_t *dB = to_dev(B), *dEBR = to_dev(EBR);
  int8_t *dEAR = to_dev(EAR_k), *dEBL = to_dev(EBL_r);
  int8_t* dBpEB; cudaMalloc(&dBpEB, (size_t)N * K);
  int32_t* dEARxBpEB; cudaMalloc(&dEARxBpEB, (size_t)N * R * sizeof(int32_t));

  launch_noise_B(dB, dEBR, dEAR, dEBL, dBpEB, dEARxBpEB, N, K, R, 0);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) { printf("  CUDA err: %s\n", cudaGetErrorString(err)); return false; }

  std::vector<int8_t> BpEB(N * K); std::vector<int32_t> EARxBpEB(N * R);
  cudaMemcpy(BpEB.data(), dBpEB, BpEB.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(EARxBpEB.data(), dEARxBpEB, EARxBpEB.size() * 4, cudaMemcpyDeviceToHost);

  int bad = 0;
  for (int n = 0; n < N; ++n) {
    std::vector<int> bpeb(K);
    for (int k = 0; k < K; ++k) {
      int acc = 0;
      for (int r = 0; r < R; ++r) acc += (int)EBR[n * R + r] * (int)EBL_r[(size_t)k * R + r];
      int ref = (int)B[n * K + k] + acc;
      if (ref > 127) ref = 127; if (ref < -128) ref = -128;
      bpeb[k] = ref;
      if ((int)BpEB[n * K + k] != ref) ++bad;
    }
    for (int r = 0; r < R; ++r) {
      long acc = 0;
      for (int k = 0; k < K; ++k) acc += bpeb[k] * (int)EAR_k[(size_t)r * K + k];
      if (EARxBpEB[n * R + r] != (int32_t)acc) ++bad;
    }
  }
  cudaFree(dB); cudaFree(dEBR); cudaFree(dEAR); cudaFree(dEBL); cudaFree(dBpEB); cudaFree(dEARxBpEB);
  printf("  noise_B N=%d K=%d R=%d  bad=%d  -> %s\n", N, K, R, bad, bad ? "FAIL" : "PASS");
  return bad == 0;
}

int main() {
  int dev = 0;
  if (const char* e = getenv("GEMM_TEST_DEV")) dev = atoi(e);
  cudaSetDevice(dev);
  cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
  printf("Device %d: %s (sm_%d%d)\n", dev, prop.name, prop.major, prop.minor);
  srand(99);
  bool ok = true;
  ok &= check_A(128, 256, 64);
  ok &= check_A(100, 130, 64);
  ok &= check_B(128, 256, 64);
  ok &= check_B(77, 200, 64);
  printf("\n%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
