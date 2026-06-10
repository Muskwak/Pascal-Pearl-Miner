// Standalone numerical correctness test for the Pascal DP4A GEMM.
// Builds against the compiled dp4a_gemm_sm61 object; no torch needed.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

void launch_dp4a_gemm(const int8_t* A, const int8_t* B,
                      const float* A_scales, const float* B_scales,
                      half* C, int M, int N, int K, cudaStream_t stream);

static bool run_case(int M, int N, int K) {
  std::vector<int8_t> hA(M * K), hB(N * K);
  std::vector<float> hAs(M), hBs(N);
  for (auto& x : hA) x = (int8_t)((rand() % 255) - 127);
  for (auto& x : hB) x = (int8_t)((rand() % 255) - 127);
  for (auto& s : hAs) s = ((rand() % 1000) / 1000.0f) * 0.01f + 0.001f;
  for (auto& s : hBs) s = ((rand() % 1000) / 1000.0f) * 0.01f + 0.001f;

  int8_t *dA, *dB; float *dAs, *dBs; half* dC;
  cudaMalloc(&dA, hA.size()); cudaMalloc(&dB, hB.size());
  cudaMalloc(&dAs, M * sizeof(float)); cudaMalloc(&dBs, N * sizeof(float));
  cudaMalloc(&dC, (size_t)M * N * sizeof(half));
  cudaMemcpy(dA, hA.data(), hA.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB.data(), hB.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dAs, hAs.data(), M * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dBs, hBs.data(), N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(dC, 0, (size_t)M * N * sizeof(half));

  launch_dp4a_gemm(dA, dB, dAs, dBs, dC, M, N, K, 0);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("  CUDA error: %s\n", cudaGetErrorString(err));
    return false;
  }

  std::vector<half> hC((size_t)M * N);
  cudaMemcpy(hC.data(), dC, hC.size() * sizeof(half), cudaMemcpyDeviceToHost);

  double max_rel = 0.0; int bad = 0;
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      long acc = 0;
      for (int k = 0; k < K; ++k)
        acc += (int)hA[m * K + k] * (int)hB[n * K + k];
      float ref = acc * hAs[m] * hBs[n];
      ref = fmaxf(-65504.0f, fminf(65504.0f, ref));
      float got = __half2float(hC[m * N + n]);
      float denom = fmaxf(1e-3f, fabsf(ref));
      float rel = fabsf(got - ref) / denom;
      if (rel > max_rel) max_rel = rel;
      if (rel > 0.02f) ++bad;  // fp16 rounding tolerance
    }
  }
  cudaFree(dA); cudaFree(dB); cudaFree(dAs); cudaFree(dBs); cudaFree(dC);
  printf("  M=%d N=%d K=%d  max_rel=%.4f  bad=%d/%d  -> %s\n",
         M, N, K, max_rel, bad, M * N, bad == 0 ? "PASS" : "FAIL");
  return bad == 0;
}

int main() {
  int dev = 0;
  if (const char* e = getenv("GEMM_TEST_DEV")) dev = atoi(e);
  cudaSetDevice(dev);
  cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
  printf("Device %d: %s (sm_%d%d)\n", dev, prop.name, prop.major, prop.minor);
  srand(1234);
  bool ok = true;
  ok &= run_case(64, 64, 64);     // single tile
  ok &= run_case(256, 256, 256);  // multi-tile, aligned
  ok &= run_case(128, 256, 64);
  ok &= run_case(100, 70, 50);    // ragged edges (not multiples of 64)
  ok &= run_case(65, 130, 4);     // minimal K
  printf("\n%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
