// Torch-free C ABI over the Pascal Pearl kernels.
//
// Compiles to a standalone shared library (p40cuda.dll / libp40cuda.so) that
// links ONLY the CUDA runtime (cudart, ~0.5 MB) — no torch, no pybind. Callers
// drive it from Python via stdlib ctypes, managing device memory with
// cuda-python (cuMemAlloc returns a device pointer that these functions accept).
//
// Every function returns a cudaError_t as int (0 == cudaSuccess).

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

#ifdef _WIN32
#define P40_API extern "C" __declspec(dllexport)
#else
#define P40_API extern "C" __attribute__((visibility("default")))
#endif

// --- kernel launchers (defined in the kernel .cu translation units) ---
void launch_noise_gen(int8_t*, int8_t*, int8_t*, int8_t*, const uint8_t*,
                      const uint8_t*, int, int, int, int, cudaStream_t);
void launch_noise_A(const int8_t*, const int8_t*, const int8_t*, const int8_t*,
                    int8_t*, int32_t*, int, int, int, cudaStream_t);
void launch_noise_B(const int8_t*, const int8_t*, const int8_t*, const int8_t*,
                    int8_t*, int32_t*, int, int, int, cudaStream_t);
void launch_pearl_gemm_only(const int8_t*, const int8_t*, int, int, int, int,
                            uint32_t*, int, cudaStream_t);
void launch_pearl_blake3(const uint32_t*, int, int, const uint32_t*,
                         const uint32_t*, uint8_t*, int*, int*, cudaStream_t);

// =========================== memory helpers ================================
P40_API int p40_malloc(void** p, size_t n) { return (int)cudaMalloc(p, n); }
P40_API int p40_free(void* p) { return (int)cudaFree(p); }
P40_API int p40_memcpy_htod(void* d, const void* h, size_t n) {
  return (int)cudaMemcpy(d, h, n, cudaMemcpyHostToDevice);
}
P40_API int p40_memcpy_dtoh(void* h, const void* d, size_t n) {
  return (int)cudaMemcpy(h, d, n, cudaMemcpyDeviceToHost);
}
P40_API int p40_memset(void* d, int v, size_t n) { return (int)cudaMemset(d, v, n); }
P40_API int p40_sync(void) { return (int)cudaDeviceSynchronize(); }

// =========================== kernels =======================================

// Generate the four noise operands. EAR is K-major [R,k], EBL is R-major [k,R]
// (the corrected layout mapping, matching the torch noise_gen binding).
P40_API int p40_noise_gen(void* EAL, void* EAR, void* EBL, void* EBR,
                          const void* key_A, const void* key_B,
                          int m, int n, int k, int R) {
  launch_noise_gen((int8_t*)EAL, (int8_t*)EAR, (int8_t*)EBL, (int8_t*)EBR,
                   (const uint8_t*)key_A, (const uint8_t*)key_B, m, n, k, R, 0);
  return (int)cudaGetLastError();
}

// A_ns = A + round(EAL @ EAR), int8 (ApEA). AxEBL is the int32 side-product.
P40_API int p40_noise_apply_A(const void* A, const void* EAL, const void* EAR,
                              const void* EBL, void* ApEA, void* AxEBL,
                              int M, int K, int R) {
  launch_noise_A((const int8_t*)A, (const int8_t*)EAL, (const int8_t*)EAR,
                 (const int8_t*)EBL, (int8_t*)ApEA, (int32_t*)AxEBL, M, K, R, 0);
  return (int)cudaGetLastError();
}

P40_API int p40_noise_apply_B(const void* B, const void* EBR, const void* EAR,
                              const void* EBL, void* BpEB, void* EARxBpEB,
                              int N, int K, int R) {
  launch_noise_B((const int8_t*)B, (const int8_t*)EBR, (const int8_t*)EAR,
                 (const int8_t*)EBL, (int8_t*)BpEB, (int32_t*)EARxBpEB, N, K, R, 0);
  return (int)cudaGetLastError();
}

// Two-step search: GEMM-only -> transcript buffer (allocated internally) ->
// BLAKE3. digests[(m/16)*(n/16), 32], found[1], coord[2] are caller-allocated.
P40_API int p40_pearl_pow_split(const void* A, const void* Bt, int m, int n,
                                int k, int R, const void* key, const void* target,
                                void* digests, void* found, void* coord,
                                int variant) {
  const int num_tiles = (m / 16) * (n / 16);
  uint32_t* tb = nullptr;
  cudaError_t e = cudaMalloc((void**)&tb, (size_t)num_tiles * 16 * sizeof(uint32_t));
  if (e != cudaSuccess) return (int)e;
  launch_pearl_gemm_only((const int8_t*)A, (const int8_t*)Bt, m, n, k, R, tb,
                         variant, 0);
  launch_pearl_blake3(tb, num_tiles, n, (const uint32_t*)key,
                      (const uint32_t*)target, (uint8_t*)digests, (int*)found,
                      (int*)coord, 0);
  e = cudaGetLastError();
  cudaFree(tb);
  return (int)e;
}
