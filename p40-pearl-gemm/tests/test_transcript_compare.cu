// test_transcript_compare.cu
// Compare Pascal DP4A vs Ampere TC transcripts on identical input.
// nvcc -O3 -std=c++17 -arch=sm_89 -I../.deps/cutlass/include -o test_transcript_compare test_transcript_compare.cu

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s (%d)\n", \
                __FILE__, __LINE__, cudaGetErrorString(err), err); \
        exit(1); \
    } \
} while(0)

// Include both kernels
#include "../csrc/gemm/pearl_gemm_only_sm61.cu"
#include "../csrc/gemm/pearl_ampere_tc.cu"

// Deterministic fill
__global__ void fill_det(int8_t* buf, int64_t numel, uint64_t seed) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    uint64_t s = seed + idx * 0x9E3779B97F4A7C15ULL;
    s = s * 0xD2511F53CD9E8D57ULL;
    s ^= s >> 31;
    s *= 0x9E3779B9;
    buf[idx] = (int8_t)((s & 0xFF) - 128);
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    const int m = 256, n = 256, k = 256, R = 128;
    const int tiles = (m/16) * (n/16);
    const int T = k / R;  // 2

    int8_t *dA, *dBt;
    uint32_t *dT_p, *dT_a, *hT_p, *hT_a;

    size_t szA = (size_t)m * k;
    size_t szBt = (size_t)n * k;
    size_t szTr = (size_t)tiles * 16 * 4;

    CUDA_CHECK(cudaMalloc(&dA, szA));
    CUDA_CHECK(cudaMalloc(&dBt, szBt));
    CUDA_CHECK(cudaMalloc(&dT_p, szTr));
    CUDA_CHECK(cudaMalloc(&dT_a, szTr));
    hT_p = (uint32_t*)malloc(szTr);
    hT_a = (uint32_t*)malloc(szTr);

    int thr = 256;
    fill_det<<<(szA+thr-1)/thr, thr>>>(dA, szA, 0x12345678);
    fill_det<<<(szBt+thr-1)/thr, thr>>>(dBt, szBt, 0x87654321);
    CUDA_CHECK(cudaMemset(dT_p, 0, szTr));
    CUDA_CHECK(cudaMemset(dT_a, 0, szTr));
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Running Pascal DP4A kernel (R=%d)...\n", R);
    launch_pearl_gemm_only(dA, dBt, m, n, k, R, dT_p, 1, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Running Ampere TC kernel (R=%d)...\n", R);
    cudaError_t ae = launch_pearl_ampere(dA, dBt, m, n, k, R, dT_a, 0);
    if (ae != cudaSuccess) {
        fprintf(stderr, "Ampere launch failed: %s\n", cudaGetErrorString(ae));
        return 1;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hT_p, dT_p, szTr, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hT_a, dT_a, szTr, cudaMemcpyDeviceToHost));

    int diff = 0;
    for (int t = 0; t < tiles; t++) {
        for (int w = 0; w < 16; w++) {
            uint32_t a = hT_p[t * 16 + w];
            uint32_t b = hT_a[t * 16 + w];
            if (a != b) {
                if (diff < 20)
                    printf("DIFF tile=%d word=%d pascal=0x%08x ampere=0x%08x\n", t, w, a, b);
                diff++;
            }
        }
    }

    if (diff == 0) {
        printf("*** PASS: All %d transcripts MATCH bit-exact! ***\n", tiles);
    } else {
        printf("*** FAIL: %d transcript words differ ***\n", diff);
    }

    int nz_p = 0, nz_a = 0;
    for (int i = 0; i < tiles * 16; i++) {
        if (hT_p[i] != 0) nz_p++;
        if (hT_a[i] != 0) nz_a++;
    }
    printf("Pascal non-zero words: %d/%d\n", nz_p, tiles * 16);
    printf("Ampere non-zero words: %d/%d\n", nz_a, tiles * 16);

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dBt));
    CUDA_CHECK(cudaFree(dT_p));
    CUDA_CHECK(cudaFree(dT_a));
    free(hT_p);
    free(hT_a);

    return diff == 0 ? 0 : 1;
}
