#include <cstdint>
#include <cuda_runtime.h>

// Philox-4x32-10 constants
static constexpr __device__ uint32_t PHILOX_M4x32_0 = 0xD2511F53;
static constexpr __device__ uint32_t PHILOX_M4x32_1 = 0xCD9E8D57;
static constexpr __device__ uint32_t PHILOX_KEY_ROUND = 0x9E3779B9;

__device__ __forceinline__ uint2 philox_round(uint4& ctr, uint2 key) {
    uint64_t p0 = (uint64_t)ctr.x * PHILOX_M4x32_0;
    uint64_t p1 = (uint64_t)ctr.z * PHILOX_M4x32_1;
    uint32_t lo0 = (uint32_t)p0, hi0 = (uint32_t)(p0 >> 32);
    uint32_t lo1 = (uint32_t)p1, hi1 = (uint32_t)(p1 >> 32);
    ctr = make_uint4(hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0);
    return key;
}

__device__ __forceinline__ uint2 philox_key_round(uint2 key) {
    uint64_t p = (uint64_t)key.x * PHILOX_KEY_ROUND;
    return make_uint2((uint32_t)(p >> 32) ^ key.y ^ PHILOX_KEY_ROUND, (uint32_t)p);
}

// Fill buffer with random int8 in [-64, 63] using Philox-4x32-10.
// Each thread generates 16 int8 values from 4 uint32 Philox outputs.
// Grid: ceil(numel / (blockDim.x * 16))
__global__ void fill_rand_i8_kernel(int8_t* out, int64_t numel, uint64_t seed) {
    int64_t base = (int64_t)blockIdx.x * blockDim.x * 16 + (int64_t)threadIdx.x * 16;
    if (base >= numel) return;

    uint4 ctr[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int64_t idx = base + i * 4;
        ctr[i] = make_uint4(
            (uint32_t)(idx & 0xFFFFFFFF),
            (uint32_t)((idx >> 32) & 0xFFFFFFFF),
            (uint32_t)((idx >> 32) & 0xFFFFFFFF) ^ (uint32_t)(idx & 0xFFFFFFFF),
            0
        );
    }

    uint2 key = make_uint2((uint32_t)(seed & 0xFFFFFFFF), (uint32_t)(seed >> 32));

    #pragma unroll
    for (int r = 0; r < 10; ++r) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            key = philox_round(ctr[i], key);
        }
        key = philox_key_round(key);
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int64_t idx = base + i * 4;
        uint32_t v0 = ctr[i].x, v1 = ctr[i].y, v2 = ctr[i].z, v3 = ctr[i].w;
        if (idx < numel) out[idx] = (int8_t)((v0 & 0x7F) - 64);
        if (idx + 1 < numel) out[idx + 1] = (int8_t)((v1 & 0x7F) - 64);
        if (idx + 2 < numel) out[idx + 2] = (int8_t)((v2 & 0x7F) - 64);
        if (idx + 3 < numel) out[idx + 3] = (int8_t)((v3 & 0x7F) - 64);
    }
}

// Transpose int8 matrix: src [rows, cols] → dst [cols, rows].
// 32×32 shared-memory tile, 1024 threads/block.
__global__ void transpose_i8_kernel(const int8_t* src, int8_t* dst, int rows, int cols) {
    __shared__ int8_t tile[32][33];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = src[(int64_t)y * cols + x];
    }
    __syncthreads();
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    if (x < rows && y < cols) {
        dst[(int64_t)y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

extern "C" void launch_fill_rand_i8(int8_t* out, int64_t numel, uint64_t seed, cudaStream_t stream) {
    const int threads = 128;
    const int elems_per_thread = 16;
    int64_t blocks = (numel + threads * elems_per_thread - 1) / (threads * elems_per_thread);
    fill_rand_i8_kernel<<<(int)blocks, threads, 0, stream>>>(out, numel, seed);
}

extern "C" void launch_transpose_i8(const int8_t* src, int8_t* dst, int rows, int cols, cudaStream_t stream) {
    dim3 block(32, 32);
    dim3 grid((cols + 31) / 32, (rows + 31) / 32);
    transpose_i8_kernel<<<grid, block, 0, stream>>>(src, dst, rows, cols);
}
