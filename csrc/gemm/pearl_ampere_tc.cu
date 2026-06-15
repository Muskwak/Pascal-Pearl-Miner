// pearl_ampere_tc.cu — Production Ampere (sm_80+) Pearl GEMM kernel
// Bit-exact transcript with Pascal sm61 dp4a kernel.
// Uses direct register packing (NO ldmatrix) — CuTe ALayout/BLayout bit
// decomposition for m16n8k32 int8 tensor-core.
//
// Target any sm_80+ GPU (compile with -arch=sm_86, sm_89, sm_90, etc.).
// Compile: nvcc -arch=sm_89 -O3 -std=c++17 -c pearl_ampere_tc.cu

#include <cuda_runtime.h>
#include <cstdint>

// ==================================================================
// PTX helper functions — device-only (use asm which requires sm_80+)
// ==================================================================
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

__device__ __forceinline__ void mma_m16n8k32(
    int32_t d[4], const uint32_t a[4], const uint32_t b[2], const int32_t c[4])
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3])
    );
}

__device__ __forceinline__ void cp_async_16B(void* smem, const void* gmem) {
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)), "l"(gmem)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

template <int n>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;" :: "n"(n));
}

// ====== HARDWARE REGISTER-TO-MATRIX MAPPING FOR m16n8k32.s8 ======
//
// Each warp runs 32 threads, split into 8 groups of 4 by tid_y:
//   tid_x = lane & 3   (0..3)
//   tid_y = lane >> 2  (0..7)
//
// --- A matrix (16x32 int8, row-major smem_A[row][k]) ---
// Each thread holds 16 int8 values (2 rows x 8 k) in 4 uint32 regs:
//   a[0]: row=tid_y,       k=tid_x*4 .. tid_x*4+3   (4 consecutive k)
//   a[1]: row=tid_y+8,     k=tid_x*4 .. tid_x*4+3
//   a[2]: row=tid_y,       k=tid_x*4+16 .. tid_x*4+19
//   a[3]: row=tid_y+8,     k=tid_x*4+16 .. tid_x*4+19
// 4 threads (tid_x=0..3) cover all 32 k per row.
// 8 groups (tid_y=0..7) cover all 16 rows.
//
// --- B matrix (32x8 int8, column-major smem_B[col][k]) ---
// Each thread holds 8 int8 values (1 col x 8 k) in 2 uint32 regs:
//   b[0]: col=tid_y,     k=tid_x*4 .. tid_x*4+3
//   b[1]: col=tid_y,     k=tid_x*4+16 .. tid_x*4+19
// 4 threads (tid_x=0..3) cover all 32 k per column.
// 8 groups (tid_y=0..7) cover all 8 columns.
//
// --- D accumulator (16x8 int32) ---
//   d[0] = D[tid_y][tid_x*2]
//   d[1] = D[tid_y][tid_x*2+1]
//   d[2] = D[tid_y+8][tid_x*2]
//   d[3] = D[tid_y+8][tid_x*2+1]

__device__ __forceinline__ void load_A_frag_m16n8k32(
    uint32_t a[4], const int8_t* smem_A, int BLOCK_K)
{
    const int lane = threadIdx.x & 31;
    const int tid_x = lane & 3;
    const int tid_y = lane >> 2;

    const int base_k = tid_x * 4;

    auto ld = [&](int row, int k) -> uint32_t {
        return (uint32_t)(uint8_t)smem_A[row * BLOCK_K + k];
    };

    a[0] = ld(tid_y,     base_k + 0) |
          (ld(tid_y,     base_k + 1) << 8) |
          (ld(tid_y,     base_k + 2) << 16) |
          (ld(tid_y,     base_k + 3) << 24);

    a[1] = ld(tid_y + 8, base_k + 0) |
          (ld(tid_y + 8, base_k + 1) << 8) |
          (ld(tid_y + 8, base_k + 2) << 16) |
          (ld(tid_y + 8, base_k + 3) << 24);

    a[2] = ld(tid_y,     base_k + 16) |
          (ld(tid_y,     base_k + 17) << 8) |
          (ld(tid_y,     base_k + 18) << 16) |
          (ld(tid_y,     base_k + 19) << 24);

    a[3] = ld(tid_y + 8, base_k + 16) |
          (ld(tid_y + 8, base_k + 17) << 8) |
          (ld(tid_y + 8, base_k + 18) << 16) |
          (ld(tid_y + 8, base_k + 19) << 24);
}

__device__ __forceinline__ void load_B_frag_m16n8k32(
    uint32_t b[2], const int8_t* smem_B, int BLOCK_K)
{
    const int lane = threadIdx.x & 31;
    const int tid_x = lane & 3;
    const int tid_y = lane >> 2;

    // B is column-major in shared memory: smem_B[col * BLOCK_K + row]
    const int col = tid_y;

    auto ld = [&](int k) -> uint32_t {
        return (uint32_t)(uint8_t)smem_B[col * BLOCK_K + k];
    };

    b[0] = ld(tid_x * 4 + 0) |
          (ld(tid_x * 4 + 1) << 8) |
          (ld(tid_x * 4 + 2) << 16) |
          (ld(tid_x * 4 + 3) << 24);

    b[1] = ld(tid_x * 4 + 16) |
          (ld(tid_x * 4 + 17) << 8) |
          (ld(tid_x * 4 + 18) << 16) |
          (ld(tid_x * 4 + 19) << 24);
}

__device__ __forceinline__ void extract_D_m16n8k32(
    int32_t acc[4], int32_t* smem_D, int ldm)
{
    const int lane = threadIdx.x & 31;
    const int tid_x = lane & 3;
    const int tid_y = lane >> 2;

    smem_D[tid_y * ldm + tid_x * 2]         = acc[0];
    smem_D[tid_y * ldm + tid_x * 2 + 1]     = acc[1];
    smem_D[(tid_y + 8) * ldm + tid_x * 2]   = acc[2];
    smem_D[(tid_y + 8) * ldm + tid_x * 2 + 1] = acc[3];
}

#endif // __CUDA_ARCH__ >= 800

// ==================================================================
// Constants (identical to sm61 kernel)
// ==================================================================
static constexpr int HT              = 16;
static constexpr int HASH_ROT        = 13;
static constexpr int TRANSCRIPT_LEN  = 16;
static constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 32;

// ==================================================================
// Kernel template — visible to host for <<<>>> launch syntax
// Body uses #if __CUDA_ARCH__ to guard PTX calls.
// Guarded by PEARL_UNIT_TEST — unit tests only need the helpers above.
// ==================================================================

#ifndef PEARL_UNIT_TEST

template <int BLOCK_M, int BLOCK_N, int BLOCK_K,
          int WARPS_M, int WARPS_N, int STAGES, int MINB>
__global__ void __launch_bounds__(WARPS_M * WARPS_N * 32, MINB)
pearl_ampere_fused_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ Bt,
    int n, int k, int R,
    uint32_t* __restrict__ transcript_buffer)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    static_assert(BLOCK_M == WARPS_M * 16, "BLOCK_M must equal WARPS_M * 16");
    static_assert(BLOCK_N == WARPS_N * 16, "BLOCK_N must equal WARPS_N * 16");
    static_assert(BLOCK_K == 32, "BLOCK_K must equal 32 for m16n8k32 with current loaders");
    (void)R;

    const int tid      = threadIdx.x;
    const int warp     = tid >> 5;
    const int lane     = tid & 31;
    const int warp_m   = warp / WARPS_N;
    const int warp_n   = warp % WARPS_N;

    const int tiles_w  = n / HT;
    const int blocks_n = tiles_w / WARPS_N;
    const int block_row = blockIdx.x / blocks_n;
    const int block_col = blockIdx.x % blocks_n;
    const int row_base  = block_row * BLOCK_M;
    const int col_base  = block_col * BLOCK_N;
    const int warp_row0 = row_base + warp_m * HT;
    const int warp_col0 = col_base + warp_n * HT;

    constexpr int SMEM_A = BLOCK_M * BLOCK_K;
    constexpr int SMEM_B = BLOCK_N * BLOCK_K;
    constexpr int SMEM_STAGE = SMEM_A + SMEM_B;

    __shared__ __align__(16) int8_t smem_pipe[STAGES * SMEM_STAGE];
    __shared__ __align__(16) uint32_t sT[WARPS_M * WARPS_N][TRANSCRIPT_LEN];
    __shared__ __align__(16) int32_t tile_buf[WARPS_M * WARPS_N][16][16];

    if (lane == 0) {
        #pragma unroll
        for (int i = 0; i < TRANSCRIPT_LEN; ++i) sT[warp][i] = 0;
    }

    int32_t accL[4] = {0,0,0,0};
    int32_t accR[4] = {0,0,0,0};

    const int T       = k / R;
    const int INNER_K = R / BLOCK_K;

    for (int t = 0; t < T; ++t) {

        for (int step = 0; step < INNER_K + STAGES - 1; ++step) {

            if (step < INNER_K) {
                const int k_off = t * R + step * BLOCK_K;
                const int stg   = step % STAGES;
                int8_t* smem_A_stg = &smem_pipe[stg * SMEM_STAGE];
                int8_t* smem_B_stg = &smem_pipe[stg * SMEM_STAGE + SMEM_A];

                for (int i = tid * 16; i < SMEM_A; i += blockDim.x * 16) {
                    const int row = i / BLOCK_K;
                    const int col = i % BLOCK_K;
                    cp_async_16B(&smem_A_stg[i],
                                 &A[(size_t)(row_base + row) * k + k_off + col]);
                }
                for (int i = tid * 16; i < SMEM_B; i += blockDim.x * 16) {
                    const int col = i / BLOCK_K;
                    const int row = i % BLOCK_K;
                    cp_async_16B(&smem_B_stg[i],
                                 &Bt[(size_t)(col_base + col) * k + k_off + row]);
                }
                cp_async_commit();
            }

            if (step >= STAGES - 1) {
                const int comp_stage = (step - (STAGES - 1)) % STAGES;
                cp_async_wait_group<STAGES - 2>();
                __syncthreads();

                const int8_t* smem_A_stage = &smem_pipe[comp_stage * SMEM_STAGE];
                const int8_t* smem_B_stage = &smem_pipe[comp_stage * SMEM_STAGE + SMEM_A];

                // Left half (cols 0-7)
                {
                    uint32_t a_frag[4];
                    uint32_t b_frag[2];
                    load_A_frag_m16n8k32(a_frag,
                        &smem_A_stage[warp_m * 16 * BLOCK_K], BLOCK_K);
                    load_B_frag_m16n8k32(b_frag,
                        &smem_B_stage[warp_n * 16 * BLOCK_K], BLOCK_K);
                    mma_m16n8k32(accL, a_frag, b_frag, accL);
                }

                // Right half (cols 8-15)
                {
                    uint32_t a_frag[4];
                    uint32_t b_frag[2];
                    load_A_frag_m16n8k32(a_frag,
                        &smem_A_stage[warp_m * 16 * BLOCK_K], BLOCK_K);
                    load_B_frag_m16n8k32(b_frag,
                        &smem_B_stage[(warp_n * 16 + 8) * BLOCK_K], BLOCK_K);
                    mma_m16n8k32(accR, a_frag, b_frag, accR);
                }

                __syncthreads();
            }
        }

        extract_D_m16n8k32(accL, &tile_buf[warp][0][0], 16);
        extract_D_m16n8k32(accR, &tile_buf[warp][0][8], 16);
        __syncthreads();

        const int mtr = lane >> 3;
        const int mtc = lane & 7;
        int32_t my_vals[8];
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 2; ++j) {
                int row = mtr * 4 + i;
                int col = mtc * 2 + j;
                my_vals[i*2 + j] = tile_buf[warp][row][col];
            }
        }

        uint32_t lx = 0;
        #pragma unroll
        for (int e = 0; e < 8; ++e) lx ^= (uint32_t)my_vals[e];

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            lx ^= __shfl_xor_sync(0xffffffffu, lx, off);

        if (lane == 0) {
            const int idx = t % TRANSCRIPT_LEN;
            sT[warp][idx] = ((sT[warp][idx] << HASH_ROT) |
                             (sT[warp][idx] >> (32 - HASH_ROT))) ^ lx;
        }
        __syncthreads();
    }

    if (lane == 0) {
        const int gi = warp_row0;
        const int gj = warp_col0;
        const int tile_id = (gi / HT) * tiles_w + (gj / HT);
        uint32_t* tb = &transcript_buffer[(size_t)tile_id * TRANSCRIPT_LEN];

        #pragma unroll
        for (int i = 0; i < TRANSCRIPT_LEN; i += 4) {
            *((int4*)&tb[i]) = *((int4*)&sT[warp][i]);
        }
    }
#else
    // On pre-sm_80, this kernel should never be launched — return.
    (void)A; (void)Bt; (void)n; (void)k; (void)R;
    (void)transcript_buffer;
#endif
}

// ==================================================================
// Host dispatcher
// ==================================================================
cudaError_t launch_pearl_ampere(
    const int8_t* A, const int8_t* Bt,
    int m, int n, int k, int R,
    uint32_t* transcript_buffer,
    cudaStream_t stream)
{
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) return err;

    if (prop.major < 8) {
        return cudaErrorNotSupported;
    }

    const int block_m = 64, block_n = 64, block_k = 32;
    const int warps_m = 4, warps_n = 4, stages = 2, minb = 3;

    if (m % block_m != 0 || n % block_n != 0 || k % block_k != 0) {
        return cudaErrorInvalidValue;
    }

    dim3 block(warps_m * warps_n * 32);
    int grids_m = m / block_m;
    int grids_n = n / block_n;
    dim3 grid(grids_m * grids_n);

    #define LAUNCH(BM,BN,BK,WM,WN,STG,MNB) \
        if (block_m==BM && block_n==BN && block_k==BK && \
            warps_m==WM && warps_n==WN && stages==STG && minb==MNB) { \
            pearl_ampere_fused_kernel<BM,BN,BK,WM,WN,STG,MNB> \
                <<<grid, block, 0, stream>>>(A,Bt,n,k,R,transcript_buffer); \
            return cudaGetLastError(); \
        }

    LAUNCH(64,  64, 32, 4,4,2,3)

    #undef LAUNCH

    return cudaErrorUnknown;
}

#endif // !defined(PEARL_UNIT_TEST)
