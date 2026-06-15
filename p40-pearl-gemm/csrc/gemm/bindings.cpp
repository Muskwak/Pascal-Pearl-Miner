// Python bindings for the Pascal (sm_61) and Ampere+ (sm_80+) Pearl GEMM kernels.
//
// Auto-detects GPU architecture and dispatches to DP4A (Pascal) or tensor-core
// (Ampere/Ada) kernel. Multi-GPU: each device uses its optimal kernel path.
//
// Registers the C++/CUDA launchers as functions on the `p40_pearl_gemm_cuda`
// extension module so they are callable from `p40_gemm_bindings.py`.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cstdint>

// --- Launcher declarations (definitions live in the *_sm61.cu / *.cu files) --
// Pascal-native launchers have plain C++ linkage; the denoise converter is
// exported with C linkage from api_sm61.cu.
void launch_dp4a_gemm(
    const int8_t* A, const int8_t* B,
    const float* A_scales, const float* B_scales,
    half* C, int M, int N, int K, cudaStream_t stream);

void launch_noise_A(
    const int8_t* A, const int8_t* EAL, const int8_t* EAR, const int8_t* EBL,
    int8_t* ApEA, int32_t* AxEBL, int M, int K, int R, cudaStream_t stream);

// GPU noise generator (definition in noise_generation.cu)
void launch_noise_gen(
    int8_t* EAL, int8_t* EAR_R_major, int8_t* EBL_K_major, int8_t* EBR,
    const uint8_t* key_A, const uint8_t* key_B,
    int m, int n, int k, int R, cudaStream_t stream);

void launch_noise_B(
    const int8_t* B, const int8_t* EBR, const int8_t* EAR, const int8_t* EBL,
    int8_t* BpEB, int32_t* EARxBpEB, int N, int K, int R, cudaStream_t stream);

extern "C" void launch_denoise_converter(
    const int32_t* EARxBpEB_in, const int32_t* AxEBL_in,
    half* EARxBpEB_out, half* AxEBL_out,
    int M, int N, int R, cudaStream_t stream);

void launch_inner_hash_kernel(
    uint32_t* input_buffer, int input_size,
    uint32_t* output_hash, int64_t iterations, cudaStream_t stream);

// Pascal Pearl PoW kernel (definition in pearl_pow_sm61.cu)
void launch_pearl_pow(
    const int8_t* A, const int8_t* Bt, int m, int n, int k, int R,
    const uint32_t* pow_key, const uint32_t* pow_target,
    uint8_t* out_digests, int* found_flag, int* found_coord, cudaStream_t stream);

// Fused high-throughput variant (definition in pearl_pow_fused_sm61.cu)
void launch_pearl_pow_fused(
    const int8_t* A, const int8_t* Bt, int m, int n, int k, int R,
    const uint32_t* pow_key, const uint32_t* pow_target,
    uint8_t* out_digests, int* found_flag, int* found_coord, cudaStream_t stream);

void launch_pearl_pow_fused_v(
    const int8_t* A, const int8_t* Bt, int m, int n, int k, int R,
    const uint32_t* pow_key, const uint32_t* pow_target,
    uint8_t* out_digests, int* found_flag, int* found_coord,
    int variant, cudaStream_t stream);

// GEMM-only variant (definition in pearl_gemm_only_sm61.cu) — writes transcript
// to global buffer, no BLAKE3.
void launch_pearl_gemm_only(
    const int8_t* A, const int8_t* Bt, int m, int n, int k, int R,
    uint32_t* transcript_buffer, int variant, cudaStream_t stream);

// BLAKE3-only kernel (definition in pearl_blake3_sm61.cu) — consumes transcript
// buffer, produces digests + found-flag.
void launch_pearl_blake3(
    const uint32_t* transcript_buffer, int num_tiles, int n,
    const uint32_t* pow_key, const uint32_t* pow_target,
    uint8_t* out_digests, int* found_flag, int* found_coord,
    cudaStream_t stream);

// Ampere+ tensor-core GEMM+transcript kernel (definition in pearl_ampere_tc.cu).
// Produces transcript buffer identical format to launch_pearl_gemm_only.
// Returns cudaError_t (cudaSuccess on success, cudaErrorNotSupported on pre-sm_80).
cudaError_t launch_pearl_ampere(
    const int8_t* A, const int8_t* Bt, int m, int n, int k, int R,
    uint32_t* transcript_buffer, cudaStream_t stream);

// tensor_hash host entry (definition in tensor_hash.cu -> tensor_hash_host_sm61.hpp)
void tensor_hash(
    const uint8_t* data, uint32_t data_size, uint8_t* out,
    const uint8_t key[32], uint32_t num_blocks, uint32_t threads_per_block,
    uint32_t num_stages, uint32_t leaves_per_mt_block, uint8_t* roots,
    cudaDeviceProp& deviceProp, cudaStream_t stream);

// Commitment hash from merkle roots (definition in tensor_hash_host_sm61.hpp)
void commitment_hash_from_merkle_roots(
    const uint8_t* A_merkle_root, const uint8_t* B_merkle_root,
    const uint8_t* key, uint8_t* A_commitment_hash, uint8_t* B_commitment_hash,
    cudaDeviceProp& deviceProp, cudaStream_t stream);

// RNG fill + transpose (definition in rng_fill_sm61.cu, extern "C")
extern "C" void launch_fill_rand_i8(int8_t* out, int64_t numel, uint64_t seed, cudaStream_t stream);
extern "C" void launch_transpose_i8(const int8_t* src, int8_t* dst, int rows, int cols, cudaStream_t stream);

namespace {

inline cudaStream_t cur_stream() {
  return at::cuda::getCurrentCUDAStream().stream();
}

inline half* half_ptr(at::Tensor& t) {
  return reinterpret_cast<half*>(t.data_ptr<at::Half>());
}

}  // namespace

void dp4a_gemm(at::Tensor A, at::Tensor B, at::Tensor A_scales,
               at::Tensor B_scales, at::Tensor C,
               int64_t M, int64_t N, int64_t K) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda(),
              "dp4a_gemm: all tensors must be CUDA");
  TORCH_CHECK(A.scalar_type() == at::kChar && B.scalar_type() == at::kChar,
              "dp4a_gemm: A and B must be int8");
  TORCH_CHECK(C.scalar_type() == at::kHalf, "dp4a_gemm: C must be float16");
  TORCH_CHECK(A_scales.scalar_type() == at::kFloat &&
                  B_scales.scalar_type() == at::kFloat,
              "dp4a_gemm: scales must be float32");
  // Run on the tensors' device, and use that device's current stream, so the
  // kernel does not dereference pointers from another device's address space.
  const c10::cuda::CUDAGuard device_guard(A.device());
  launch_dp4a_gemm(
      A.data_ptr<int8_t>(), B.data_ptr<int8_t>(),
      A_scales.data_ptr<float>(), B_scales.data_ptr<float>(),
      half_ptr(C), (int)M, (int)N, (int)K, cur_stream());
}

void noise_A(at::Tensor A, at::Tensor EAL, at::Tensor EAR, at::Tensor EBL,
             at::Tensor ApEA, at::Tensor AxEBL,
             int64_t M, int64_t K, int64_t R) {
  TORCH_CHECK(A.is_cuda(), "noise_A: tensors must be CUDA");
  const c10::cuda::CUDAGuard device_guard(A.device());
  launch_noise_A(
      A.data_ptr<int8_t>(), EAL.data_ptr<int8_t>(), EAR.data_ptr<int8_t>(),
      EBL.data_ptr<int8_t>(), ApEA.data_ptr<int8_t>(),
      AxEBL.data_ptr<int32_t>(), (int)M, (int)K, (int)R, cur_stream());
}

void noise_B(at::Tensor B, at::Tensor EBR, at::Tensor EAR, at::Tensor EBL,
             at::Tensor BpEB, at::Tensor EARxBpEB,
             int64_t N, int64_t K, int64_t R) {
  TORCH_CHECK(B.is_cuda(), "noise_B: tensors must be CUDA");
  const c10::cuda::CUDAGuard device_guard(B.device());
  launch_noise_B(
      B.data_ptr<int8_t>(), EBR.data_ptr<int8_t>(), EAR.data_ptr<int8_t>(),
      EBL.data_ptr<int8_t>(), BpEB.data_ptr<int8_t>(),
      EARxBpEB.data_ptr<int32_t>(), (int)N, (int)K, (int)R, cur_stream());
}

void denoise_converter(c10::optional<at::Tensor> EARxBpEB_in,
                       c10::optional<at::Tensor> AxEBL_in,
                       c10::optional<at::Tensor> EARxBpEB_out,
                       c10::optional<at::Tensor> AxEBL_out,
                       int64_t M, int64_t N, int64_t R) {
  c10::optional<c10::cuda::CUDAGuard> device_guard;
  if (AxEBL_in) device_guard.emplace(AxEBL_in->device());
  else if (EARxBpEB_in) device_guard.emplace(EARxBpEB_in->device());
  const int32_t* ear_in =
      EARxBpEB_in ? EARxBpEB_in->data_ptr<int32_t>() : nullptr;
  const int32_t* axebl_in =
      AxEBL_in ? AxEBL_in->data_ptr<int32_t>() : nullptr;
  half* ear_out = EARxBpEB_out ? half_ptr(*EARxBpEB_out) : nullptr;
  half* axebl_out = AxEBL_out ? half_ptr(*AxEBL_out) : nullptr;
  launch_denoise_converter(
      ear_in, axebl_in, ear_out, axebl_out,
      (int)M, (int)N, (int)R, cur_stream());
}

at::Tensor inner_hash(at::Tensor input_buffer, int64_t iterations) {
  TORCH_CHECK(input_buffer.is_cuda(), "inner_hash: input must be CUDA");
  const c10::cuda::CUDAGuard device_guard(input_buffer.device());
  auto out = at::empty({1}, input_buffer.options());
  launch_inner_hash_kernel(
      reinterpret_cast<uint32_t*>(input_buffer.data_ptr()),
      (int)input_buffer.numel(),
      reinterpret_cast<uint32_t*>(out.data_ptr()),
      iterations, cur_stream());
  return out;
}

void tensor_hash_py(at::Tensor data, at::Tensor key, at::Tensor out,
                    at::Tensor roots, int64_t threads_per_block,
                    int64_t num_stages, int64_t leaves_per_mt_block) {
  TORCH_CHECK(data.is_cuda() && key.is_cuda() && out.is_cuda() && roots.is_cuda(),
              "tensor_hash: all tensors must be CUDA");
  TORCH_CHECK(data.is_contiguous(), "tensor_hash: data must be contiguous");
  TORCH_CHECK(key.scalar_type() == at::kByte && out.scalar_type() == at::kByte &&
                  roots.scalar_type() == at::kByte,
              "tensor_hash: key/out/roots must be uint8");
  TORCH_CHECK(key.numel() == 32, "tensor_hash: key must be 32 bytes");
  TORCH_CHECK(out.numel() == 32, "tensor_hash: out must be 32 bytes");

  const c10::cuda::CUDAGuard device_guard(data.device());

  const uint32_t data_size = static_cast<uint32_t>(data.nbytes());
  const uint32_t chunk_size = 1024u;
  const uint32_t num_chunks = (data_size + chunk_size - 1) / chunk_size;
  const uint32_t num_blocks =
      (num_chunks + (uint32_t)threads_per_block - 1) / (uint32_t)threads_per_block;
  TORCH_CHECK((uint32_t)(roots.nbytes()) >= num_blocks * 32u,
              "tensor_hash: roots scratch too small; need ", num_blocks * 32u,
              " bytes");

  cudaDeviceProp* dprops = at::cuda::getCurrentDeviceProperties();
  tensor_hash(reinterpret_cast<const uint8_t*>(data.data_ptr()), data_size,
              out.data_ptr<uint8_t>(), key.data_ptr<uint8_t>(), num_blocks,
              (uint32_t)threads_per_block, (uint32_t)num_stages,
              (uint32_t)leaves_per_mt_block, roots.data_ptr<uint8_t>(), *dprops,
              cur_stream());
}

// Returns {digests[num_tiles,32] uint8, found[1] int32, coord[2] int32}.
// A: [m,k] int8 noised; Bt: [n,k] int8 noised (B transposed). pow_key/pow_target:
// 32-byte uint8 tensors (pow_target little-endian uint256).
//
// Auto-detects GPU architecture:
//   sm_80+ (Ampere/Ada): tensor-core kernel (launch_pearl_ampere) + BLAKE3
//   pre-sm_80 (Pascal/Volta): DP4A kernel (launch_pearl_pow)
std::vector<at::Tensor> pearl_pow(at::Tensor A, at::Tensor Bt,
                                  at::Tensor pow_key, at::Tensor pow_target,
                                  int64_t R) {
  TORCH_CHECK(A.is_cuda() && Bt.is_cuda(), "pearl_pow: A/Bt must be CUDA");
  TORCH_CHECK(A.scalar_type() == at::kChar && Bt.scalar_type() == at::kChar,
              "pearl_pow: A/Bt must be int8");
  TORCH_CHECK(A.is_contiguous() && Bt.is_contiguous(), "pearl_pow: A/Bt must be contiguous");
  TORCH_CHECK(pow_key.numel() == 32 && pow_target.numel() == 32,
              "pearl_pow: pow_key/pow_target must be 32 bytes");
  const int m = (int)A.size(0), k = (int)A.size(1), n = (int)Bt.size(0);
  TORCH_CHECK(Bt.size(1) == k, "pearl_pow: A[k] must match Bt[k]");
  TORCH_CHECK(R == 256 || R == 128 || R == 64, "pearl_pow: R must be 64, 128, or 256");

  const c10::cuda::CUDAGuard device_guard(A.device());
  auto dprops = at::cuda::getCurrentDeviceProperties();

  const int num_tiles = (m / 16) * (n / 16);
  auto u8 = at::TensorOptions().dtype(at::kByte).device(A.device());
  auto i32 = at::TensorOptions().dtype(at::kInt).device(A.device());
  auto digests = at::empty({num_tiles, 32}, u8);
  auto found = at::zeros({1}, i32);
  auto coord = at::full({2}, -1, i32);

  if (dprops->major >= 8) {
    // Ampere+ tensor-core path
    TORCH_CHECK(m % 64 == 0 && n % 64 == 0 && k % R == 0,
                "pearl_pow (Ampere): require m%64==0, n%64==0, k%R==0");
    auto transcript_buffer = at::zeros({num_tiles, 16}, i32);
    cudaError_t err = launch_pearl_ampere(
        A.data_ptr<int8_t>(), Bt.data_ptr<int8_t>(), m, n, k, (int)R,
        reinterpret_cast<uint32_t*>(transcript_buffer.data_ptr()),
        cur_stream());
    TORCH_CHECK(err == cudaSuccess, "pearl_pow (Ampere): kernel launch failed: ",
                cudaGetErrorString(err));
    launch_pearl_blake3(
        reinterpret_cast<const uint32_t*>(transcript_buffer.data_ptr()),
        num_tiles, n,
        reinterpret_cast<const uint32_t*>(pow_key.data_ptr()),
        reinterpret_cast<const uint32_t*>(pow_target.data_ptr()),
        digests.data_ptr<uint8_t>(), found.data_ptr<int>(),
        coord.data_ptr<int>(), cur_stream());
  } else {
    // Pascal DP4A path
    TORCH_CHECK(m % 16 == 0 && n % 16 == 0 && k % R == 0,
                "pearl_pow (Pascal): require m%16==0, n%16==0, k%R==0");
    launch_pearl_pow(
        A.data_ptr<int8_t>(), Bt.data_ptr<int8_t>(), m, n, k, (int)R,
        reinterpret_cast<const uint32_t*>(pow_key.data_ptr()),
        reinterpret_cast<const uint32_t*>(pow_target.data_ptr()),
        digests.data_ptr<uint8_t>(), found.data_ptr<int>(), coord.data_ptr<int>(),
        cur_stream());
  }
  return {digests, found, coord};
}

// Fused variant: same outputs/semantics as pearl_pow, but requires the block
// region to divide m/n (WM=WN=4 -> m%64==0, n%64==0).
std::vector<at::Tensor> pearl_pow_fused(at::Tensor A, at::Tensor Bt,
                                        at::Tensor pow_key, at::Tensor pow_target,
                                        int64_t R, int64_t variant) {
  TORCH_CHECK(A.is_cuda() && Bt.is_cuda(), "pearl_pow_fused: A/Bt must be CUDA");
  TORCH_CHECK(A.scalar_type() == at::kChar && Bt.scalar_type() == at::kChar,
              "pearl_pow_fused: A/Bt must be int8");
  TORCH_CHECK(A.is_contiguous() && Bt.is_contiguous(),
              "pearl_pow_fused: A/Bt must be contiguous");
  TORCH_CHECK(pow_key.numel() == 32 && pow_target.numel() == 32,
              "pearl_pow_fused: pow_key/pow_target must be 32 bytes");
  const int m = (int)A.size(0), k = (int)A.size(1), n = (int)Bt.size(0);
  TORCH_CHECK(Bt.size(1) == k, "pearl_pow_fused: A[k] must match Bt[k]");
  TORCH_CHECK(m % 64 == 0 && n % 64 == 0 && k % R == 0,
              "pearl_pow_fused: require m%64==0, n%64==0, k%R==0");
  TORCH_CHECK(R == 256 || R == 128 || R == 64,
              "pearl_pow_fused: R must be 64, 128, or 256");

  const c10::cuda::CUDAGuard device_guard(A.device());
  const int num_tiles = (m / 16) * (n / 16);
  auto u8 = at::TensorOptions().dtype(at::kByte).device(A.device());
  auto i32 = at::TensorOptions().dtype(at::kInt).device(A.device());
  auto digests = at::empty({num_tiles, 32}, u8);
  auto found = at::zeros({1}, i32);
  auto coord = at::full({2}, -1, i32);

  launch_pearl_pow_fused_v(
      A.data_ptr<int8_t>(), Bt.data_ptr<int8_t>(), m, n, k, (int)R,
      reinterpret_cast<const uint32_t*>(pow_key.data_ptr()),
      reinterpret_cast<const uint32_t*>(pow_target.data_ptr()),
      digests.data_ptr<uint8_t>(), found.data_ptr<int>(), coord.data_ptr<int>(),
      (int)variant, cur_stream());
  return {digests, found, coord};
}

// GEMM-only kernel: writes 16-word transcript per tile to the provided buffer.
// transcript_buffer must be [num_tiles, 16] int32, caller-allocated on the device.
void pearl_gemm_only(at::Tensor A, at::Tensor Bt,
                     at::Tensor transcript_buffer,
                     int64_t R, int64_t variant) {
  TORCH_CHECK(A.is_cuda() && Bt.is_cuda(), "pearl_gemm_only: A/Bt must be CUDA");
  TORCH_CHECK(A.scalar_type() == at::kChar && Bt.scalar_type() == at::kChar,
              "pearl_gemm_only: A/Bt must be int8");
  TORCH_CHECK(A.is_contiguous() && Bt.is_contiguous(),
              "pearl_gemm_only: A/Bt must be contiguous");
  TORCH_CHECK(transcript_buffer.is_cuda() &&
              transcript_buffer.scalar_type() == at::kInt,
              "pearl_gemm_only: transcript_buffer must be CUDA int32");
  const int m = (int)A.size(0), k = (int)A.size(1), n = (int)Bt.size(0);
  TORCH_CHECK(Bt.size(1) == k, "pearl_gemm_only: A[k] must match Bt[k]");
  TORCH_CHECK(m % 64 == 0 && n % 64 == 0 && k % R == 0,
              "pearl_gemm_only: require m%64==0, n%64==0, k%R==0");
  TORCH_CHECK(R == 256 || R == 128 || R == 64,
              "pearl_gemm_only: R must be 64, 128, or 256");

  const c10::cuda::CUDAGuard device_guard(A.device());
  launch_pearl_gemm_only(
      A.data_ptr<int8_t>(), Bt.data_ptr<int8_t>(), m, n, k, (int)R,
      reinterpret_cast<uint32_t*>(transcript_buffer.data_ptr()),
      (int)variant, cur_stream());
}

// Two-step kernel: GEMM-only → transcript buffer → BLAKE3-only.
// Returns {digests[num_tiles,32] uint8, found[1] int32, coord[2] int32}.
// Same semantics as pearl_pow_fused but with higher occupancy in the GEMM step.
std::vector<at::Tensor> pearl_pow_split(at::Tensor A, at::Tensor Bt,
                                        at::Tensor pow_key,
                                        at::Tensor pow_target,
                                        int64_t R, int64_t variant) {
  TORCH_CHECK(A.is_cuda() && Bt.is_cuda(), "pearl_pow_split: A/Bt must be CUDA");
  TORCH_CHECK(A.scalar_type() == at::kChar && Bt.scalar_type() == at::kChar,
              "pearl_pow_split: A/Bt must be int8");
  TORCH_CHECK(A.is_contiguous() && Bt.is_contiguous(),
              "pearl_pow_split: A/Bt must be contiguous");
  TORCH_CHECK(pow_key.numel() == 32 && pow_target.numel() == 32,
              "pearl_pow_split: pow_key/pow_target must be 32 bytes");
  const int m = (int)A.size(0), k = (int)A.size(1), n = (int)Bt.size(0);
  TORCH_CHECK(Bt.size(1) == k, "pearl_pow_split: A[k] must match Bt[k]");
  TORCH_CHECK(m % 64 == 0 && n % 64 == 0 && k % R == 0,
              "pearl_pow_split: require m%64==0, n%64==0, k%R==0");
  TORCH_CHECK(R == 256 || R == 128 || R == 64,
              "pearl_pow_split: R must be 64, 128, or 256");

  const c10::cuda::CUDAGuard device_guard(A.device());
  const int num_tiles = (m / 16) * (n / 16);
  auto u8 = at::TensorOptions().dtype(at::kByte).device(A.device());
  auto i32 = at::TensorOptions().dtype(at::kInt).device(A.device());
  auto digests = at::empty({num_tiles, 32}, u8);
  auto found = at::zeros({1}, i32);
  auto coord = at::full({2}, -1, i32);
  auto transcript_buffer = at::zeros({num_tiles, 16}, i32);

  launch_pearl_gemm_only(
      A.data_ptr<int8_t>(), Bt.data_ptr<int8_t>(), m, n, k, (int)R,
      reinterpret_cast<uint32_t*>(transcript_buffer.data_ptr()),
      (int)variant, cur_stream());

  launch_pearl_blake3(
      reinterpret_cast<const uint32_t*>(transcript_buffer.data_ptr()),
      num_tiles, n,
      reinterpret_cast<const uint32_t*>(pow_key.data_ptr()),
      reinterpret_cast<const uint32_t*>(pow_target.data_ptr()),
      digests.data_ptr<uint8_t>(), found.data_ptr<int>(),
      coord.data_ptr<int>(), cur_stream());

  return {digests, found, coord};
}

// Generate the Pearl noise tensors on the GPU from the commitment keys.
// Returns {EAL[m,R], EAR[R,k], EBL[k,R], EBR[n,R]} int8 (matches Python
// generate_noise: EAL/EBR uniform, EAR/EBL sparse-perm; seeds "A_tensor"/"B_tensor").
std::vector<at::Tensor> noise_gen(at::Tensor key_A, at::Tensor key_B,
                                  int64_t m, int64_t n, int64_t k, int64_t R) {
  TORCH_CHECK(key_A.is_cuda() && key_B.is_cuda(), "noise_gen: keys must be CUDA");
  TORCH_CHECK(key_A.scalar_type() == at::kByte && key_B.scalar_type() == at::kByte,
              "noise_gen: keys must be uint8");
  TORCH_CHECK(key_A.numel() == 32 && key_B.numel() == 32,
              "noise_gen: keys must be 32 bytes");
  const c10::cuda::CUDAGuard device_guard(key_A.device());
  auto i8 = at::TensorOptions().dtype(at::kChar).device(key_A.device());
  auto EAL = at::empty({m, R}, i8);
  auto EAR = at::empty({R, k}, i8);
  auto EBL = at::empty({k, R}, i8);
  auto EBR = at::empty({n, R}, i8);
  launch_noise_gen(
      EAL.data_ptr<int8_t>(), EAR.data_ptr<int8_t>(), EBL.data_ptr<int8_t>(),
      EBR.data_ptr<int8_t>(),
      reinterpret_cast<const uint8_t*>(key_A.data_ptr()),
      reinterpret_cast<const uint8_t*>(key_B.data_ptr()),
      (int)m, (int)n, (int)k, (int)R, cur_stream());
  return {EAL, EAR, EBL, EBR};
}

// Fill int8 tensor(s) on GPU with random values in [-64, 63] using Philox RNG.
// out: int8 tensor on CUDA — filled in-place.
// seed: optional 64-bit seed (default 0 uses a hash of the current ns timestamp).
void fill_rand_i8_py(at::Tensor out, c10::optional<int64_t> seed_opt) {
  TORCH_CHECK(out.is_cuda(), "fill_rand_i8: tensor must be CUDA");
  TORCH_CHECK(out.scalar_type() == at::kChar, "fill_rand_i8: tensor must be int8");
  TORCH_CHECK(out.is_contiguous(), "fill_rand_i8: tensor must be contiguous");
  const c10::cuda::CUDAGuard device_guard(out.device());
  uint64_t seed = seed_opt.value_or(0xDEADBEEFCAFEBABEull);
  launch_fill_rand_i8(out.data_ptr<int8_t>(), out.numel(), seed, cur_stream());
}

// Full GPU setup: generate A,B on device, compute commitment hashes, return
// {A, B, noise_seed_A, noise_seed_B} as tensors.
// key: 32-byte uint8 CUDA tensor (job key = BLAKE3(header||config))
// M,N,K,R: matrix dims; seed: Philox seed.
std::vector<at::Tensor> setup_job(at::Tensor key,
                                  int64_t M, int64_t N, int64_t K, int64_t R,
                                  int64_t seed) {
  TORCH_CHECK(key.is_cuda() && key.scalar_type() == at::kByte && key.numel() == 32,
              "setup_job: key must be 32-byte uint8 on CUDA");
  const c10::cuda::CUDAGuard device_guard(key.device());
  auto stream = cur_stream();
  auto dprops = *at::cuda::getCurrentDeviceProperties();
  auto dev = key.device();

  auto i8 = at::TensorOptions().dtype(at::kChar).device(dev);
  auto u8 = at::TensorOptions().dtype(at::kByte).device(dev);

  // 1. Allocate and fill A [M,K], B [K,N]
  auto A = at::empty({M, K}, i8);
  auto B = at::empty({K, N}, i8);
  launch_fill_rand_i8(A.data_ptr<int8_t>(), M * K, seed, stream);
  launch_fill_rand_i8(B.data_ptr<int8_t>(), K * N, seed + 1, stream);

  // 2. Transpose B → Bt [N,K] for B^T Merkle root
  auto Bt = at::empty({N, K}, i8);
  launch_transpose_i8(B.data_ptr<int8_t>(), Bt.data_ptr<int8_t>(), (int)K, (int)N, stream);

  // 3. Tensor hash scratch sizing
  constexpr uint32_t chunk_size = 1024u;
  const uint32_t tpb = 128;
  uint32_t num_chunks_A = (uint32_t)(M * K + chunk_size - 1) / chunk_size;
  uint32_t num_blocks_A = (num_chunks_A + tpb - 1) / tpb;
  uint32_t num_chunks_B = (uint32_t)(N * K + chunk_size - 1) / chunk_size;
  uint32_t num_blocks_B = (num_chunks_B + tpb - 1) / tpb;
  uint32_t scratch_bytes = std::max(num_blocks_A, num_blocks_B) * 32u;

  // 4. Allocate Merkle root outputs + scratch
  auto A_root = at::empty(32, u8);
  auto B_root = at::empty(32, u8);
  auto scratch = at::empty({(int64_t)scratch_bytes}, u8);

  // 5. tensor_hash A → A_root
  tensor_hash(
      reinterpret_cast<const uint8_t*>(A.data_ptr<int8_t>()), (uint32_t)(M * K),
      A_root.data_ptr<uint8_t>(), key.data_ptr<uint8_t>(),
      num_blocks_A, tpb, 2, 512,
      scratch.data_ptr<uint8_t>(), dprops, stream);

  // 6. tensor_hash Bt → B_root
  tensor_hash(
      reinterpret_cast<const uint8_t*>(Bt.data_ptr<int8_t>()), (uint32_t)(N * K),
      B_root.data_ptr<uint8_t>(), key.data_ptr<uint8_t>(),
      num_blocks_B, tpb, 2, 512,
      scratch.data_ptr<uint8_t>(), dprops, stream);

  // 7. Commitment hash from merkle roots → noise seeds
  auto noise_seed_A = at::empty(32, u8);
  auto noise_seed_B = at::empty(32, u8);
  commitment_hash_from_merkle_roots(
      A_root.data_ptr<uint8_t>(), B_root.data_ptr<uint8_t>(),
      key.data_ptr<uint8_t>(),
      noise_seed_A.data_ptr<uint8_t>(), noise_seed_B.data_ptr<uint8_t>(),
      dprops, stream);

  return {A, B, noise_seed_A, noise_seed_B};
}

// Free Bt scratch allocated in setup_job (frees GPU memory eagerly).
void free_tensor(at::Tensor t) {
  // Force release by assigning empty. The tensor goes out of scope immediately.
  at::Tensor empty;
  t = empty;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Pascal (sm_61) + Ampere (sm_80+) Pearl GEMM CUDA kernels; auto-selects DP4A or tensor-core";
  m.def("dp4a_gemm", &dp4a_gemm, "INT8 DP4A GEMM (C = A @ B^T, dequantized)");
  m.def("noise_A", &noise_A, "Pearl noise A kernel");
  m.def("noise_gen", &noise_gen,
        "Generate Pearl noise EAL/EAR/EBL/EBR on GPU from commitment keys",
        pybind11::arg("key_A"), pybind11::arg("key_B"),
        pybind11::arg("m"), pybind11::arg("n"), pybind11::arg("k"),
        pybind11::arg("R"));
  m.def("noise_B", &noise_B, "Pearl noise B kernel");
  m.def("denoise_converter", &denoise_converter,
        "int32 -> fp16 denoise conversion");
  m.def("inner_hash", &inner_hash, "PoW inner hash (XOR reduction)");
  m.def("pearl_pow", &pearl_pow,
        "Pascal Pearl PoW: per-16x16-tile transcript + keyed BLAKE3 vs target",
        pybind11::arg("A"), pybind11::arg("Bt"), pybind11::arg("pow_key"),
        pybind11::arg("pow_target"), pybind11::arg("R") = 128);
  m.def("pearl_pow_fused", &pearl_pow_fused,
        "Fused high-throughput Pearl PoW (warp-per-tile, shared-mem reuse)",
        pybind11::arg("A"), pybind11::arg("Bt"), pybind11::arg("pow_key"),
        pybind11::arg("pow_target"), pybind11::arg("R") = 256,
        pybind11::arg("variant") = 0);
  m.def("pearl_gemm_only", &pearl_gemm_only,
        "GEMM-only: compute transcripts into global buffer (no BLAKE3). "
        "Higher occupancy than pearl_pow_fused.",
        pybind11::arg("A"), pybind11::arg("Bt"),
        pybind11::arg("transcript_buffer"),
        pybind11::arg("R") = 256, pybind11::arg("variant") = 0);
  m.def("pearl_pow_split", &pearl_pow_split,
        "Two-step PoW: GEMM-only → transcript buffer → BLAKE3-only. "
        "Same semantics as pearl_pow_fused but higher occupancy.",
        pybind11::arg("A"), pybind11::arg("Bt"), pybind11::arg("pow_key"),
        pybind11::arg("pow_target"), pybind11::arg("R") = 256,
        pybind11::arg("variant") = 0);
  m.def("tensor_hash", &tensor_hash_py,
        "BLAKE3 keyed Merkle hash of a tensor (Pascal)",
        pybind11::arg("data"), pybind11::arg("key"), pybind11::arg("out"),
        pybind11::arg("roots"), pybind11::arg("threads_per_block") = 128,
        pybind11::arg("num_stages") = 2,
        pybind11::arg("leaves_per_mt_block") = 512);
  m.def("fill_rand_i8", &fill_rand_i8_py,
        "Fill int8 CUDA tensor with random values [-64,63] using Philox RNG",
        pybind11::arg("out"), pybind11::arg("seed") = pybind11::none());
  m.def("setup_job", &setup_job,
        "Full GPU setup: RNG A,B → commit → noise seeds. Returns {A,B,noise_A,noise_B}",
        pybind11::arg("key"), pybind11::arg("M"), pybind11::arg("N"),
        pybind11::arg("K"), pybind11::arg("R"), pybind11::arg("seed") = 0);
}
