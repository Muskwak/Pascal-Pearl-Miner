#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "pearl_gemm_constants.hpp"

// Pascal (sm_61) denoise converter: int32 -> fp16 with a fixed power-of-two
// rescale.  Replaces the upstream CuTe/CUTLASS DenoiseConverterKernel for the
// Pascal path.  The two operands (EARxBpEB of size N*R and AxEBL of size M*R)
// are converted in a single fused launch; either may be disabled by passing a
// null pointer pair, in which case its element count is zero.

namespace {

__global__ void denoise_convert_kernel(
    const int32_t* __restrict__ EARxBpEB_in,
    const int32_t* __restrict__ AxEBL_in,
    half* __restrict__ EARxBpEB_out,
    half* __restrict__ AxEBL_out,
    int n_earxbpeb,
    int n_axebl) {

  // The two operands use different fixed-point scales (see DenoiseConverterKernel
  // / pearl_gemm_constants.hpp): AxEBL is divided by 1<<14, EARxBpEB by 1<<12.
  constexpr float kInvScaleEARxBpEB =
      1.0f / float(pearl::kEARxBpEBScaleFactor);  // 1 / (1<<12)
  constexpr float kInvScaleAxEBL =
      1.0f / float(pearl::kAxEBLScaleFactor);      // 1 / (1<<14)

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const int total = n_earxbpeb + n_axebl;

  for (int i = idx; i < total; i += stride) {
    if (i < n_earxbpeb) {
      float fval = float(EARxBpEB_in[i]) * kInvScaleEARxBpEB;
      fval = fmaxf(-65504.0f, fminf(65504.0f, fval));
      EARxBpEB_out[i] = __float2half(fval);
    } else {
      int j = i - n_earxbpeb;
      float fval = float(AxEBL_in[j]) * kInvScaleAxEBL;
      fval = fmaxf(-65504.0f, fminf(65504.0f, fval));
      AxEBL_out[j] = __float2half(fval);
    }
  }
}

}  // namespace

extern "C" void launch_denoise_converter(
    const int32_t* EARxBpEB_in,
    const int32_t* AxEBL_in,
    half* EARxBpEB_out,
    half* AxEBL_out,
    int M, int N, int R,
    cudaStream_t stream) {

  int n_earxbpeb = (EARxBpEB_in && EARxBpEB_out) ? N * R : 0;
  int n_axebl = (AxEBL_in && AxEBL_out) ? M * R : 0;
  int total = n_earxbpeb + n_axebl;
  if (total == 0) return;

  int threads = 256;
  int blocks = (total + threads - 1) / threads;

  denoise_convert_kernel<<<blocks, threads, 0, stream>>>(
      EARxBpEB_in, AxEBL_in, EARxBpEB_out, AxEBL_out, n_earxbpeb, n_axebl);
}
