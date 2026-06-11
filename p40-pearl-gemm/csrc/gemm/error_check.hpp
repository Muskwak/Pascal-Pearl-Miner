#pragma once
#include <cuda_runtime.h>
#include <cstdio>
// P40_NO_TORCH: build the kernels into the standalone (torch-free) shared library
// without the c10/TORCH_CHECK dependency. The normal torch extension build leaves
// the macro undefined and keeps the original behavior.
#ifdef P40_NO_TORCH
#include <stdexcept>
#else
#include <c10/util/Exception.h>
#endif

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool do_abort = true) {
  if (code != cudaSuccess) {
    const char* error_str = cudaGetErrorString(code);
    // CUDA has a sticky global error state. We must clear it with cudaGetLastError()
    // before throwing, otherwise subsequent CUDA calls may fail unexpectedly.
    cudaGetLastError();
#ifdef P40_NO_TORCH
    fprintf(stderr, "CUDA error: %s at %s:%d\n", error_str, file, line);
    throw std::runtime_error(error_str);
#else
    TORCH_CHECK(false, "CUDA error: ", error_str, " at ", file, ":", line);
#endif
  }
}

#define CHECK_CUDA_KERNEL_LAUNCH() gpuErrchk(cudaGetLastError())
