#!/usr/bin/env bash
# Build the torch-free CUDA shared library (p40cuda.dll / libp40cuda.so).
# Links only the CUDA runtime — no torch, no pybind. Driven from Python via ctypes.
#
# Run from p40-pearl-gemm/ :  bash packaging/build_capi.sh
set -euo pipefail
cd "$(dirname "$0")/.."

: "${CUTLASS_DIR:=C:/Users/ADMIN/audits/aphrodite-engine/.deps/cutlass-src/include}"
NVCC="${NVCC:-nvcc}"

# Default arch: native Pascal SASS + a compute_61 PTX fallback so the library
# JIT-loads on ANY newer NVIDIA card too (mixed rigs). DP4A lives in this PTX,
# so every sm_61+ card runs the same kernel. Override GENCODE to add native
# SASS for specific newer arches (avoids first-run JIT), e.g.:
#   GENCODE="-gencode arch=compute_61,code=sm_61 \
#            -gencode arch=compute_75,code=sm_75 \
#            -gencode arch=compute_86,code=sm_86 \
#            -gencode arch=compute_89,code=sm_89 \
#            -gencode arch=compute_61,code=compute_61"
GENCODE="${GENCODE:-"-gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=compute_61"}"

case "$(uname -s)" in
  *NT*|*MINGW*|*MSYS*) OUT="p40cuda.dll"; CUDART_FLAG="" ;;
  # Static cudart on Linux -> no libcudart dependency on the mining rig.
  *) OUT="libp40cuda.so"; CUDART_FLAG="--cudart=static" ;;
esac

SRC=(
  csrc/capi/p40_capi.cu
  csrc/gemm/pearl_gemm_only_sm61.cu
  csrc/gemm/pearl_blake3_sm61.cu
  csrc/gemm/noising_sm61.cu
  csrc/gemm/noise_generation.cu
  csrc/blake3/blake3.cu
  csrc/gemm/rng_fill_sm61.cu
  csrc/tensor_hash/tensor_hash.cu
  csrc/gemm/noise_gemm_sm61.cu
)

# -Xcompiler -fPIC is required for a Linux shared library; -allow-unsupported-
# compiler keeps newer host GCC (e.g. 13.3 on Ubuntu 24.04) from being rejected.
"$NVCC" -shared -o "$OUT" "${SRC[@]}" \
  -I csrc -I csrc/gemm -I csrc/blake3 -I csrc/tensor_hash -I "$CUTLASS_DIR" \
  -Xcompiler -fPIC -std=c++20 --expt-relaxed-constexpr --expt-extended-lambda \
  --use_fast_math $GENCODE -O3 -DNDEBUG -DP40_NO_TORCH \
  $CUDART_FLAG -allow-unsupported-compiler ${EXTRA_NVCC_FLAGS:-}

echo "built $OUT"
