#!/usr/bin/env bash
# Build the closed-source Linux binary (dist/p40-miner/p40-miner).
# MUST be run on Linux (PyInstaller does not cross-compile from Windows) on a
# machine with the CUDA toolkit + a matching PyTorch (CUDA) install.
#
# Run from the p40-pearl-gemm directory:  bash packaging/build_linux.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[1/3] Building the CUDA extension (sm_61) if missing..."
if ! ls build/lib.*/p40_pearl_gemm_cuda.* >/dev/null 2>&1; then
  # CUTLASS headers are required to compile the extension; set CUTLASS_DIR.
  : "${CUTLASS_DIR:?Set CUTLASS_DIR to the include dir containing cutlass/ and cute/}"
  python setup.py build_ext --inplace
fi

echo "[2/3] Installing PyInstaller if needed..."
python -c "import PyInstaller" 2>/dev/null || python -m pip install pyinstaller

echo "[3/3] Freezing the miner..."
pyinstaller packaging/p40-miner.spec --noconfirm --distpath dist --workpath build_pyi

echo
echo "Done. Share the whole folder:  dist/p40-miner/"
echo "Run with:  ./dist/p40-miner/p40-miner --wallet prl1YOURWALLET --worker p40"
