#!/usr/bin/env bash
# Build the closed-source Linux binary (dist/p40-miner/p40-miner).
# MUST be run on Linux (PyInstaller does not cross-compile from Windows) on a
# machine with the CUDA toolkit + a matching PyTorch (CUDA) install.
#
# Run from the p40-pearl-gemm directory:  bash packaging/build_linux.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[1/3] Building the torch-free CUDA library (libp40cuda.so) if missing..."
if [ ! -f libp40cuda.so ]; then
  : "${CUTLASS_DIR:?Set CUTLASS_DIR to the include dir containing cutlass/ and cute/}"
  bash packaging/build_capi.sh
fi

echo "[2/3] Installing PyInstaller if needed..."
python -c "import PyInstaller" 2>/dev/null || python -m pip install pyinstaller

echo "[3/3] Freezing the torch-free miner..."
pyinstaller packaging/p40-miner-lite.spec --noconfirm --distpath dist --workpath build_pyi

echo
echo "Done (~60 MB). Share the whole folder:  dist/p40-miner/"
echo "Run with:  ./dist/p40-miner/p40-miner --wallet prl1YOURWALLET --worker p40"
