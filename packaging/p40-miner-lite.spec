# PyInstaller spec for the TORCH-FREE miner (target <100 MB).
#
# Bundles: miner_capi + cuda_capi + pearl_host + pool_common, the torch-free
# p40cuda.dll, cudart64, pearl_mining (Rust), numpy, blake3. NO torch.
#
# Build (from p40-pearl-gemm/), after building p40cuda.dll:
#   pyinstaller packaging/p40-miner-lite.spec --noconfirm --distpath dist --workpath build_pyi
#
# The CUDA extension is platform-specific, so build the Linux binary on Linux.

import glob
import os

from PyInstaller.utils.hooks import collect_all

ROOT = os.path.abspath(os.getcwd())
PYDIR = os.path.join(ROOT, "python")

datas, binaries, hiddenimports = [], [], []
for pkg in ("pearl_mining", "numpy", "blake3"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# Torch-free CUDA library + the CUDA runtime it links against.
for dll in glob.glob(os.path.join(ROOT, "p40cuda.dll")) + glob.glob(os.path.join(ROOT, "libp40cuda.so")):
    binaries += [(dll, ".")]
for cuda_env in ("CUDA_PATH", "CUDA_PATH_V12_8", "CUDA_PATH_V12_4"):
    cp = os.environ.get(cuda_env)
    if cp:
        for rt in glob.glob(os.path.join(cp, "bin", "cudart64_*.dll")):
            binaries += [(rt, ".")]
        break

hiddenimports += ["miner_capi", "cuda_capi", "pearl_host", "pool_common", "_dev",
                  "gateway_client"]

a = Analysis(
    [os.path.join(ROOT, "packaging", "p40_miner_lite_main.py")],
    pathex=[PYDIR, ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["torch", "tkinter", "matplotlib", "PIL", "pytest", "IPython", "scipy", "cupy"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="p40-miner", console=True)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="p40-miner")
