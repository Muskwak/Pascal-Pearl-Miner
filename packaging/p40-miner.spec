# PyInstaller spec for the p40 Pearl miner (closed-source binary distribution).
#
# Build (run from the p40-pearl-gemm/ directory):
#   Windows:  pyinstaller packaging/p40-miner.spec --noconfirm --distpath dist --workpath build_pyi
#   Linux:    pyinstaller packaging/p40-miner.spec --noconfirm --distpath dist --workpath build_pyi
#
# Produces a onedir bundle at dist/p40-miner/ containing p40-miner(.exe) plus all
# dependencies (torch, the CUDA extension, pearl_mining, blake3, numpy). The .pyc
# sources are bundled, not the .py files, so the miner ships without source.
#
# NOTE: the CUDA extension and torch are platform + python-version specific, so
# the Linux binary MUST be built on Linux (PyInstaller does not cross-compile).

import glob
import os

from PyInstaller.utils.hooks import collect_all

ROOT = os.path.abspath(os.getcwd())
PYDIR = os.path.join(ROOT, "python")

datas, binaries, hiddenimports = [], [], []
for pkg in ("torch", "pearl_mining", "numpy", "blake3"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# Bundle the compiled CUDA extension (.pyd on Windows, .so on Linux).
ext = glob.glob(os.path.join(ROOT, "build", "lib.*", "p40_pearl_gemm_cuda.*"))
ext += glob.glob(os.path.join(PYDIR, "p40_pearl_gemm_cuda.*"))
binaries += [(e, ".") for e in ext]

# Bundle the pure-python miner modules (as compiled .pyc inside the archive).
py_modules = [f[:-3] for f in os.listdir(PYDIR) if f.endswith(".py") and not f.startswith("_")]
hiddenimports += py_modules + ["p40_pearl_gemm_cuda"]

a = Analysis(
    [os.path.join(ROOT, "packaging", "p40_miner_main.py")],
    pathex=[PYDIR, ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "PIL", "pytest", "IPython"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="p40-miner",
    console=True,
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="p40-miner",
)
