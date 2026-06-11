"""Frozen entry point for the packaged p40 Pearl miner.

PyInstaller bundles torch, the CUDA extension (p40_pearl_gemm_cuda), pearl_mining,
and the pure-python miner modules. At runtime we make sure the bundled DLL
directories are on the search path so the CUDA extension can load torch/cuda
DLLs, then hand off to luckypool_miner.main().
"""
import os
import sys


def _add_bundled_dll_dirs():
    if not getattr(sys, "frozen", False):
        return
    base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    candidates = [base, os.path.join(base, "torch", "lib")]
    # Bundled CUDA runtime DLLs (cudart/cublas) may land in a few places.
    for sub in ("", "torch/lib", "nvidia"):
        p = os.path.join(base, sub)
        if os.path.isdir(p) and p not in candidates:
            candidates.append(p)
    for p in candidates:
        if os.path.isdir(p):
            try:
                os.add_dll_directory(p)
            except (OSError, AttributeError):
                pass
        os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")


_add_bundled_dll_dirs()

from luckypool_miner import main  # noqa: E402

if __name__ == "__main__":
    main()
