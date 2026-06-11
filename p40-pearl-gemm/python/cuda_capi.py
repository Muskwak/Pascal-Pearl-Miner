"""Torch-free CUDA backend: stdlib ctypes wrapper over p40cuda.dll.

Provides device-memory management and the Pearl kernels (noise_gen, noise_A/B,
transpose_i8, pearl_pow_split) with no torch / cuda-python / cupy dependency.
"""
from __future__ import annotations

import ctypes
import os
import sys

import numpy as np

_NAME = "p40cuda.dll" if sys.platform == "win32" else "libp40cuda.so"
# Search next to this file, the package root, and the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    os.path.join(_HERE, _NAME),
    os.path.join(os.path.dirname(_HERE), _NAME),
    os.path.join(getattr(sys, "_MEIPASS", _HERE), _NAME),
]
_path = next((p for p in _CANDIDATES if os.path.exists(p)), _NAME)

# Make cudart findable: the directory holding p40cuda.dll (where a bundled
# cudart64_*.dll sits when frozen), then the CUDA toolkit bin from the env.
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    _dirs = [os.path.dirname(os.path.abspath(_path))]
    for _env in ("CUDA_PATH", "CUDA_PATH_V12_8", "CUDA_PATH_V12_4"):
        _v = os.environ.get(_env)
        if _v:
            _dirs.append(os.path.join(_v, "bin"))
    for _d in _dirs:
        if os.path.isdir(_d):
            try:
                os.add_dll_directory(_d)
            except OSError:
                pass

_lib = ctypes.CDLL(_path)

_VP = ctypes.c_void_p
_I = ctypes.c_int
_Z = ctypes.c_size_t

_lib.p40_malloc.argtypes = [ctypes.POINTER(_VP), _Z]
_lib.p40_free.argtypes = [_VP]
_lib.p40_memcpy_htod.argtypes = [_VP, _VP, _Z]
_lib.p40_memcpy_dtoh.argtypes = [_VP, _VP, _Z]
_lib.p40_memset.argtypes = [_VP, _I, _Z]
_lib.p40_sync.argtypes = []
_lib.p40_transpose_i8.argtypes = [_VP, _VP, _I, _I, _I, _I]
_lib.p40_noise_gen.argtypes = [_VP, _VP, _VP, _VP, _VP, _VP, _I, _I, _I, _I]
_lib.p40_noise_apply_A.argtypes = [_VP, _VP, _VP, _VP, _VP, _VP, _I, _I, _I]
_lib.p40_noise_apply_B.argtypes = [_VP, _VP, _VP, _VP, _VP, _VP, _I, _I, _I]
_lib.p40_pearl_pow_split.argtypes = [_VP, _VP, _I, _I, _I, _I, _VP, _VP, _VP, _VP, _VP, _I]


def _chk(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what} failed (cuda error {rc})")


class DBuf:
    """A device allocation. `ptr` is a c_void_p usable as a kernel argument."""

    def __init__(self, nbytes: int):
        self.nbytes = int(nbytes)
        self.ptr = _VP()
        _chk(_lib.p40_malloc(ctypes.byref(self.ptr), self.nbytes), "malloc")

    def offset(self, byte_off: int) -> _VP:
        return _VP(self.ptr.value + int(byte_off))

    def from_host(self, arr: np.ndarray):
        a = np.ascontiguousarray(arr)
        _chk(_lib.p40_memcpy_htod(self.ptr, a.ctypes.data_as(_VP), a.nbytes), "htod")

    def to_host(self, arr: np.ndarray):
        _chk(_lib.p40_memcpy_dtoh(arr.ctypes.data_as(_VP), self.ptr, arr.nbytes), "dtoh")

    def memset(self, v: int):
        _chk(_lib.p40_memset(self.ptr, v, self.nbytes), "memset")

    def free(self):
        if self.ptr:
            _lib.p40_free(self.ptr)
            self.ptr = _VP()


def sync():
    _chk(_lib.p40_sync(), "sync")


def transpose_i8(src, dst, rows, cols, src_ld, col_off):
    _chk(_lib.p40_transpose_i8(_as(src), _as(dst), rows, cols, src_ld, col_off), "transpose")


def noise_gen(EAL, EAR, EBL, EBR, key_A, key_B, m, n, k, R):
    _chk(_lib.p40_noise_gen(_as(EAL), _as(EAR), _as(EBL), _as(EBR), _as(key_A),
                            _as(key_B), m, n, k, R), "noise_gen")


def noise_apply_A(A, EAL, EAR_t, EBL_t, ApEA, AxEBL, M, K, R):
    _chk(_lib.p40_noise_apply_A(_as(A), _as(EAL), _as(EAR_t), _as(EBL_t),
                                _as(ApEA), _as(AxEBL), M, K, R), "noise_A")


def noise_apply_B(B, EBR, EAR, EBL, BpEB, EARxBpEB, N, K, R):
    _chk(_lib.p40_noise_apply_B(_as(B), _as(EBR), _as(EAR), _as(EBL),
                                _as(BpEB), _as(EARxBpEB), N, K, R), "noise_B")


def pearl_pow_split(A, Bt, m, n, k, R, key, target, digests, found, coord, variant):
    _chk(_lib.p40_pearl_pow_split(_as(A), _as(Bt), m, n, k, R, _as(key), _as(target),
                                  _as(digests), _as(found), _as(coord), variant),
         "pearl_pow_split")


def _as(x):
    """Accept a DBuf, a c_void_p (e.g. from DBuf.offset), or a raw int address."""
    if isinstance(x, DBuf):
        return x.ptr
    return x
