"""Phase 1 gate: the torch-free C ABI (p40cuda.dll via ctypes) must produce the
SAME pearl_pow_split digests as the torch extension. Proves the kernels work
bit-exact with no torch in the compute path.

Run: python tests/test_capi_phase1.py
"""
import ctypes
import os
import sys

import numpy as np
import torch  # only to produce the reference + initial CUDA context

import p40_pearl_gemm_cuda as _C  # torch path (reference)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DLL = os.path.join(HERE, "p40cuda.dll" if sys.platform == "win32" else "libp40cuda.so")
lib = ctypes.CDLL(DLL)

lib.p40_malloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
lib.p40_memcpy_htod.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
lib.p40_memcpy_dtoh.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
lib.p40_memset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
lib.p40_sync.argtypes = []
lib.p40_pearl_pow_split.argtypes = (
    [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
     ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
     ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
)


def dmalloc(nbytes):
    p = ctypes.c_void_p()
    assert lib.p40_malloc(ctypes.byref(p), nbytes) == 0
    return p


def htod(dptr, arr):
    assert lib.p40_memcpy_htod(dptr, arr.ctypes.data_as(ctypes.c_void_p), arr.nbytes) == 0


def dtoh(arr, dptr):
    assert lib.p40_memcpy_dtoh(arr.ctypes.data_as(ctypes.c_void_p), dptr, arr.nbytes) == 0


def run(m, n, k, R, seed):
    g = torch.Generator().manual_seed(seed)
    A = torch.randint(-110, 110, (m, k), dtype=torch.int8, generator=g)
    B = torch.randint(-110, 110, (k, n), dtype=torch.int8, generator=g)
    Bt = B.t().contiguous()
    key = bytes(torch.randint(0, 256, (32,), dtype=torch.uint8, generator=g).numpy().tobytes())
    target = b"\xff" * 32

    dev = torch.device("cuda", 0)
    key_t = torch.frombuffer(bytearray(key), dtype=torch.uint8).to(dev)
    tgt_t = torch.frombuffer(bytearray(target), dtype=torch.uint8).to(dev)
    ref_digests, _, _ = _C.pearl_pow_split(A.to(dev), Bt.to(dev), key_t, tgt_t, R, 1)
    ref = ref_digests.cpu().numpy()

    An = np.ascontiguousarray(A.numpy())
    Btn = np.ascontiguousarray(Bt.numpy())
    keyn = np.frombuffer(key, dtype=np.uint8).copy()
    tgtn = np.frombuffer(target, dtype=np.uint8).copy()
    num_tiles = (m // 16) * (n // 16)

    dA, dBt = dmalloc(An.nbytes), dmalloc(Btn.nbytes)
    dkey, dtgt = dmalloc(32), dmalloc(32)
    dtb = dmalloc(num_tiles * 16 * 4)
    ddig, dfound, dcoord = dmalloc(num_tiles * 32), dmalloc(4), dmalloc(8)
    htod(dA, An); htod(dBt, Btn); htod(dkey, keyn); htod(dtgt, tgtn)
    lib.p40_memset(dfound, 0, 4); lib.p40_memset(dcoord, 255, 8)

    r = lib.p40_pearl_pow_split(dA, dBt, m, n, k, R, dkey, dtgt, dtb, ddig, dfound, dcoord, 1)
    lib.p40_sync()
    assert r == 0, f"p40_pearl_pow_split returned {r}"

    out = np.empty((num_tiles, 32), dtype=np.uint8)
    dtoh(out, ddig)
    match = np.array_equal(out, ref)
    print(f"  [{'PASS' if match else 'FAIL'}] capi vs torch  m={m} n={n} k={k} R={R}: "
          f"{num_tiles} tiles")
    return match


ok = True
print("--- Phase 1: torch-free C ABI vs torch pearl_pow_split ---")
ok &= run(64, 128, 256, 256, 1)
ok &= run(128, 64, 256, 256, 2)
ok &= run(256, 256, 256, 256, 3)
print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
sys.exit(0 if ok else 1)
