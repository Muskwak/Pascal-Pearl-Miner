"""Phase 2 gate: the torch-free noise-apply (p40_noise_apply_A/B in the DLL) must
produce the SAME noised operands as the torch `_imatmul_i8` path the live miner
uses (and which lands accepted shares).

Passes torch CUDA tensors' data_ptr() straight to the DLL (shared primary
context, zero-copy). Run: python tests/test_capi_phase2.py
"""
import ctypes
import os
import sys

import torch

import p40_pearl_gemm_cuda as _C

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DLL = os.path.join(HERE, "p40cuda.dll" if sys.platform == "win32" else "libp40cuda.so")
lib = ctypes.CDLL(DLL)
lib.p40_noise_apply_A.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3
lib.p40_noise_apply_B.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3
lib.p40_sync.argtypes = []

dev = torch.device("cuda", 0)


def imatmul_i8(X, Y):
    return (X.float() @ Y.float()).round().to(torch.int32).to(torch.int8)


def run(M, N, K, R, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randint(-63, 63, (M, K), dtype=torch.int8, generator=g).to(dev)
    B = torch.randint(-63, 63, (K, N), dtype=torch.int8, generator=g).to(dev)
    key = torch.randint(0, 256, (32,), dtype=torch.uint8, generator=g).to(dev)
    keyB = torch.randint(0, 256, (32,), dtype=torch.uint8, generator=g).to(dev)

    EAL, EAR, EBL, EBR = _C.noise_gen(key, keyB, M, N, K, R)
    E_BLt = EBL.t().contiguous()  # [R,K]

    # ---- A side ----
    A_ns_torch = (A.int() + imatmul_i8(EAL, EAR).int()).to(torch.int8)
    EAR_t = EAR.t().contiguous()  # noise_A wants EAR as [K,R]
    EBL_t = EBL.t().contiguous()  # and EBL as [R,K] (for the AxEBL side product)
    ApEA = torch.empty((M, K), dtype=torch.int8, device=dev)
    AxEBL = torch.empty((M, R), dtype=torch.int32, device=dev)
    lib.p40_noise_apply_A(A.data_ptr(), EAL.data_ptr(), EAR_t.data_ptr(),
                          EBL_t.data_ptr(), ApEA.data_ptr(), AxEBL.data_ptr(), M, K, R)
    lib.p40_sync()
    a_ok = torch.equal(ApEA, A_ns_torch)

    # ---- B side: mimic the miner's column operand for the first RS=N block ----
    Bt_cols = B.t().contiguous()  # [N,K] = B^T (full, c0=0)
    Bt_ns_torch = (Bt_cols.int() + imatmul_i8(EBR, E_BLt).int()).to(torch.int8)
    BpEB = torch.empty((N, K), dtype=torch.int8, device=dev)
    EARxBpEB = torch.empty((N, R), dtype=torch.int32, device=dev)
    # noise_B: EAR as [R,K] (=EAR torch), EBL as [K,R] (=EBL torch)
    lib.p40_noise_apply_B(Bt_cols.data_ptr(), EBR.data_ptr(), EAR.data_ptr(),
                          EBL.data_ptr(), BpEB.data_ptr(), EARxBpEB.data_ptr(), N, K, R)
    lib.p40_sync()
    b_ok = torch.equal(BpEB, Bt_ns_torch)

    print(f"  [{'PASS' if a_ok else 'FAIL'}] noise_A  M={M} K={K} R={R}  "
          f"(max|diff|={(ApEA.int()-A_ns_torch.int()).abs().max().item()})")
    print(f"  [{'PASS' if b_ok else 'FAIL'}] noise_B  N={N} K={K} R={R}  "
          f"(max|diff|={(BpEB.int()-Bt_ns_torch.int()).abs().max().item()})")
    return a_ok and b_ok


ok = True
print("--- Phase 2: torch-free noise-apply vs torch _imatmul_i8 ---")
ok &= run(256, 256, 256, 256, 1)
ok &= run(512, 256, 4096, 256, 2)
print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
sys.exit(0 if ok else 1)
