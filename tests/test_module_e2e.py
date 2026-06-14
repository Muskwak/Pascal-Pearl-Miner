"""End-to-end test of the built p40_pearl_gemm_cuda extension through torch.

Run from the project root (where p40_pearl_gemm_cuda*.pyd lives):
    python tests/test_module_e2e.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import p40_pearl_gemm_cuda as _C

dev = torch.device("cuda", int(os.environ.get("GEMM_TEST_DEV", "0")))
print(f"Device: {torch.cuda.get_device_name(dev)} "
      f"(cc {torch.cuda.get_device_capability(dev)})")
print("module functions:", [f for f in dir(_C) if not f.startswith("_")])

ok = True


def check(name, cond):
    global ok
    ok = ok and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


# ---- dp4a_gemm ----
M, N, K = 256, 192, 128
A = torch.randint(-63, 63, (M, K), dtype=torch.int8, device=dev)
B = torch.randint(-63, 63, (N, K), dtype=torch.int8, device=dev)
As = (torch.rand(M, device=dev) * 0.01 + 0.001).float()
Bs = (torch.rand(N, device=dev) * 0.01 + 0.001).float()
C = torch.empty(M, N, dtype=torch.float16, device=dev)
_C.dp4a_gemm(A, B, As, Bs, C, M, N, K)
torch.cuda.synchronize()
ref = (A.int().float() @ B.int().float().t()) * As[:, None] * Bs[None, :]
rel = (C.float() - ref).abs() / ref.abs().clamp_min(1e-3)
check(f"dp4a_gemm max_rel={rel.max().item():.4f}", rel.max().item() < 0.02)

# ---- noise_A ----
R = 64
EAL = torch.randint(-32, 32, (M, R), dtype=torch.int8, device=dev)
EAR = torch.zeros(K, R, dtype=torch.int8, device=dev)   # R-major sparse +-1
EBL_k = torch.zeros(R, K, dtype=torch.int8, device=dev)  # K-major sparse +-1
for k in range(K):
    r0, r1 = torch.randint(0, R, (2,))
    if r1 == r0:
        r1 = (r0 + 1) % R
    EAR[k, r0] = 1; EAR[k, r1] = -1
    EBL_k[r0, k] = 1; EBL_k[r1, k] = -1
ApEA = torch.empty(M, K, dtype=torch.int8, device=dev)
AxEBL = torch.empty(M, R, dtype=torch.int32, device=dev)
_C.noise_A(A, EAL, EAR, EBL_k, ApEA, AxEBL, M, K, R)
torch.cuda.synchronize()
# int matmul isn't implemented on CUDA; use float (exact for these magnitudes).
ref_ApEA = (A.float() + (EAL.float() @ EAR.float().t())).clamp(-128, 127)
ref_AxEBL = (A.float() @ EBL_k.float().t())
check("noise_A ApEA", torch.equal(ApEA.float(), ref_ApEA))
check("noise_A AxEBL", torch.equal(AxEBL.float(), ref_AxEBL))

# ---- noise_B ----
EBR = torch.randint(-32, 32, (N, R), dtype=torch.int8, device=dev)
EBL_r = torch.zeros(K, R, dtype=torch.int8, device=dev)  # R-major sparse
EAR_k = torch.zeros(R, K, dtype=torch.int8, device=dev)  # K-major sparse
for k in range(K):
    r0, r1 = torch.randint(0, R, (2,))
    if r1 == r0:
        r1 = (r0 + 1) % R
    EBL_r[k, r0] = 1; EBL_r[k, r1] = -1
    EAR_k[r0, k] = 1; EAR_k[r1, k] = -1
BpEB = torch.empty(N, K, dtype=torch.int8, device=dev)
EARxBpEB = torch.empty(N, R, dtype=torch.int32, device=dev)
_C.noise_B(B, EBR, EAR_k, EBL_r, BpEB, EARxBpEB, N, K, R)
torch.cuda.synchronize()
ref_BpEB = (B.float() + (EBR.float() @ EBL_r.float().t())).clamp(-128, 127)
ref_EARxBpEB = ref_BpEB @ EAR_k.float().t()
check("noise_B BpEB", torch.equal(BpEB.float(), ref_BpEB))
check("noise_B EARxBpEB", torch.equal(EARxBpEB.float(), ref_EARxBpEB))

# ---- denoise_converter ----  (AxEBL /1<<14, EARxBpEB /1<<12)
AxEBL_h = torch.empty(M, R, dtype=torch.float16, device=dev)
EARxBpEB_h = torch.empty(N, R, dtype=torch.float16, device=dev)
_C.denoise_converter(EARxBpEB, AxEBL, EARxBpEB_h, AxEBL_h, M, N, R)
torch.cuda.synchronize()
ref_axebl_h = (AxEBL.float() / (1 << 14)).half()
ref_ear_h = (EARxBpEB.float() / (1 << 12)).half()
check("denoise AxEBL/2^14", torch.equal(AxEBL_h, ref_axebl_h))
check("denoise EARxBpEB/2^12", torch.equal(EARxBpEB_h, ref_ear_h))

# ---- inner_hash ----  (smoke: runs and returns a uint32)
buf = torch.randint(0, 2**31, (64,), dtype=torch.int32, device=dev).view(torch.int32)
h = _C.inner_hash(buf, 1)
torch.cuda.synchronize()
check(f"inner_hash returns scalar (val={int(h.view(torch.int32)[0]) & 0xffffffff:#010x})",
      h.numel() == 1)

print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
sys.exit(0 if ok else 1)
