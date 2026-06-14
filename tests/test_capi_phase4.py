"""Phase 4 gate (offline): the FULL torch-free pipeline (cuda_capi + pearl_host)
must produce the same pearl_pow_split digests as the torch pipeline, end to end:
host commitments -> noise_gen -> transposes -> noise_A/B -> pearl_pow_split.

Run: python tests/test_capi_phase4.py
"""
import os
import sys

import blake3
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import p40_pearl_gemm_cuda as _C   # torch reference
import pearl_miner as tmin
import cuda_capi as cc             # torch-free backend
import pearl_host

dev = torch.device("cuda", 0)
M = N = 256
K = 4096
R = 256
VAR = 0


def imatmul(X, Y):
    return (X.float() @ Y.float()).round().to(torch.int32).to(torch.int8)


g = torch.Generator().manual_seed(11)
A_t = torch.randint(-64, 63, (M, K), dtype=torch.int8, generator=g)
B_t = torch.randint(-64, 63, (K, N), dtype=torch.int8, generator=g)
A = A_t.numpy()
B = B_t.numpy()
key = blake3.blake3(b"phase4").digest()

# Commitments on host; cross-check vs the torch/GPU commitment.
a_seed, b_seed = pearl_host.commitment_hashes(A, B, key)
aS, bS = tmin.commitment_hashes(A_t.to(dev), B_t.to(dev), key)
assert bytes(aS) == a_seed and bytes(bS) == b_seed, "commitment mismatch"

# ---- torch reference pipeline ----
aS_t = torch.frombuffer(bytearray(a_seed), dtype=torch.uint8).to(dev)
bS_t = torch.frombuffer(bytearray(b_seed), dtype=torch.uint8).to(dev)
EAL, EAR, EBL, EBR = _C.noise_gen(aS_t, bS_t, M, N, K, R)
E_BLt = EBL.t().contiguous()
A_ns = (A_t.to(dev).int() + imatmul(EAL, EAR).int()).to(torch.int8)
Bt_ns = (B_t.to(dev).t().contiguous().int() + imatmul(EBR, E_BLt).int()).to(torch.int8)
tgt = torch.full((32,), 255, dtype=torch.uint8, device=dev)
ref_dig, _, _ = _C.pearl_pow_split(A_ns, Bt_ns, aS_t, tgt, R, VAR)
ref = ref_dig.cpu().numpy()

# ---- torch-free pipeline ----
dA, dB = cc.DBuf(M * K), cc.DBuf(K * N)
dA.from_host(A); dB.from_host(B)
dEAL, dEAR, dEBL, dEBR = cc.DBuf(M * R), cc.DBuf(R * K), cc.DBuf(K * R), cc.DBuf(N * R)
dka, dkb = cc.DBuf(32), cc.DBuf(32)
dka.from_host(np.frombuffer(a_seed, np.uint8).copy())
dkb.from_host(np.frombuffer(b_seed, np.uint8).copy())
cc.noise_gen(dEAL, dEAR, dEBL, dEBR, dka, dkb, M, N, K, R)

dEAR_t, dEBL_t = cc.DBuf(K * R), cc.DBuf(R * K)
cc.transpose_i8(dEAR, dEAR_t, R, K, K, 0)   # [R,K]->[K,R]
cc.transpose_i8(dEBL, dEBL_t, K, R, R, 0)   # [K,R]->[R,K]

dApEA, dAxEBL = cc.DBuf(M * K), cc.DBuf(M * R * 4)
cc.noise_apply_A(dA, dEAL, dEAR_t, dEBL_t, dApEA, dAxEBL, M, K, R)

dBt = cc.DBuf(N * K)
cc.transpose_i8(dB, dBt, K, N, N, 0)        # B[K,N]->B^T[N,K]
dBpEB, dEARx = cc.DBuf(N * K), cc.DBuf(N * R * 4)
cc.noise_apply_B(dBt, dEBR, dEAR, dEBL, dBpEB, dEARx, N, K, R)

num_tiles = (M // 16) * (N // 16)
dtb = cc.DBuf(num_tiles * 16 * 4)
ddig, dfound, dcoord = cc.DBuf(num_tiles * 32), cc.DBuf(4), cc.DBuf(8)
dtgt = cc.DBuf(32); dtgt.memset(-1)         # 0xFF target
dfound.memset(0)
cc.pearl_pow_split(dApEA, dBpEB, M, N, K, R, dka, dtgt, dtb, ddig, dfound, dcoord, VAR)
cc.sync()
out = np.empty((num_tiles, 32), dtype=np.uint8)
ddig.to_host(out)

ok = np.array_equal(out, ref)
print(f"  [{'PASS' if ok else 'FAIL'}] full torch-free chain vs torch  "
      f"M={M} N={N} K={K} R={R}: {num_tiles} tiles")
print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
sys.exit(0 if ok else 1)
