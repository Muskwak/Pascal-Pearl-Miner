"""End-to-end Pascal mining pipeline self-test (option A).

Constructs a job, picks int7 A/B, runs the full commit -> noise -> noised GEMM ->
PoW pipeline on the P40, and independently verifies a found block's transcript.
Run from the project root:  python tests/test_pipeline_e2e.py
"""
import os
import struct
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)                       # for p40_pearl_gemm_cuda*.pyd
sys.path.insert(0, os.path.join(_root, "python"))  # for pearl_miner.py

import blake3
import numpy as np
import torch

import pearl_miner as pm
from mining_config import MiningConfiguration, PeriodicPattern, patterns_for_tile

dev = torch.device("cuda", int(os.environ.get("GEMM_TEST_DEV", "0")))
print(f"Device: {torch.cuda.get_device_name(dev)} (cc {torch.cuda.get_device_capability(dev)})")

R = 128
HT = 16
ok = True


def verify_tile(A_noised, B_noised, seed_A, target, row, col):
    """Recompute one 16x16 tile's transcript from scratch and check <= target."""
    A = A_noised.cpu().numpy().astype(np.int64)
    B = B_noised.cpu().numpy().astype(np.int64)
    k = A.shape[1]
    T = k // R
    transcript = np.zeros(16, dtype=np.uint32)
    Csum = np.zeros((HT, HT), dtype=np.int64)
    for t in range(T):
        p = t * R
        Csum = Csum + A[row:row + HT, p:p + R] @ B[p:p + R, col:col + HT]
        cur = Csum.astype(np.int32).reshape(-1).view(np.uint32)
        h = np.bitwise_xor.reduce(cur)
        idx = t % 16
        x = transcript[idx]
        transcript[idx] = np.uint32(((int(x) << 13) | (int(x) >> 19)) & 0xFFFFFFFF) ^ h
    tb = b"".join(struct.pack("<I", int(w)) for w in transcript)
    digest = blake3.blake3(tb, key=seed_A).digest()
    return int.from_bytes(digest, "little") <= target


def check(name, cond):
    global ok
    ok = ok and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


# A job: arbitrary header template + target. The miner picks A, B.
g = torch.Generator().manual_seed(2024)
header = bytes(torch.randint(0, 256, (80,), dtype=torch.uint8, generator=g).numpy().tobytes())
m, n, k = 256, 256, 256
# int7 data range for noise_range=128 is [-64, 63]
A = torch.randint(-64, 64, (m, k), dtype=torch.int8, generator=g).to(dev)
B = torch.randint(-64, 64, (k, n), dtype=torch.int8, generator=g).to(dev)

# Build a MiningConfiguration matching the matmul dims
rp, cp = patterns_for_tile(m, n)
mining_config = MiningConfiguration(common_dim=k, rank=R, rows_pattern=rp, cols_pattern=cp)

# Recompute the noised operands the way mine_once does, so we can verify a hit.
key = pm.derive_key(header, mining_config)
seed_A, seed_B = pm.commitment_hashes(A, B, key)
E_AL, E_AR, E_BL, E_BR = pm.generate_noise(seed_A, seed_B, m, k, n, R)

# Noise must stay in [-64, 64) so the int8 noised matrices don't overflow.
E_A = E_AL.int() @ E_AR.int()
E_B = E_BL.int() @ E_BR.int()
check(f"noise A range [{E_A.min()},{E_A.max()}] in [-64,64)", E_A.min() >= -64 and E_A.max() < 64)
check(f"noise B range [{E_B.min()},{E_B.max()}] in [-64,64)", E_B.min() >= -64 and E_B.max() < 64)

E_AL, E_AR = E_AL.to(dev), E_AR.to(dev)
E_BL, E_BR = E_BL.to(dev), E_BR.to(dev)
A_noised, B_noised = pm.noised_operands(A, B, E_AL, E_AR, E_BL, E_BR)
check("A_noised dtype/shape", A_noised.dtype == torch.int8 and tuple(A_noised.shape) == (m, k))

# Easiest target -> must find a block; verify the winning tile independently.
easy = 2**256 - 1
res = pm.run_pow(A_noised, B_noised, seed_A, easy, R)
check(f"found at easy target (tile {res.row},{res.col})", res.found)
if res.found:
    check("winning tile transcript verified <= target",
          verify_tile(A_noised, B_noised, seed_A, easy, res.row, res.col))

# Hardest target -> must NOT find.
res0 = pm.run_pow(A_noised, B_noised, seed_A, 0, R)
check("no block at hardest target (0)", not res0.found)

# Full mine_once convenience path runs and agrees on found-ness at easy target.
res2 = pm.mine_once(header, easy, A, B, mining_config=mining_config)
check("mine_once finds at easy target", res2.found)

# A realistic-ish target: scan a few nonces, expect to eventually find.
found_any = False
for nonce in range(8):
    hdr = header[:-4] + struct.pack("<I", nonce)
    medium = (2**256 - 1) // (1 << 40)  # ~1 in 2^40 per tile; many tiles per matmul
    r = pm.mine_once(hdr, medium, A, B, mining_config=mining_config)
    if r.found:
        found_any = True
        break
print(f"  [info] medium-target scan over 8 nonces: found={found_any}")

print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
sys.exit(0 if ok else 1)
