"""Validate the Pascal Pearl PoW kernel against a faithful CPU reference.

The reference here mirrors miner-base/noisy_gemm.py exactly (the transcript build
in `_process_output_tile` and the keyed-BLAKE3 check in `_check_pow_target`):
per 16x16 hash tile, accumulate over k-tiles of size R the XOR-reduction of the
*cumulative* int32 partial sum, rotl-xor into a 16-word transcript at position
(k_tile_index % 16), then digest = blake3(transcript_LE, key=pow_key).

Run from the project root:  python tests/test_pearl_pow.py
"""
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import blake3
import numpy as np
import torch
import p40_pearl_gemm_cuda as _C

dev = torch.device("cuda", int(os.environ.get("GEMM_TEST_DEV", "0")))
print(f"Device: {torch.cuda.get_device_name(dev)} (cc {torch.cuda.get_device_capability(dev)})")

HT = 16
ROT = 13


def rotl32(x, n):
    x = np.uint32(x)
    return np.uint32(((int(x) << n) | (int(x) >> (32 - n))) & 0xFFFFFFFF)


def reference_digests(A, B, R, pow_key):
    """A: [m,k] int8 np, B: [k,n] int8 np. Returns dict (gi,gj)->32-byte digest."""
    m, k = A.shape
    n = B.shape[1]
    T = k // R
    out = {}
    for i in range(0, m, R):
        for j in range(0, n, R):
            tiles_h = R // HT
            tiles_w = R // HT
            transcripts = np.zeros((tiles_h, tiles_w, 16), dtype=np.uint32)
            C_block = np.zeros((R, R), dtype=np.int64)
            for t in range(T):
                p = t * R
                C_tile = A[i:i + R, p:p + R].astype(np.int64) @ B[p:p + R, j:j + R].astype(np.int64)
                C_block = C_block + C_tile
                cur = C_block.astype(np.int32)  # int32 wrap (matches kernel int32 accum)
                idx = t % 16
                for hi in range(tiles_h):
                    for wi in range(tiles_w):
                        tile = cur[hi * HT:(hi + 1) * HT, wi * HT:(wi + 1) * HT]
                        h = np.bitwise_xor.reduce(tile.reshape(-1).view(np.uint32))
                        transcripts[hi, wi, idx] = rotl32(transcripts[hi, wi, idx], ROT) ^ h
            for hi in range(tiles_h):
                for wi in range(tiles_w):
                    tb = b"".join(struct.pack("<I", int(w)) for w in transcripts[hi, wi])
                    gi = (i // HT) + hi
                    gj = (j // HT) + wi
                    out[(gi, gj)] = blake3.blake3(tb, key=pow_key).digest()
    return out


ok = True


def run(m, n, k, R, seed):
    global ok
    g = torch.Generator().manual_seed(seed)
    # "noised" int8 operands (the PoW kernel only sees noised A, B^T)
    A = torch.randint(-110, 110, (m, k), dtype=torch.int8, generator=g)
    B = torch.randint(-110, 110, (k, n), dtype=torch.int8, generator=g)
    Bt = B.t().contiguous()
    pow_key = bytes(torch.randint(0, 256, (32,), dtype=torch.uint8, generator=g).numpy().tobytes())

    ref = reference_digests(A.numpy(), B.numpy(), R, pow_key)

    key_t = torch.frombuffer(bytearray(pow_key), dtype=torch.uint8).to(dev)
    target_t = torch.full((32,), 255, dtype=torch.uint8, device=dev)  # easiest target
    digests, found, coord = _C.pearl_pow(A.to(dev), Bt.to(dev), key_t, target_t, R)
    torch.cuda.synchronize()

    tiles_w = n // HT
    mism = 0
    for (gi, gj), d in ref.items():
        kd = bytes(digests[gi * tiles_w + gj].cpu().numpy().tobytes())
        if kd != d:
            mism += 1
            if mism <= 2:
                print(f"    mismatch tile ({gi},{gj}): ref={d[:8].hex()} gpu={kd[:8].hex()}")
    passed = mism == 0
    ok = ok and passed
    print(f"  m={m} n={n} k={k} R={R}: {len(ref)} tiles, "
          f"{'ALL MATCH' if passed else f'{mism} MISMATCH'}  found={int(found[0])}")


run(128, 128, 128, 128, 1)
run(256, 256, 256, 128, 2)
run(256, 384, 512, 128, 3)
run(128, 256, 256, 64, 4)

# Found-flag check: set an impossible (hardest) target -> never found;
# set easiest target -> found on first tile.
g = torch.Generator().manual_seed(9)
A = torch.randint(-110, 110, (128, 128), dtype=torch.int8, generator=g)
Bt = torch.randint(-110, 110, (128, 128), dtype=torch.int8, generator=g)
key_t = torch.zeros(32, dtype=torch.uint8, device=dev)
hardest = torch.zeros(32, dtype=torch.uint8, device=dev)
easiest = torch.full((32,), 255, dtype=torch.uint8, device=dev)
_, f_hard, _ = _C.pearl_pow(A.to(dev), Bt.to(dev), key_t, hardest, 128)
_, f_easy, c_easy = _C.pearl_pow(A.to(dev), Bt.to(dev), key_t, easiest, 128)
torch.cuda.synchronize()
fc = (int(f_hard[0]) == 0 and int(f_easy[0]) == 1)
ok = ok and fc
print(f"  found-flag: hardest={int(f_hard[0])} easiest={int(f_easy[0])} -> {'OK' if fc else 'FAIL'}")

print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
sys.exit(0 if ok else 1)
