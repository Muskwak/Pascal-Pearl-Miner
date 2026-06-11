"""Validate the split pearl_pow pipeline (GEMM-only + BLAKE3-only) against the
fused pearl_pow_fused kernel. Both must produce bit-identical digests.

Run:  uv run python tests/test_pearl_pow_split.py
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
TILESZ = 64  # bytes per transcript (16 x uint32)
ok = True


def check(name, cond):
    global ok
    ok = ok and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


# ---------------------------------------------------------------------------
# Reference transcript + digest (matches pearl_pow_fused semantics).
# ---------------------------------------------------------------------------
def reference_transcripts(A, B, R):
    """A: [m,k] int8 np, B: [k,n] int8 np.
    Returns dict (gi,gj) -> 16-word transcript (list of uint32).
    """
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
                cur = C_block.astype(np.int32)
                idx = t % 16
                for hi in range(tiles_h):
                    for wi in range(tiles_w):
                        tile = cur[hi * HT:(hi + 1) * HT, wi * HT:(wi + 1) * HT]
                        h = np.bitwise_xor.reduce(tile.reshape(-1).view(np.uint32))
                        x = transcripts[hi, wi, idx]
                        transcripts[hi, wi, idx] = np.uint32(((int(x) << ROT) | (int(x) >> (32 - ROT))) & 0xFFFFFFFF) ^ h
            for hi in range(tiles_h):
                for wi in range(tiles_w):
                    gi = (i // HT) + hi
                    gj = (j // HT) + wi
                    out[(gi, gj)] = [int(w) for w in transcripts[hi, wi]]
    return out


# ---------------------------------------------------------------------------
# Test shapes: must satisfy m%64==0, n%64==0 (WM=WN=4 requirement).
# ---------------------------------------------------------------------------
def run_gemm_only_vs_fused(m, n, k, R, seed):
    """Compare the GEMM-only transcript buffer against fused kernel digests.

    Strategy:
      1. Run pearl_gemm_only -> transcript_buffer.
      2. For each tile, manually BLAKE3 the transcript -> digest.
      3. Run pearl_pow_fused -> fused_digests.
      4. Compare: every tile's digest must match.
    """
    global ok
    g = torch.Generator().manual_seed(seed)
    A = torch.randint(-110, 110, (m, k), dtype=torch.int8, generator=g)
    B = torch.randint(-110, 110, (k, n), dtype=torch.int8, generator=g)
    Bt = B.t().contiguous()
    pow_key = bytes(torch.randint(0, 256, (32,), dtype=torch.uint8, generator=g).numpy().tobytes())

    num_tiles = (m // HT) * (n // HT)
    tiles_w = n // HT

    # -- fused kernel digests --
    key_t = torch.frombuffer(bytearray(pow_key), dtype=torch.uint8).to(dev)
    target_t = torch.full((32,), 255, dtype=torch.uint8, device=dev)
    fused_digests, _, _ = _C.pearl_pow_fused(A.to(dev), Bt.to(dev), key_t, target_t, R, 0)

    # -- GEMM-only transcript buffer --
    tb = torch.zeros(num_tiles, 16, dtype=torch.int32, device=dev)
    _C.pearl_gemm_only(A.to(dev), Bt.to(dev), tb, R, 0)
    torch.cuda.synchronize()

    # -- Manually BLAKE3 each transcript --
    mism = 0
    tb_cpu = tb.cpu().numpy()
    for tile_id in range(num_tiles):
        transcript = [np.uint32(tb_cpu[tile_id, i]) for i in range(16)]
        tb_bytes = b"".join(struct.pack("<I", int(w)) for w in transcript)
        expected_digest = blake3.blake3(tb_bytes, key=pow_key).digest()
        fused_d = bytes(fused_digests[tile_id].cpu().numpy().tobytes())
        if expected_digest != fused_d:
            mism += 1
            if mism <= 2:
                gi = (tile_id // tiles_w) * HT
                gj = (tile_id % tiles_w) * HT
                print(f"    mismatch tile ({gi},{gj}) gemm-only={expected_digest[:8].hex()} fused={fused_d[:8].hex()}")

    passed = mism == 0
    ok = ok and passed
    print(f"  gemm-only m={m} n={n} k={k} R={R}: {num_tiles} tiles, "
          f"{'ALL MATCH' if passed else f'{mism} MISMATCH'}")


def run_split_vs_fused_digest(m, n, k, R, seed):
    """Compare pearl_pow_split digests vs pearl_pow_fused digests (end-to-end)."""
    global ok
    g = torch.Generator().manual_seed(seed)
    A = torch.randint(-110, 110, (m, k), dtype=torch.int8, generator=g)
    B = torch.randint(-110, 110, (k, n), dtype=torch.int8, generator=g)
    Bt = B.t().contiguous()
    pow_key = bytes(torch.randint(0, 256, (32,), dtype=torch.uint8, generator=g).numpy().tobytes())

    num_tiles = (m // HT) * (n // HT)
    key_t = torch.frombuffer(bytearray(pow_key), dtype=torch.uint8).to(dev)
    target_t = torch.full((32,), 255, dtype=torch.uint8, device=dev)

    fused_digests, _, _ = _C.pearl_pow_fused(A.to(dev), Bt.to(dev), key_t, target_t, R, 0)
    split_digests, _, _ = _C.pearl_pow_split(A.to(dev), Bt.to(dev), key_t, target_t, R, 0)
    torch.cuda.synchronize()

    mism = 0
    for tile_id in range(num_tiles):
        fd = bytes(fused_digests[tile_id].cpu().numpy().tobytes())
        sd = bytes(split_digests[tile_id].cpu().numpy().tobytes())
        if fd != sd:
            mism += 1
            if mism <= 2:
                tiles_w = n // HT
                gi = (tile_id // tiles_w) * HT
                gj = (tile_id % tiles_w) * HT
                print(f"    mismatch tile ({gi},{gj}) split={sd[:8].hex()} fused={fd[:8].hex()}")

    passed = mism == 0
    ok = ok and passed
    print(f"  split-vs-fused m={m} n={n} k={k} R={R}: {num_tiles} tiles, "
          f"{'ALL MATCH' if passed else f'{mism} MISMATCH'}")


def run_found_flag(m, n, k, R, seed):
    """Found-flag: hardest -> never, easiest -> always found."""
    global ok
    g = torch.Generator().manual_seed(seed)
    A = torch.randint(-110, 110, (m, k), dtype=torch.int8, generator=g)
    B = torch.randint(-110, 110, (k, n), dtype=torch.int8, generator=g)
    Bt = B.t().contiguous()

    key_t = torch.zeros(32, dtype=torch.uint8, device=dev)
    hardest = torch.zeros(32, dtype=torch.uint8, device=dev)
    easiest = torch.full((32,), 255, dtype=torch.uint8, device=dev)

    _, f_hard, _ = _C.pearl_pow_split(A.to(dev), Bt.to(dev), key_t, hardest, R, 0)
    _, f_easy, c_easy = _C.pearl_pow_split(A.to(dev), Bt.to(dev), key_t, easiest, R, 0)
    torch.cuda.synchronize()

    fc = (int(f_hard[0]) == 0 and int(f_easy[0]) == 1)
    ok = ok and fc
    print(f"  found-flag (split): hardest={int(f_hard[0])} easiest={int(f_easy[0])} "
          f"coord=({int(c_easy[0])},{int(c_easy[1])}) -> {'OK' if fc else 'FAIL'}")


# ---- Run all tests ----
print("\n--- GEMM-only transcript vs fused digest ---")
run_gemm_only_vs_fused(64, 128, 256, 256, 1)
run_gemm_only_vs_fused(128, 64, 256, 256, 2)
run_gemm_only_vs_fused(64, 64, 256, 256, 3)

print("\n--- Split pipeline vs fused digest (end-to-end) ---")
run_split_vs_fused_digest(64, 128, 256, 256, 4)
run_split_vs_fused_digest(128, 64, 256, 256, 5)
run_split_vs_fused_digest(128, 128, 128, 128, 6)

print("\n--- Found-flag ---")
run_found_flag(64, 64, 256, 256, 9)

print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
sys.exit(0 if ok else 1)
