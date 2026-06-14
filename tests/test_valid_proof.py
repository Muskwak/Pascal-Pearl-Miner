"""Prove the P40 produces CONSENSUS-VALID Pearl proofs.

Uses the real Rust `pearl_mining` for serialization + `verify_plain_proof` as the
oracle. Part 1 sanity-checks the toolchain with the reference `mine`. Part 2 finds
a winning tile with the Pascal `pearl_pow` GPU kernel, builds the PlainProof with
`pearl_mining`, and verifies it.
"""
import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "python"))

import blake3
import numpy as np
import torch
import pearl_mining as pmrust

import p40_pearl_gemm_cuda as _C
import pearl_miner as pm

dev = torch.device("cuda", int(os.environ.get("GEMM_TEST_DEV", "0")))
print(f"Device: {torch.cuda.get_device_name(dev)}")

R = 128
HT = 16
M, N, K = 256, 256, 2048  # k >= 16*rank (sanity-check constraint)
NBITS = 0x1F000080  # bound ~2^250: GPU searches, non-degenerate


def make_header(nbits):
    return pmrust.IncompleteBlockHeader(
        version=0, prev_block=b"\x00" * 32,
        merkle_root=b"0123456789abcdef" * 2, timestamp=0x66666666, nbits=nbits,
    )


def make_config(k, rank):
    p = pmrust.PeriodicPattern.from_list(list(range(HT)))  # dense 16-wide tile
    return pmrust.MiningConfiguration(
        common_dim=k, rank=rank, mma_type=pmrust.MMAType.Int7xInt7ToInt32,
        rows_pattern=p, cols_pattern=pmrust.PeriodicPattern.from_list(list(range(HT))),
        reserved=pmrust.MiningConfiguration.RESERVED,
    )


def nbits_to_target(nbits):
    exp = nbits >> 24
    mant = nbits & 0xFFFFFF
    return mant << (8 * (exp - 3)) if exp > 3 else mant >> (8 * (3 - exp))


def _v(x):
    return x() if callable(x) else x


def difficulty_bound(nbits, config):
    t = nbits_to_target(nbits)
    h = _v(config.rows_pattern.size)
    w = _v(config.cols_pattern.size)
    dp = _v(config.rounded_common_dim)      # dot_product_length = (k//rank)*rank
    factor = h * w * dp
    bound = t * factor
    return min(bound, (1 << 256) - 1)


ok = True

# ---- Part 1: reference mine + verify (toolchain ground truth) ----
hdr = make_header(NBITS)
cfg = make_config(K, R)
print(f"config.to_bytes() len = {len(cfg.to_bytes())}  hash_tile = {_v(cfg.hash_tile_h)}x{_v(cfg.hash_tile_w)}")
proof_ref = pmrust.mine(M, N, K, hdr, cfg, signal_range=None, wrong_jackpot_hash=False)
v, msg = pmrust.verify_plain_proof(hdr, proof_ref)
print(f"[Part1] reference mine -> verify_plain_proof: valid={v} ({msg})  b64_len={len(proof_ref.to_base64())}")
ok = ok and v

# ---- Part 2: GPU finds the tile, we build + verify the proof ----
key = blake3.blake3(hdr.to_bytes() + cfg.to_bytes()).digest()


def merkle_root_and_tree(mat_int8):
    flat = mat_int8.to(torch.uint8).cpu().numpy().tobytes()
    padded = pmrust.pad_to_chunk_boundary(flat)
    tree = pmrust.MerkleTree(padded, key)
    return tree


g = torch.Generator().manual_seed(7)
A = torch.randint(-64, 63, (M, K), dtype=torch.int8, generator=g)
B = torch.randint(-64, 63, (K, N), dtype=torch.int8, generator=g)
Bt = B.t().contiguous()

tree_A = merkle_root_and_tree(A)
tree_B = merkle_root_and_tree(Bt)
root_A = bytes(tree_A.root)
root_B = bytes(tree_B.root)
b_seed = blake3.blake3(key + root_B).digest()
a_seed = blake3.blake3(b_seed + root_A).digest()   # = noise_seed_A = pow_key

E_AL, E_AR, E_BL, E_BR = pm.generate_noise(a_seed, b_seed, M, K, N, R)
A_n, B_n = pm.noised_operands(A.to(dev), B.to(dev),
                              E_AL.to(dev), E_AR.to(dev), E_BL.to(dev), E_BR.to(dev))

bound = difficulty_bound(NBITS, cfg)
print(f"[Part2] difficulty bound = 2^{bound.bit_length()-1}.. ; searching on P40")

found_proof = None
attempts = 0
while found_proof is None and attempts < 64:
    attempts += 1
    res = pm.run_pow(A_n, B_n, a_seed, bound, R)
    if res.found:
        rows = list(range(res.row, res.row + HT))
        cols = list(range(res.col, res.col + HT))
        li_A = pmrust.MerkleTree.compute_leaf_indices_from_rows(rows, (M, K))
        li_B = pmrust.MerkleTree.compute_leaf_indices_from_rows(cols, (N, K))
        mp_A = pmrust.MatrixMerkleProof(tree_A.get_multileaf_proof(li_A), rows)
        mp_B = pmrust.MatrixMerkleProof(tree_B.get_multileaf_proof(li_B), cols)
        found_proof = pmrust.PlainProof(M, N, K, R, mp_A, mp_B)
        print(f"[Part2] GPU hit at tile ({res.row},{res.col}) after {attempts} launch(es)")
        break
    # rotate to a fresh nonce -> fresh noise -> new tiles
    hdr = make_header(NBITS)  # same header; re-randomize A,B instead
    A = torch.randint(-64, 63, (M, K), dtype=torch.int8, generator=g)
    B = torch.randint(-64, 63, (K, N), dtype=torch.int8, generator=g)
    Bt = B.t().contiguous()
    tree_A = merkle_root_and_tree(A)
    tree_B = merkle_root_and_tree(Bt)
    root_A, root_B = bytes(tree_A.root), bytes(tree_B.root)
    b_seed = blake3.blake3(key + root_B).digest()
    a_seed = blake3.blake3(b_seed + root_A).digest()
    E_AL, E_AR, E_BL, E_BR = pm.generate_noise(a_seed, b_seed, M, K, N, R)
    A_n, B_n = pm.noised_operands(A.to(dev), B.to(dev),
                                  E_AL.to(dev), E_AR.to(dev), E_BL.to(dev), E_BR.to(dev))

if found_proof is None:
    print("[Part2] no GPU hit in 64 launches (try easier NBITS)")
    ok = False
else:
    v2, msg2 = pmrust.verify_plain_proof(hdr, found_proof)
    print(f"[Part2] GPU proof -> verify_plain_proof: valid={v2} ({msg2})  b64_len={len(found_proof.to_base64())}")
    ok = ok and v2

print("\n" + ("ALL PASS — P40 produces consensus-valid Pearl proofs" if ok else "FAILED"))
sys.exit(0 if ok else 1)
