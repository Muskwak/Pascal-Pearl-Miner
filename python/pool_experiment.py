"""Instrumented AlphaPool submission experiment.

Mines a CORRECT proof on the P40 (real pearl_mining config + Merkle roots, which
the earlier silently-dropped attempts got wrong) and submits it under a few
seed->header hypotheses, capturing all pool traffic after each submit.

Usage: python pool_experiment.py [host:port] [wallet]
"""
import json
import os
import socket
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import blake3
import torch
import pearl_mining as pm

import pearl_miner as miner

HOST_PORT = sys.argv[1] if len(sys.argv) > 1 else "us2.alphapool.tech:5566"
WALLET = sys.argv[2] if len(sys.argv) > 2 else \
    "prl1p6zqnlkklar74yfgmf2c4v6vfs6kme48lx9nnwxr7uyzpky88kf3s48ccxv"
WORKER = "p40"
HOST, PORT = HOST_PORT.rsplit(":", 1)
PORT = int(PORT)

dev = torch.device("cuda:0")
R, K, M, N = 128, 2048, 256, 256  # k >= 16*rank


def real_config():
    p = pm.PeriodicPattern.from_list(list(range(16)))
    return pm.MiningConfiguration(
        common_dim=K, rank=R, mma_type=pm.MMAType.Int7xInt7ToInt32,
        rows_pattern=p, cols_pattern=pm.PeriodicPattern.from_list(list(range(16))),
        reserved=pm.MiningConfiguration.RESERVED,
    )


def bound_from_difficulty(difficulty, cfg):
    raw = 1 << (256 - int(difficulty))
    h = miner.pearl_miner._v(cfg.rows_pattern.size) if hasattr(miner.pearl_miner, "_v") else cfg.rows_pattern.size()
    return raw  # adjustment applied below


def _sz(x):
    return x() if callable(x) else x


def diff_bound(difficulty, cfg):
    raw = 1 << (256 - int(difficulty))
    h = _sz(cfg.rows_pattern.size)
    w = _sz(cfg.cols_pattern.size)
    dp = _sz(cfg.rounded_common_dim)
    return min(raw * h * w * dp, (1 << 256) - 1)


def mine_proof(header_bytes, cfg, bound):
    """GPU-mine until a tile hits <= bound; return (proof_b64, plain_proof, A, B, row, col, key)."""
    key = blake3.blake3(header_bytes + cfg.to_bytes()).digest()
    for _ in range(4000):
        A = torch.randint(-64, 63, (M, K), dtype=torch.int8).to(dev)
        B = torch.randint(-64, 63, (K, N), dtype=torch.int8).to(dev)
        res = miner.mine_once(header_bytes, bound, A, B, cfg, rank=R)
        if res.found:
            proof = miner.build_proof(A, B, res.row, res.col, key, R)
            return proof.to_base64(), proof, header_bytes, res.row, res.col
    return None, None, header_bytes, -1, -1


def connect():
    s = socket.create_connection((HOST, PORT), timeout=30)
    s.settimeout(8)
    return s


def recv_lines(s, seconds):
    s.settimeout(seconds)
    out = []
    end = time.time() + seconds
    buf = b""
    while time.time() < end:
        try:
            d = s.recv(8192)
        except Exception:
            break
        if not d:
            break
        buf += d
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            if line.strip():
                out.append(line.decode(errors="replace"))
    return out


print(f"== AlphaPool experiment ==  host={HOST}:{PORT} wallet={WALLET[:12]}... device={torch.cuda.get_device_name(dev)}")
s = connect()
init = recv_lines(s, 6)
print(f"on-connect ({len(init)} msgs):")
for m in init:
    print("  <", m[:200])

challenge = None
for m in init:
    try:
        j = json.loads(m)
    except Exception:
        continue
    if j.get("method") == "pearl.challenge":
        challenge = j["params"]
if challenge is None:
    # poke it
    s.sendall((json.dumps({"id": 1, "method": "mining.authorize", "params": [f"{WALLET}.{WORKER}", "x"]}) + "\n").encode())
    for m in recv_lines(s, 5):
        print("  < (after authorize)", m[:200])
        try:
            j = json.loads(m)
            if j.get("method") == "pearl.challenge":
                challenge = j["params"]
        except Exception:
            pass

if challenge is None:
    print("No pearl.challenge received; cannot continue.")
    s.close(); sys.exit(1)

seed = bytes.fromhex(challenge["seed"])
difficulty = challenge["difficulty"]
print(f"challenge: seed={seed.hex()[:24]}.. difficulty={difficulty}")
cfg = real_config()
bound = diff_bound(difficulty, cfg)
print(f"config.to_bytes len={len(cfg.to_bytes())}  bound=2^{bound.bit_length()-1}")

# Seed->header hypotheses
hdr76 = lambda prev, merk, nbits: pm.IncompleteBlockHeader(
    version=0, prev_block=prev, merkle_root=merk, timestamp=0, nbits=nbits).to_bytes()
hypotheses = [
    ("H1 raw-seed-as-header", seed),
    ("H2 76B prev=seed nbits=0", hdr76(seed, b"\x00" * 32, 0)),
    ("H3 76B merkle=seed nbits=0", hdr76(b"\x00" * 32, seed, 0)),
    ("H4 seed padded to 76", (seed + b"\x00" * 44)[:76]),
]

for name, header_bytes in hypotheses:
    print(f"\n--- {name}: mining correct proof on P40 ---")
    t0 = time.time()
    proof_b64, proof, hb, row, col = mine_proof(header_bytes, cfg, bound)
    if proof_b64 is None:
        print("  no GPU hit (bound too hard?)"); continue
    print(f"  HIT tile ({row},{col}) in {time.time()-t0:.1f}s, proof b64 len={len(proof_b64)}")
    # local structural verify if header is 76 bytes
    if len(header_bytes) == 76:
        try:
            v, msg = miner.verify_proof_local(header_bytes, proof)
            print(f"  local verify_plain_proof: valid={v} ({msg})")
        except Exception as e:
            print(f"  local verify error: {e}")
    # submit
    params = [f"{WALLET}.{WORKER}", seed.hex(), proof_b64]
    s.sendall((json.dumps({"id": 100, "method": "mining.submit", "params": params}) + "\n").encode())
    print(f"  submitted ({len(proof_b64)}B proof); waiting for response...")
    resp = recv_lines(s, 8)
    if resp:
        for r in resp:
            print("  POOL >", r[:300])
    else:
        print("  (silent — no response)")

s.close()
print("\nexperiment done")
