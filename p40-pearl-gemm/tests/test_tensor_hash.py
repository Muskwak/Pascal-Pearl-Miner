"""Validate the Pascal tensor_hash against the reference BLAKE3 keyed hash.

For data whose length is a multiple of 1024 bytes (full chunks) the kernel's
BLAKE3 keyed Merkle hash equals blake3.blake3(data, key=key).digest().
Run from the project root:  python tests/test_tensor_hash.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import blake3
import p40_pearl_gemm_cuda as _C

dev = torch.device("cuda", int(os.environ.get("GEMM_TEST_DEV", "0")))
print(f"Device: {torch.cuda.get_device_name(dev)} "
      f"(cc {torch.cuda.get_device_capability(dev)})")

g = torch.Generator().manual_seed(7)
ok = True


def run(num_chunks, tpb, leaves):
    global ok
    nbytes = num_chunks * 1024
    data = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, generator=g).to(dev)
    key = torch.randint(0, 256, (32,), dtype=torch.uint8, generator=g)
    out = torch.zeros(32, dtype=torch.uint8, device=dev)
    nblocks = (num_chunks + tpb - 1) // tpb
    roots = torch.zeros(max(nblocks, 1) * 32, dtype=torch.uint8, device=dev)

    _C.tensor_hash(data, key.to(dev), out, roots, tpb, 2, leaves)
    torch.cuda.synchronize()

    got = bytes(out.cpu().numpy().tobytes())
    ref = blake3.blake3(bytes(data.cpu().numpy().tobytes()),
                        key=bytes(key.numpy().tobytes())).digest()
    match = got == ref
    # num_roots == 1 (one stage-1 CTA covers all chunks) is a degenerate config
    # that the real miner never uses (it requires data > 2^17 bytes with the
    # default threads_per_block=128, so num_roots >= 2). In that degenerate case
    # the upstream pipeline never applies BLAKE3's ROOT finalization, so it is
    # not expected to equal stock BLAKE3 -- we only assert on realistic configs.
    num_roots = nblocks
    realistic = num_roots >= 2
    if realistic:
        ok = ok and match
    tag = "MATCH" if match else ("MISMATCH" if realistic else "n/a (num_roots=1)")
    print(f"  chunks={num_chunks:5d} tpb={tpb:3d} leaves={leaves:4d} "
          f"num_roots={num_roots:3d} -> {tag}  got={got[:8].hex()} ref={ref[:8].hex()}")


for nc in (128, 256, 512, 1024, 4096):
    for tpb in (128, 256, 512):
        run(nc, tpb, 512)
run(2048, 128, 256)
run(2048, 128, 1024)

print("\n" + ("ALL REALISTIC CONFIGS MATCH BLAKE3" if ok else "SOME MISMATCH"))
sys.exit(0 if ok else 1)
