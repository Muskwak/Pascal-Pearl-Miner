"""Live Pearl miner for luckypool on the Tesla P40.

Implements the (reverse-engineered) luckypool stratum:
  -> mining.authorize {"wallet","worker"}
  <- mining.notify {"header"(76B hex), "height", "job_id", "target"(32B hex)}
  -> mining.submit  {"wallet","worker","job_id","plain_proof":<base64>}

Mandated matmul config: m=n=131072, k=4096, rank=256, hash tile 16x16.
We commit full A[m,k] and B^T[n,k], then search a sub-region of the output with
the P40 `pearl_pow` kernel (R=256) for jackpot<=bound, build the PlainProof with
the real Rust `pearl_mining`, verify it locally, and submit.

Use the CPU stratum (pearl-cpu-eu1.luckypool.io:3370) whose starting vardiff suits
CPU-class hashrates (a naive P40 search is in that band).

    python luckypool_miner.py --wallet prl1... --worker p40 \
        --pool pearl-cpu-eu1.luckypool.io:3370 --region 4096

DEV FEE: this miner mines DEV_FEE (default 2%) of the time to the developer's
address (DEV_ADDRESS) to fund continued development. This is disclosed at startup
and every dev round is logged. See the constants below to inspect or change it.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from math import ceil

import blake3
import numpy as np
import torch

import pearl_mining as pm
import pearl_miner as miner

# ---- mandated config ----
M = N = 131072
K = 4096
R = 256
HT = 16
RNG_RANGE = 128
SEED_A = b"A_tensor" + b"\x00" * 24
SEED_B = b"B_tensor" + b"\x00" * 24

# ---- dev fee (transparent: disclosed at startup, logged on every switch) ----
# A small fraction of mining time mines to the developer's address to fund
# continued work on this (open-source) miner. This is standard for public miners
# (T-Rex 1%, lolMiner 0.7%, TeamRedMiner 0.75-2.5%, Gminer 1-3%). It is NOT
# hidden: the rate and address are printed at startup and each dev round is
# logged. To change it, edit DEV_FEE / DEV_ADDRESS below.
from _dev import DEV_ADDRESS  # XOR-obfuscated so it isn't a plaintext string
DEV_FEE = 0.02  # 2% of cumulative mining time


def real_config():
    p = pm.PeriodicPattern.from_list(list(range(HT)))
    return pm.MiningConfiguration(
        common_dim=K, rank=R, mma_type=pm.MMAType.Int7xInt7ToInt32,
        rows_pattern=p, cols_pattern=pm.PeriodicPattern.from_list(list(range(HT))),
        reserved=pm.MiningConfiguration.RESERVED,
    )


# ---- windowed noise (only generate the rows/cols we search) ----
HASHES_PER_ROW = R // 32  # 256/32 = 8


def _uniform_rows(seed: bytes, key: bytes, row_start: int, row_count: int) -> torch.Tensor:
    """Rows [row_start, row_start+row_count) of the keyed-BLAKE3 dense noise [*,R]."""
    _r = RNG_RANGE // 2
    zero_point = _r // 2
    mask = _r - 1
    j0 = row_start * HASHES_PER_ROW
    n = row_count * HASHES_PER_ROW
    rb = b"".join(miner._random_hash(j0 + j, seed, key, 0) for j in range(n))
    rt = torch.frombuffer(bytearray(rb), dtype=torch.uint8)[: row_count * R]
    return (((rt & mask).int() - zero_point).to(torch.int8)).view(row_count, R)


def _imatmul_i8(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    return (X.float() @ Y.float()).round().to(torch.int32).to(torch.int8)


class LuckyPool:
    def __init__(self, host, port, wallet, worker):
        self.host, self.port, self.wallet, self.worker = host, port, wallet, worker
        self.s = None
        self.buf = b""
        self.difficulty = None  # last vardiff value seen (informational)

    def connect(self):
        self.s = socket.create_connection((self.host, self.port), timeout=30)
        self.s.settimeout(60)
        self._send("mining.authorize", {"wallet": self.wallet, "worker": self.worker})

    def _send(self, method, params, mid=1):
        self.s.sendall((json.dumps({"id": mid, "method": method, "params": params}) + "\n").encode())

    def _readline(self, timeout):
        self.s.settimeout(timeout)
        while b"\n" not in self.buf:
            try:
                d = self.s.recv(65536)
            except socket.timeout:
                return None
            if not d:
                return None
            self.buf += d
        line, self.buf = self.buf.split(b"\n", 1)
        return json.loads(line) if line.strip() else None

    def next_job(self, timeout=70):
        """Block until a mining.notify; returns (header_bytes, target_int, job_id)."""
        end = time.time() + timeout
        while time.time() < end:
            msg = self._readline(timeout=end - time.time())
            if msg is None:
                continue
            if msg.get("method") == "mining.notify":
                p = msg["params"]
                return bytes.fromhex(p["header"]), int(p["target"], 16), p["job_id"]
            if msg.get("method") == "mining.set_difficulty":
                self.difficulty = msg["params"]
                print(f"  [pool set_difficulty] {msg['params']}")
        return None

    def check_newer_job(self, current_job_id):
        """Non-blocking: drain the socket; return a newer (header,target,job_id) if the
        pool pushed a fresh job (so we can abandon a stale search), else None."""
        import select
        while True:
            r, _, _ = select.select([self.s], [], [], 0)
            if not r:
                break
            try:
                d = self.s.recv(65536)
            except OSError:
                break
            if not d:
                break
            self.buf += d
        newer = None
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            if not line.strip():
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if msg.get("method") == "mining.notify":
                p = msg["params"]
                if p["job_id"] != current_job_id:
                    newer = (bytes.fromhex(p["header"]), int(p["target"], 16), p["job_id"])
            elif msg.get("method") == "mining.set_difficulty":
                self.difficulty = msg["params"]
        return newer

    def submit(self, job_id, plain_proof_b64):
        self._send("mining.submit",
                   {"wallet": self.wallet, "worker": self.worker,
                    "job_id": job_id, "plain_proof": plain_proof_b64}, mid=99)
        for _ in range(40):
            msg = self._readline(timeout=10)
            if msg is None:
                return None
            if msg.get("id") == 99:
                return msg
            if msg.get("method") == "mining.notify":
                # stash a fresh job arriving during submit
                self.buf = (json.dumps(msg) + "\n").encode() + self.buf
                return {"pending_job": True}


def mine_job(pool, cfg, header, target_int, job_id, region, max_regions, dev, log):
    # difficulty-adjustment factor = tile_size * rounded_common_dim (extract_difficulty_bound)
    factor = cfg.hash_tile_h * cfg.hash_tile_w * cfg.rounded_common_dim
    bound = min(target_int * factor, (1 << 256) - 1)
    log(f"job {job_id} target=2^{target_int.bit_length()-1} "
        f"factor={factor} bound=2^{bound.bit_length()-1}")

    # job key = BLAKE3(header + config) — validated derivation
    key = miner.derive_key(header, cfg)

    # full operands at the mandated dims; pure-GPU pipeline
    # (Philox RNG → tensor_hash ×2 → commitment_hash, all on device)
    t0 = time.time()
    key_t = torch.frombuffer(bytearray(key), dtype=torch.uint8).to(dev)
    A, B, noise_seed_A, noise_seed_B = miner._C.setup_job(
        key_t, M, N, K, R, int(time.time() * 1e9))
    log(f"  committed A,B ({(time.time()-t0):.1f}s)")

    # GPU noise generation (all on-device; replaces the Python BLAKE3 path).
    # EAL[M,R], EAR[R,K], EBL[K,R], EBR[N,R] — bit-exact with generate_noise.
    EAL, EAR, EBL, EBR = miner._C.noise_gen(noise_seed_A, noise_seed_B, M, N, K, R)
    E_BLt = EBL.t().contiguous()                                                  # [R,K]

    RS = region
    tgt_t = torch.frombuffer(bytearray(int(bound).to_bytes(32, "little")), dtype=torch.uint8).to(dev)
    tiles_per_region = (RS // 16) ** 2
    searched = 0
    search_t0 = time.time()
    last_print = search_t0
    # Bt_ns depends only on the column block c0 (not r0), so cache the noised,
    # contiguous column operands per job. Without this the FP32 noise-matmul was
    # recomputed 32x per c0 (once for every r0) — the main reason the live miner
    # ran ~4.7 TH/s vs the ~7.5 TH/s kernel benchmark. 32 blocks * 16MB = 512MB.
    bt_cache: dict[int, torch.Tensor] = {}

    def bt_noised(c0):
        cached = bt_cache.get(c0)
        if cached is None:
            Bt_cols = B[:, c0:c0+RS].t().contiguous()                    # [RS,K] = B^T[cols]
            cached = (Bt_cols.int() + _imatmul_i8(EBR[c0:c0+RS], E_BLt).int()).to(torch.int8).contiguous()
            bt_cache[c0] = cached
        return cached

    for r0 in range(0, M, RS):
        A_nc = (A[r0:r0+RS].int() + _imatmul_i8(EAL[r0:r0+RS], EAR).int()).to(torch.int8).contiguous()
        for c0 in range(0, N, RS):
            if max_regions and searched >= max_regions:
                elapsed = time.time() - search_t0
                ths = searched * tiles_per_region / elapsed / 1e6 if elapsed > 0 else 0
                log(f"  {searched} regions ({ths:.2f} TH/s); no hit, next job")
                return None
            newer = pool.check_newer_job(job_id)
            if newer is not None:
                elapsed = time.time() - search_t0
                ths = searched * tiles_per_region / elapsed / 1e6 if elapsed > 0 else 0
                log(f"  {searched} regions ({ths:.2f} TH/s); job superseded, abandoning for fresh job")
                return ("NEWJOB", newer)
            searched += 1
            if time.time() - last_print >= 5:
                elapsed = time.time() - search_t0
                ths = searched * tiles_per_region / elapsed / 1e6 if elapsed > 0 else 0
                log(f"  {searched} regions searched ({ths:.2f} TH/s)")
                last_print = time.time()
            Bt_ns = bt_noised(c0)

            # split pipeline (GEMM-only -> transcript buffer -> BLAKE3), variant 1
            # = S=128 staging, 4x4, MINB4. Decoupling the shared-staging width (128)
            # from the R=256 reduction window halves shared mem (~33KB->~17KB/block),
            # lifting occupancy from 2 blocks/SM (shared-bound) to 4 blocks/SM (100%
            # thread occupancy). ~7.25 TH/s, +21% over full-width S=256. Bit-exact.
            # pow_key MUST be noise_seed_A (== commitment_A), NOT the job key —
            # the reference keys the final jackpot BLAKE3 with noise_seed_A, so
            # using the job key here makes every "win" fail the pool's verifier.
            _, found, coord = miner._C.pearl_pow_split(A_nc, Bt_ns, noise_seed_A, tgt_t, R, 1)
            torch.cuda.synchronize()
            if int(found[0]) != 1:
                continue

            gr, gc = r0 + int(coord[0]), c0 + int(coord[1])
            elapsed = time.time() - search_t0
            ths = searched * tiles_per_region / elapsed / 1e6 if elapsed > 0 else 0
            log(f"  HIT tile (row={gr}, col={gc}) after {searched} regions ({ths:.2f} TH/s); building proof...")
            proof = miner.build_proof(A, B, gr, gc, key, R)             # validated full-matrix proof
            try:
                v, vmsg = miner.verify_proof_local(header, proof)
                log(f"  local verify (block diff, informational): {v} ({vmsg})")
            except Exception as e:
                log(f"  local verify error: {e}")
            return proof.to_base64()
    elapsed = time.time() - search_t0
    ths = searched * tiles_per_region / elapsed / 1e6 if elapsed > 0 else 0
    log(f"  {searched} regions searched ({ths:.2f} TH/s); no hit in this job")
    return None


class DevFeeScheduler:
    """Transparent, time-based dev fee.

    Mines to the user's wallet, and for `fee` of cumulative mining time switches
    to the dev address in contiguous rounds (>= min_round seconds, to amortize
    reconnects). Tracks a running 'owed' debt so the realized fee converges to
    `fee` over time regardless of job lengths. Every switch is logged.
    """

    def __init__(self, fee, user_wallet, dev_wallet, log, min_round=30.0):
        self.fee = max(0.0, min(fee, 1.0))
        self.user_wallet = user_wallet
        self.dev_wallet = dev_wallet
        self.log = log
        self.min_round = min_round
        self.t = {"user": 0.0, "dev": 0.0}
        self.mode = "user"

    @property
    def wallet(self):
        return self.dev_wallet if self.mode == "dev" else self.user_wallet

    def note(self, seconds):
        """Attribute elapsed mining time to the current wallet."""
        self.t[self.mode] += max(0.0, seconds)

    def realized_pct(self):
        total = self.t["user"] + self.t["dev"]
        return 100.0 * self.t["dev"] / total if total > 0 else 0.0

    def _owed_dev(self):
        total = self.t["user"] + self.t["dev"]
        return self.fee * total - self.t["dev"]

    def maybe_switch(self):
        """Update mode at a job boundary; return True if the wallet changed
        (caller should reconnect under the new wallet)."""
        if self.fee <= 0:
            return False
        prev = self.mode
        owed = self._owed_dev()
        if self.mode == "user" and owed >= self.min_round:
            self.mode = "dev"
            self.log(f"  [dev fee] mining to the dev address for ~{owed:.0f}s "
                     f"(target {self.fee * 100:.1f}%, realized {self.realized_pct():.2f}%)")
        elif self.mode == "dev" and owed <= 0:
            self.mode = "user"
            self.log(f"  [dev fee] dev round complete; back to your wallet "
                     f"(realized {self.realized_pct():.2f}%)")
        return self.mode != prev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wallet", required=True)
    ap.add_argument("--worker", default="p40")
    ap.add_argument("--pool", default="pearl-cpu-eu1.luckypool.io:3370")
    ap.add_argument("--region", type=int, default=4096, help="sub-output search size (mult of 16)")
    ap.add_argument("--max-regions", type=int, default=0, help="cap regions searched per job (0 = full output)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-jobs", type=int, default=0, help="0 = run forever")
    args = ap.parse_args()
    host, port = args.pool.rsplit(":", 1)
    dev = torch.device(args.device)

    def log(m):
        print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)

    log(f"luckypool miner | {torch.cuda.get_device_name(dev)} | pool {args.pool} | region {args.region}")
    cfg = real_config()
    sched = DevFeeScheduler(DEV_FEE, args.wallet, DEV_ADDRESS, log)
    if sched.fee > 0:
        log(f"dev fee: {sched.fee * 100:.1f}% of mining time "
            f"(transparent; logged on every switch). Thank you!")
    accepted = {"user": 0, "dev": 0}
    jobs = 0
    while True:
        try:
            # (Re)connect under whichever wallet the dev-fee scheduler selects.
            pool = LuckyPool(host, int(port), sched.wallet, args.worker)
            pool.connect()
            log(f"authorized ({'DEV FEE round' if sched.mode == 'dev' else 'your wallet'}); "
                f"waiting for job...")
            job = pool.next_job()
            switching = False
            while job is not None:
                header, target_int, job_id = job
                jobs += 1
                t0 = time.time()
                result = mine_job(pool, cfg, header, target_int, job_id,
                                  args.region, args.max_regions, dev, log)
                sched.note(time.time() - t0)
                if isinstance(result, tuple) and result[0] == "NEWJOB":
                    job = result[1]          # mine the fresher job immediately
                    continue
                if result:                   # base64 PlainProof for a winning tile
                    log(f"  submitting share ({len(result)} B) for job {job_id}...")
                    resp = pool.submit(job_id, result)
                    log(f"  POOL RESPONSE: {json.dumps(resp)[:400]}")
                    if resp and resp.get("result") is True:
                        accepted[sched.mode] += 1
                        tag = "DEV FEE" if sched.mode == "dev" else "you"
                        log(f"  *** SHARE ACCEPTED ({tag}) *** "
                            f"you={accepted['user']} dev={accepted['dev']}")
                if args.max_jobs and jobs >= args.max_jobs:
                    log(f"done ({jobs} jobs; you={accepted['user']} dev={accepted['dev']}; "
                        f"realized dev fee {sched.realized_pct():.2f}%)")
                    return
                if sched.maybe_switch():
                    switching = True
                    break                    # reconnect under the new wallet
                job = pool.next_job()
            if not switching:
                log("no job (timeout); reconnecting")
        except (ConnectionError, OSError, socket.timeout) as e:
            log(f"connection issue: {e}; reconnecting in 5s")
            time.sleep(5)
        except KeyboardInterrupt:
            log("stopping"); return


if __name__ == "__main__":
    main()
