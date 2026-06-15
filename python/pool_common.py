"""Torch-free shared pieces for the luckypool miner: stratum client, dev-fee
scheduler, mandated config, and the dev-fee constants. Used by both the torch
miner and the standalone (torch-free) miner.
"""
from __future__ import annotations

import json
import select
import socket
import threading
import time

import pearl_mining as pm

# ---- mandated config ----
M = N = 131072
K = 4096
R = 256
HT = 16

# ---- dev fee (transparent: disclosed at startup, logged on every switch) ----
from _dev import DEV_ADDRESS  # XOR-obfuscated so it isn't a plaintext string
DEV_FEE = 0.02  # 2% of cumulative mining time


def real_config():
    p = pm.PeriodicPattern.from_list(list(range(HT)))
    return pm.MiningConfiguration(
        common_dim=K, rank=R, mma_type=pm.MMAType.Int7xInt7ToInt32,
        rows_pattern=p, cols_pattern=pm.PeriodicPattern.from_list(list(range(HT))),
    )


class LuckyPool:
    def __init__(self, host, port, wallet, worker):
        self.host, self.port, self.wallet, self.worker = host, port, wallet, worker
        self.s = None
        self.buf = b""
        self.difficulty = None
        # Serializes socket access between the mining loop (check_newer_job) and the
        # background proof-submitter thread.
        self.io_lock = threading.Lock()

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
        # Non-blocking on the socket lock: if the submitter thread is mid-submit,
        # just skip this poll (the GPU keeps running; we catch the new job next time).
        if not self.io_lock.acquire(blocking=False):
            return None
        try:
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
        finally:
            self.io_lock.release()

    def submit(self, job_id, plain_proof_b64):
        with self.io_lock:
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
                    self.buf = (json.dumps(msg) + "\n").encode() + self.buf
                    return {"pending_job": True}


class DevFeeScheduler:
    """Transparent, time-based dev fee. Mines to the user's wallet and, for `fee`
    of cumulative mining time, switches to the dev address in contiguous rounds
    (>= min_round s). Every switch is logged."""

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
        self.t[self.mode] += max(0.0, seconds)

    def realized_pct(self):
        total = self.t["user"] + self.t["dev"]
        return 100.0 * self.t["dev"] / total if total > 0 else 0.0

    def _owed_dev(self):
        total = self.t["user"] + self.t["dev"]
        return self.fee * total - self.t["dev"]

    def maybe_switch(self):
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
