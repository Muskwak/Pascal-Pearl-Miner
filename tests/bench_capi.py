"""Steady-state throughput of the torch-free region loop (transcript reused, Bt
cached, hard target so no early-out): the rate the live miner converges to once a
job is past its first sweep. Run: python tests/bench_capi.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
import cuda_capi as cc

RS = 4096
K = 256 * 16  # 4096
R = 256
tiles = (RS // 16) ** 2
TH_PER_REGION = tiles * (1 << 20) / 1e12

dApEA = cc.DBuf(RS * K); dApEA.from_host(np.random.randint(-128, 127, (RS, K), np.int8))
dBpEB = cc.DBuf(RS * K); dBpEB.from_host(np.random.randint(-128, 127, (RS, K), np.int8))
dkey = cc.DBuf(32); dkey.memset(0)
dtgt = cc.DBuf(32); dtgt.memset(0)        # hardest target -> never found, full work
dtb = cc.DBuf(tiles * 16 * 4)
dfound = cc.DBuf(4); dcoord = cc.DBuf(8)
found = np.empty(1, np.int32)

VAR = 1
for warm in range(3):
    dfound.memset(0)
    cc.pearl_pow_split(dApEA, dBpEB, RS, RS, K, R, dkey, dtgt, dtb, None, dfound, dcoord, VAR)
cc.sync()

N = 40
t0 = time.time()
for _ in range(N):
    dfound.memset(0)
    cc.pearl_pow_split(dApEA, dBpEB, RS, RS, K, R, dkey, dtgt, dtb, None, dfound, dcoord, VAR)
    cc.sync()
    dfound.to_host(found)   # the per-region hit check the miner does
dt = (time.time() - t0) / N
rps = 1.0 / dt
print(f"  steady region loop: {dt*1e3:.2f} ms/region  {rps:.1f} regions/s  "
      f"{rps * TH_PER_REGION:.2f} TH/s")
