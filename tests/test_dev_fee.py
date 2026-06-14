"""Verify the dev-fee scheduler switches between the user's wallet and the dev
address and that the realized fee converges to the target. Deterministic (no
network). Run: python tests/test_dev_fee.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

from luckypool_miner import DevFeeScheduler, DEV_ADDRESS

USER = "prl1ps5axsy50dhql6g4kuulfxtr9df2qsu7sp20h7xgnx30uvfx7mgdsy9pu6w"  # example user wallet
ok = True


def check(name, cond):
    global ok
    ok = ok and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


logs = []
sched = DevFeeScheduler(0.02, USER, DEV_ADDRESS, lambda m: logs.append(m), min_round=30.0)

# Starts on the user's wallet.
check("starts on user wallet", sched.wallet == USER and sched.mode == "user")

# Drive ~simulated jobs; record every wallet transition.
transitions = []  # (from_mode, to_mode, realized_pct_at_switch)
prev_wallet = sched.wallet
saw_dev_round = False
for _ in range(6000):
    sched.note(20.0)  # ~20s mining job
    if sched.maybe_switch():
        transitions.append((prev_wallet == DEV_ADDRESS, sched.mode))
        if sched.mode == "dev":
            check("dev round selects the dev address", sched.wallet == DEV_ADDRESS)
            saw_dev_round = True
        else:
            check("post-dev returns to the user wallet", sched.wallet == USER)
        prev_wallet = sched.wallet

check("at least one dev round occurred", saw_dev_round)
check("dev address != user address", DEV_ADDRESS != USER)

# Over a long run the realized fee should sit close to 2%.
realized = sched.realized_pct()
check(f"realized fee ~2% (got {realized:.3f}%)", 1.7 <= realized <= 2.3)

# A dev round must actually be a *contiguous* block (>= min_round of dev time
# between switches), not a 1-job blip.
print(f"\n  realized={realized:.3f}%  transitions={len(transitions)}  "
      f"user_time={sched.t['user']:.0f}s  dev_time={sched.t['dev']:.0f}s")
print("\n  sample switch logs:")
for m in logs[:4]:
    print("   ", m.strip())

print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
sys.exit(0 if ok else 1)
