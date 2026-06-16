# Dev log — Ada (sm_89) tensor-core kernel optimization

Goal: close the gap on the RTX 4050 (Ada, AD107, 20 SM) between our Ampere/Ada
tensor-core GEMM and the reference "ipminer" — **8 TH/s → ~25 TH/s**, staying
**bit-exact** with the DP4A transcript.

1 tile = 16×16×4096 = 2^20 "hashes"; **1 Mtile/s ≈ 1.05 TH/s**. One mining region
= 4096×4096×4096 (R=256) = 65536 tiles. 25 TH/s ≈ 23.8 Mtile/s ≈ 2.75 ms/region.

## Test rig & loop
- An RTX 4050 (Ada, sm_89) test box reached over SSH. Windows, CUDA 12.8, **no Nsight
  Compute** installed (so profiling is empirical, via kernel timings).
- Cross-compile here (CUDA 12.8 + VS2022, `-arch=sm_89 -cudart static -Xcompiler /MT`),
  `scp` the ~0.6 MB exe, run on the box. One cycle ≈ 1 min.
- `tests/bench_ampere.cu` — includes the DP4A ref (`pearl_gemm_only_sm61.cu`) and the
  TC kernel; does a **bit-exact transcript check** (256×256×4096 R256 vs DP4A) and times
  the TC GEMM + DP4A at the real region config + a config sweep. `tests/iter.ps1` =
  build→scp→run.

## Baseline (fused kernel, 32×64 default)
```
TC   : 7.725 ms/region -> 8.90 TH/s  (17.8 INT8 TOPS, ~18% of peak)
DP4A : 10.31 ms/region -> 6.66 TH/s
TC speedup vs DP4A: 1.34x   <-- abysmal for tensor cores (should be 3-5x)
```
GEMM-only 8.9 ≈ the end-to-end 8 TH/s the user measured → **the GEMM is the
bottleneck** (BLAKE3 + noise + host overhead ≈ 10%). So optimize the GEMM.

## Iterations

| # | Change | Result | Verdict |
|---|--------|--------|---------|
| 1 | Vectorize smem→reg packing: 24 byte-loads+shifts → 6 `uint32` loads per MMA pair | 8.90 → 8.92 | no-op — nvcc already vectorized; loads aren't the bottleneck |
| 2 | Config sweep (block/warp/stage/minb) | **64×64 s4 b3 = 10.67** vs 32×64 = 8.9; 64×128/128×64/32×128 ≈ 9.3 | **64×64 is +20%**; dispatcher was picking the worse 32×64 |
| 3 | Reorder dispatcher to prefer 64×64 | TC 8.9 → **10.3** (shipped) | ✅ free +20% in the real miner |
| 4 | **R-block-staged kernel** (stage full R=256 k-slice, ~10× fewer syncs, dynamic smem) | bit-exact but **5.2 TH/s (2× slower)** | ❌ 64 KB smem → 1 block/SM → occupancy collapse |
| 5 | **Wide kernel** — each warp computes **NT** adjacent 16×16 tiles → **NT·2 independent accumulator chains** (small 32-k smem kept) | NT2=13.0, NT4=14.6, NT8=16.7, **NT16=17.6** | ✅ **the real fix** — ILP, not occupancy/sync |
| 6 | **wide1** — same but 1 `__syncthreads`/k-tile (sw-pipeline, prefetch far stage) | NT8=16.8, NT16=16.8 | ❌ no gain — syncs were never the bottleneck |
| 7 | Wire **wide NT16 (64×256)** into the dispatcher (n%256,m%64), NT8 fallback | TC path **8.9 → 16.2 TH/s (2.41× DP4A)** | ✅ shipped in `launch_pearl_ampere` |

### Key insight — it was ILP all along
`mma(acc,a,b,acc)` makes each accumulator a **serial dependency chain**. The fused kernel
had only **2 chains/warp** (accL,accR) → the tensor pipe stalls on that 2-deep latency.
Giving each warp **NT·2** independent chains (NT=16 → 32) fills the pipe. Confirmed:
- Wide NT16 ≈ 2× fused, and wins **even at low occupancy** (ILP > occupancy).
- R-block (more k-staging, same 2 chains) and wide1 (fewer syncs) gave **nothing** → the
  bottleneck was never loads, syncs, or occupancy.

### Benchmark gotcha
TH/s must be computed from tiles **actually covered**: BM and BN must divide the region
(4096). NT∈{2,4,8,16} give BN∈{32,64,128,256} (valid); NT=10/12 (BN=160/192) silently
cover fewer tiles and *inflate* TH/s. The bench now flags non-dividing configs with `*`.

## Current best
**Wide NT=16 (64×256, 2 stages) = ~16–17.6 TH/s GEMM (≈33 INT8 TOPS, ~35% of peak)** —
bit-exact, shipped via the dispatcher. **~2× the 8 TH/s baseline.**

## Now LSU-bound
At NT=16 each step issues ~68 shared loads (`LDS.32`) vs 32 MMAs (2:1) — the load/store
unit is the new ceiling. Next lever: **`ldmatrix.sync`** (1 instr loads a whole fragment,
~2–4× fewer LDS) + swizzled smem to avoid bank conflicts. Risk: ldmatrix layout must
reproduce the exact mma fragment (bit-exact). Also: re-measure end-to-end (BLAKE3 + noise
may now be a meaningful slice of the per-region time).

## Invariant
Every change must keep `bench_ampere.exe` reporting **BIT-EXACT PASS** (TC transcript ==
DP4A transcript). A correctness regression = revert.
