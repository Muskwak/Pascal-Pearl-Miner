# Torch-free miner — DONE: 60 MB standalone binary (was 4.5 GB)

> **STATUS: COMPLETE.** All 5 phases shipped and gated. The frozen torch-free
> binary is **60 MB** (75x smaller than the 4.5 GB torch bundle) and landed an
> accepted share on luckypool running self-contained (its own bundled Python,
> no torch/conda). Build: `packaging/build_windows.bat` (or `build_linux.sh`).
> Entry: `python/miner_capi.py`. Gates: `tests/test_capi_phase{1,2,3,4}.py`.



## Why the binary is 4.5 GB today
PyInstaller bundles CUDA PyTorch: `torch/lib` alone is **4.26 GB**. Everything else
(the CUDA extension, `pearl_mining`, blake3, numpy) is <40 MB. **Removing torch is
the entire prize.**

## Why this is tractable (grounded in the code)
All heavy GPU work is already in raw-pointer custom kernels — torch is only glue:

| Pipeline step | Today (torch) | Already torch-free? |
|---|---|---|
| Generate A,B operands | `torch.randint` | needs a small RNG kernel (or host fill) |
| Commit A,B (Merkle root) | `tensor_hash` kernel | ✅ raw-ptr kernel; torch only wraps it |
| Generate noise EAL/EAR/EBL/EBR | `noise_gen` kernel | ✅ raw-ptr kernel |
| **Noise-apply** `A_ns=A+EAL@EAR` | `_imatmul_i8` (**FP32 cublas**) | ✅ **`launch_noise_A/B` already exist** (int8, raw ptr) |
| Transpose `Bt=B[:,c].T` | `torch .t().contiguous()` | needs a small transpose kernel (or fold in) |
| Search (GEMM+BLAKE3) | `pearl_pow_split` kernel | ✅ raw-ptr kernel |
| Build proof on a hit | `pearl_mining` (Rust) | ✅ **already numpy/bytes**, not torch (`.cpu().numpy().tobytes()`) |
| Verify proof | `pearl_mining` | ✅ bytes-based |
| Stratum + dev fee | pure Python sockets/json | ✅ no torch |

So torch supplies: device alloc, H2D/D2H copy, dtype casts, slicing, transpose,
RNG, and the one FP32 matmul — **all replaceable with lightweight CUDA bindings +
the kernels we already have.** The FP32 matmul replacement (`noise_A/B`) is the
biggest single win and **already written**.

## Architecture decision
**Python + a lean CUDA backend**, not a full C++ rewrite (keeps the stratum, dev
fee, and `pearl_mining` proof code as-is). Two backend choices:

- **`cuda-python`** (NVIDIA's official `cuda.bindings`, ~5 MB): leanest bundle.
  Manual `cuMemAlloc`/`cuMemcpy`/`cuLaunchKernel`. **Primary choice.**
- **`cupy`** (installed): nicer ndarray ergonomics (slicing, `.T`, RNG for free),
  but its wheel bundles CUDA libs — **must measure**; may blow the <100 MB budget.

Plan around `cuda-python`; keep `cupy` as a fallback if RNG/transpose ergonomics
save enough time to justify the size.

## Components to build/change
1. **`bindings_raw.cpp`** — a torch-free module (or `extern "C"` DLL) exposing every
   kernel with device pointers passed as integers: `noise_gen`, `tensor_hash`,
   `noise_apply` (wrap `launch_noise_A/B`), `pearl_pow_split`, plus small helpers
   (`rng_fill`, `transpose_i8`). Links **cudart only** (0.57 MB), not torch.
2. **RNG fill kernel** — Philox/xorshift to fill A,B with random int8 (replaces
   `torch.randint`). Must be reproducible so the *same* bytes are available to the
   proof builder (or just D2H-copy A,B on a hit — they're only needed then).
3. **Transpose kernel** (`transpose_i8`) for the `Bt` column operand — or fold the
   transpose into the search-kernel staging to avoid an extra pass.
4. **Memory manager (Python)** — thin wrapper over `cuda-python` for alloc / copy /
   stream / synchronize; replaces `torch.empty/zeros/frombuffer/.to(dev)`.
5. **Proof path** — on a hit, D2H-copy A and B to host `bytes`/numpy → feed the
   existing `pearl_mining` `MerkleTree`/`PlainProof` (unchanged). ~1 GB copy, but
   only on a hit (rare).
6. **`miner_core.py`** — rewrite of the per-job loop using (1)+(4). The `LuckyPool`
   stratum class and `DevFeeScheduler` are reused verbatim.
7. **PyInstaller spec** — `--exclude torch`; bundle cuda-python + pearl_mining +
   numpy + blake3 + the small .pyd + `cudart64*.dll`. Target **~50–80 MB**.

## Phased delivery (each phase has a bit-exactness gate)
- **P1 — Torch-free backend** ✅ DONE. `csrc/capi/p40_capi.cu` → standalone
  `p40cuda.dll` (**655 KB**, links cudart only; `packaging/build_capi.{bat,sh}`),
  driven via stdlib `ctypes`. `error_check.hpp` made torch-free under `P40_NO_TORCH`.
  Gate PASSED: `tests/test_capi_phase1.py` reproduces the torch `pearl_pow_split`
  digests bit-for-bit (no torch in the compute path).
- **P2 — Noise-apply via `noise_A/B`.** Gate: `A_ns`/`Bt_ns` bit-exact vs the
  current `_imatmul_i8` path (and vs the reference noisy_gemm) on a fixed seed.
- **P3 — RNG + transpose kernels.** Gate: committed roots + a full no-hit sweep
  match the torch miner.
- **P4 — Miner loop + proof path.** Gate: **land an accepted share** on luckypool
  from the torch-free miner (the real end-to-end test), dev fee intact.
- **P5 — Package.** Gate: frozen binary runs on a clean box, size <100 MB.

## Risks
- **Noise-apply bit-exactness**: `noise_A/B` (int) must produce the *same* `A_ns`
  the verifier recomputes. The reference is integer arithmetic, so int kernels
  should be *more* correct than today's FP32-round path — but must be validated
  against `pearl_mining` verification (P2 gate) before trusting it.
- **`cuda-python` ergonomics**: manual memory/launch is more verbose than torch;
  budget time for arg marshalling and a clean wrapper.
- **RNG reproducibility vs D2H-on-hit**: simplest is to D2H-copy A,B only on a hit
  (no need to reproduce RNG); chosen unless it bloats hit latency.
- **cupy bundle size** if we fall back to it — measure before committing.

## Effort
Medium-large: ~1 new C++ binding file, 2 small kernels (RNG, transpose), 1 Python
backend wrapper, 1 miner-loop rewrite, 1 spec. Days, not weeks. Existing kernels
(noise_gen, noise_A/B, tensor_hash, pearl_pow_split) are reused unchanged.

## Outcome
Standalone miner binary **~50–80 MB** (vs 4.5 GB), no torch, no cublas, same
7.5 TH/s kernel, dev fee intact. Linux binary still built on Linux (same spec).
