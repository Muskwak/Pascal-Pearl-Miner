# p40-alpha-miner

**Pascal P40-optimized Pearl (PRL) miner** — 7.5 TH/s on a single P40.

Mines Pearl using INT8 DP4A instructions on Pascal GPUs (Tesla P40, GTX 1070, GTX 1080, etc.) via luckypool.io stratum.

## Performance

| GPU | Hashrate | Config | Bottleneck |
|---|---|---|---|
| Tesla P40 | **7.5 TH/s** | v1 (S=128, 4×4, MINB4) | 32% of 23.5 TOPS dp4a peak |
| GTX 1070 | ~4 TH/s (est.) | same kernel | 15 SM vs P40's 30 |

## Quick start

```bash
# Install deps
pip install py-pearl-mining torch blake3 numpy

# Build CUDA extension
cd p40-pearl-gemm
CUTLASS_DIR=/path/to/cutlass/include pip install -e .

# Mine
p40-mine --wallet prl1YOURWALLET --worker p40
```

On Windows:

```
run_luckypool.bat --wallet prl1YOURWALLET --worker p40
```

### Required dependencies

- CUDA Toolkit 12.x
- PyTorch 2.x (matching CUDA)
- py-pearl-mining (Rust proof builder: `pip install py-pearl-mining`)
- CUTLASS headers (set `CUTLASS_DIR` or place at `~/.cache/cutlass/include`)

## Architecture

Split pipeline: **GEMM-only kernel** → transcript buffer → **BLAKE3-only kernel**.

Decouples the shared-memory-bound GEMM step (S=128 staging width for 4 blocks/SM = 100% warp occupancy) from the compute-bound BLAKE3 step. Bit-exact with the fused kernel at +37% throughput.

## Supported GPUs

- Tesla P40 (GP102) — 30 SM, 24 GB, TCC or WDDM
- Tesla P4 (GP104) — 20 SM, 8 GB
- GTX 1070/1080/1080 Ti — Pascal consumer cards
- **Not supported**: Volta+, Maxwell, AMD

## License

MIT (custom CUDA kernels). py-pearl-mining and alpha-miner upstream components under their respective licenses.
