## Pascal Pearl Miner v1.3.0

Ampere Tensor Core support — INT8 inline-PTX `mma.sync.aligned.m16n8k32` GEMM for sm_86+ GPUs.

### Added
- **Ampere/Ada Tensor Core kernels** — automatic dispatch for RTX 30xx, RTX 40xx, and other sm_80+ GPUs.
  Uses inline PTX `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` with `cp.async` double-buffered shared memory pipeline.
  Fallback to DP4A on pre-Ampere GPUs (Pascal, Volta, Turing). No user configuration required.
- Windows build includes sm_86 + sm_89 cubins via `p40cuda.dll`.
- Python wheel (`pip install`) auto-detects GPU and selects the optimal kernel path.

### Changed
- Build: `packaging/build_capi.bat` now requires CUTLASS headers (`CUTLASS_DIR` or auto-detected fallback).
- Build: `setup.py` unconditionally includes sm_80+ gencode for TC kernel support.

### Fixed
- Removed unused `<cute/tensor.hpp>` includes from Pascal-specific kernel files — resolves CUTLASS header dependency in those translation units.

### Downloads
| File | Platform |
|------|----------|
| `p40-miner-windows-x64.zip`     | Windows x64 (sm_61 + sm_86 + sm_89) |
| `p40-miner-linux-x64.tar.gz`    | Linux (Ubuntu 20.04 / 22.04 / 24.04) |
| `p40-miner-hiveos-1.3.0.tar.gz` | HiveOS custom miner |

### Known Issues
- No fused TC+BLAKE3 kernel (split pipeline only: GEMM → BLAKE3).
- sm_80 (compute capability 8.0) cubins included but untested.
