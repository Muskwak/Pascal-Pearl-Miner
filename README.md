# Pascal Pearl Miner

A high-performance **Pearl (PRL)** proof-of-work miner for **NVIDIA Pascal GPUs** —
Tesla P40, GTX 1070 / 1080 (and other `sm_61`, DP4A-capable cards).

<<<<<<< HEAD
No Python, CUDA toolkit, or PyTorch required — the CUDA runtime is bundled. Just an
NVIDIA driver and the standalone binary. **Windows, Linux, and HiveOS** builds are available.
=======
No Python, CUDA toolkit, or PyTorch required to **run** — the CUDA runtime is bundled
in pre-built releases. Just an NVIDIA driver and the standalone binary.
>>>>>>> a836de3 (Open-source v1.3.0: full source release (MIT))

**Source is open (MIT)** — build it yourself or grab a pre-built release.

## Features

- **~7.0 TH/s** sustained on a single Tesla P40.
- **Multi-GPU** — auto-detects every GPU and runs one worker per card, near-linear scaling.
- **Continuous mining** — no idle time waiting between pool jobs.
- **Background proof submission** — finding a share never stalls the search.
- **Solo mining** — mine to your local pearl-gateway node.
- **Pool mining** — built for LuckyPool's Pearl stratum (default) with more pool protocols planned.
- **HiveOS** — full HiveOS custom-miner package.

## Pre-built Releases

<<<<<<< HEAD
- An NVIDIA Pascal GPU (Tesla P40, GTX 1070/1080, or other `sm_61`)
- An up-to-date NVIDIA driver (the CUDA runtime is bundled — no CUDA install needed)
- **Windows x64**, **Linux x86-64** (Ubuntu 20.04 / 22.04 / 24.04, glibc ≥ 2.30), or **HiveOS**

> Mixed-GPU rigs are supported: native code is compiled for Pascal/Turing/Ampere/Ada
> (`sm_61/75/86/89`) plus a PTX fallback that JIT-loads on any newer NVIDIA card.
> Pascal is the optimized target; newer cards run functionally via the DP4A path.
=======
Grab the latest binary from the [Releases](https://github.com/Muskwak/Pascal-Pearl-Miner/releases) page:
>>>>>>> a836de3 (Open-source v1.3.0: full source release (MIT))

| File | Platform |
|------|----------|
| `p40-miner-windows-x64.zip` | Windows x64 |
| `p40-miner-linux-x64.tar.gz` | Linux (glibc >= 2.31) |
| `p40-miner-hiveos-<ver>.tar.gz` | HiveOS custom miner |

<<<<<<< HEAD
Grab the latest from the
[Releases](https://github.com/Muskwak/Pascal-Pearl-Miner/releases) page:
=======
### Quick Start

```bat
:: Windows
p40-miner.exe --wallet prl1YOURWALLET --worker rig1
```

```bash
# Linux
./p40-miner --wallet prl1YOURWALLET --worker rig1
```

That's it — it auto-detects all GPUs and starts mining.

## Build from Source

### Prerequisites

| Dependency | Windows | Linux |
|---|---|---|
| **CUDA Toolkit** | 12.x ([NVIDIA](https://developer.nvidia.com/cuda-downloads)) | 12.x |
| **CUTLASS headers** | `git clone --depth 1 https://github.com/NVIDIA/cutlass` | same |
| **Python** | >= 3.12 | >= 3.12 |
| **MSVC** | Visual Studio 2022 (C++ workload) | — |
| **pip packages** | `numpy`, `blake3`, `py-pearl-mining` | same |

Set `CUTLASS_DIR` to the directory containing `cutlass/` and `cute/` headers.

### Windows

```bat
git clone https://github.com/Muskwak/Pascal-Pearl-Miner.git
cd Pascal-Pearl-Miner

:: 1. Build the CUDA library (p40cuda.dll)
packaging\build_capi.bat

:: 2. Install Python deps
pip install numpy blake3 py-pearl-mining

:: 3. Run directly from source
python packaging\p40_miner_lite_main.py --wallet prl1YOURWALLET

:: Or freeze a standalone binary with PyInstaller:
pip install pyinstaller
pyinstaller packaging\p40-miner-lite.spec --noconfirm --distpath dist --workpath build_pyi
dist\p40-miner\p40-miner.exe --wallet prl1YOURWALLET
```

To rebuild the full CUDA extension (includes torch bindings):

```bat
pip install -e .   # requires torch + CUTLASS_DIR
```

### Linux

```bash
git clone https://github.com/Muskwak/Pascal-Pearl-Miner.git
cd Pascal-Pearl-Miner

# 1. Build the CUDA library (libp40cuda.so)
CUTLASS_DIR=/path/to/cutlass/include bash packaging/build_capi.sh

# 2. Install Python deps
pip install numpy blake3 py-pearl-mining

# 3. Run directly from source
python packaging/p40_miner_lite_main.py --wallet prl1YOURWALLET

# Or freeze a standalone binary:
pip install pyinstaller
pyinstaller packaging/p40-miner-lite.spec --noconfirm --distpath dist --workpath build_pyi
./dist/p40-miner/p40-miner --wallet prl1YOURWALLET
```

### HiveOS Package

Build the Linux binary first, then:

```bash
bash packaging/hiveos/build_hiveos_package.sh  # -> p40-miner-hiveos-<ver>.tar.gz
```

Upload to a URL or `scp` to the rig, then add as a Custom miner in HiveOS.

### Development Install

```bash
pip install -e .   # editable install of the torch-based extension
CUTLASS_DIR=... python -c "import p40_pearl_gemm"  # smoke test
```
>>>>>>> a836de3 (Open-source v1.3.0: full source release (MIT))

| File | Platform |
|------|----------|
| `p40-miner-windows-x64.zip`     | Windows x64 |
| `p40-miner-linux-x64.tar.gz`    | Linux (Ubuntu 20.04 / 22.04 / 24.04, and other distros) |
| `p40-miner-hiveos-1.2.1.tar.gz` | HiveOS custom miner |

## Usage (Windows / Linux)

Windows:
```
p40-miner --wallet prl1YOURWALLET --worker rig1
```

Linux:
```
tar xzf p40-miner-linux-x64.tar.gz
./p40-miner-linux-x64/p40-miner --wallet prl1YOURWALLET --worker rig1
```

### Options

<<<<<<< HEAD
| Flag       | Default                             | Description                       |
|------------|-------------------------------------|-----------------------------------|
| `--wallet` | _(required)_                        | Your Pearl payout address         |
| `--worker` | `p40`                               | Worker name shown on the pool     |
| `--pool`   | `pearl-cpu-eu1.luckypool.io:3370`   | Stratum `host:port`               |
| `--devices`| _(auto-detect all)_                 | GPUs to use, e.g. `0,1,2` or `all`|
| `--region` | `4096`                              | Sub-output search size            |
| `--solo`   | _(off)_                             | Solo mine to a local pearl-gateway `HOST:PORT` (rewards → your node) |

## HiveOS

Download `p40-miner-hiveos-1.2.1.tar.gz` and install it as a **Custom** miner:

1. **Flight Sheet → Miner = Custom.**
2. Set the **Installation URL** to the tarball, *or* `scp` it to the rig and run
   `tar -C /hive/miners/custom -xzf p40-miner-hiveos-1.2.1.tar.gz`.
3. **Wallet and worker:** your Pearl wallet `prl1...` (worker name auto-appended).
4. **Pool URL:** `pearl-cpu-eu1.luckypool.io:3370` (default LuckyPool).
5. **Extra config arguments** (optional): `--devices 0,1`, `--region 4096`, etc.

The miner reports per-GPU TH/s and accepted shares to the HiveOS dashboard.
Built against glibc 2.31, so it runs on both the *focal* (20.04) and *jammy* (22.04)
HiveOS images and newer.
=======
| Flag | Default | Description |
|---|---|---|
| `--wallet` | _(required)_ | Your Pearl payout address |
| `--worker` | `p40` | Worker name shown on the pool |
| `--pool` | `pearl-cpu-eu1.luckypool.io:3370` | Stratum `host:port` |
| `--devices` | _(auto-detect all)_ | GPU selection, e.g. `0,1,2` or `all` |
| `--region` | `4096` | Sub-output search size |
| `--solo` | _(off)_ | Solo mine to local pearl-gateway `HOST:PORT` |
>>>>>>> a836de3 (Open-source v1.3.0: full source release (MIT))

### Multi-GPU

<<<<<<< HEAD
By default the miner **auto-detects every GPU** and runs one worker per card
(worker names auto-suffixed `-gpu0`, `-gpu1`, …), with a combined-hashrate summary.
Just run it normally on a rig — no flags needed. To select specific cards, use
`--devices 0,2`. Each GPU runs as an independent pinned process (with blocking-sync
CUDA so the cards don't fight over CPU), giving near-linear scaling on 4–8 GPU rigs.

## Pool

Built for **LuckyPool**'s Pearl stratum. By default the miner connects to
`pearl-cpu-eu1.luckypool.io:3370`. Use `--pool HOST:PORT` to select another
LuckyPool region or port. Currently only LuckyPool's stratum protocol is supported.
**Support for additional pools will be added in the next update.**
=======
Auto-detects every GPU and runs one worker per card (workers named `<worker>-gpu0`,
`-gpu1`, ...) with a combined-hashrate summary. Just run it:

```
p40-miner.exe --wallet prl1YOURWALLET
```

Use `--devices` to select specific cards:

```
p40-miner.exe --wallet prl1YOURWALLET --devices 0,2
```

### Pool

Default: `pearl-cpu-eu1.luckypool.io:3370` (LuckyPool). Override with `--pool`:

```
p40-miner.exe --wallet prl1YOURWALLET --pool pearl-cpu-eu2.luckypool.io:3370
```

Currently only LuckyPool's stratum protocol is supported. More pools planned.

### Solo Mining

```
p40-miner.exe --solo GATEWAY_HOST:PORT
```

No wallet flag needed — the node's configured address receives block rewards.
The same 2% dev fee applies (GPU mines to the dev's pool wallet during the 2% window).
>>>>>>> a836de3 (Open-source v1.3.0: full source release (MIT))

## Dev Fee

A transparent **2%** dev fee is included: for 2% of cumulative mining time the miner
mines to the developer's address. This is disclosed at startup and logged on every
<<<<<<< HEAD
switch, so you can always see exactly when it is active. This applies in **`--solo`
mode too** — for those 2% the GPU mines to the dev's pool wallet instead of your node;
the rest of the time solo block rewards go to your node's configured address.
=======
switch, so you can always see exactly when it is active. The 2% applies to both pool
and solo mining.
>>>>>>> a836de3 (Open-source v1.3.0: full source release (MIT))

Thank you for supporting development!

## Tip / Donate

If this miner helped you save time or earn PRL, consider donating to the developer:

```
prl1pfu7yr6u6mfkku3mh2deyuwegcnpaunjz4vlsvaj2shg2qjkaux2q76uyud
```

Every tip is appreciated and helps fund continued development.

## License

**MIT License** — see [LICENSE](LICENSE).

Copyright (c) 2025 Muskwak / Pascal-Pearl-Miner. You must retain the copyright notice
in all copies (attribution to this repository), but you are free to use, modify, and
distribute the code for any purpose, including commercial use.
