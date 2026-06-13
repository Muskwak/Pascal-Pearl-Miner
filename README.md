# Pascal Pearl Miner

A high-performance **Pearl (PRL)** proof-of-work miner for **NVIDIA Pascal GPUs** —
Tesla P40, GTX 1070 / 1080 (and other `sm_61`, DP4A-capable cards).

No Python, CUDA toolkit, or PyTorch required — the CUDA runtime is bundled. Just an
NVIDIA driver and the standalone binary. **Windows, Linux, and HiveOS** builds are available.

## Performance

- **~7.0 TH/s** sustained on a single Tesla P40.
- **Multi-GPU** — auto-detects every GPU and runs one worker per card, near-linear scaling.
- **Continuous mining** — no idle time waiting between pool jobs.
- **Background proof submission** — finding a share never stalls the search.

## Requirements

- An NVIDIA Pascal GPU (Tesla P40, GTX 1070/1080, or other `sm_61`)
- An up-to-date NVIDIA driver (the CUDA runtime is bundled — no CUDA install needed)
- **Windows x64**, **Linux x86-64** (Ubuntu 20.04 / 22.04 / 24.04, glibc ≥ 2.30), or **HiveOS**

> Mixed-GPU rigs are supported: native code is compiled for Pascal/Turing/Ampere/Ada
> (`sm_61/75/86/89`) plus a PTX fallback that JIT-loads on any newer NVIDIA card.
> Pascal is the optimized target; newer cards run functionally via the DP4A path.

## Download

Grab the latest from the
[Releases](https://github.com/Muskwak/Pascal-Pearl-Miner/releases) page:

| File | Platform |
|------|----------|
| `p40-miner-windows-x64.zip`     | Windows x64 |
| `p40-miner-linux-x64.tar.gz`    | Linux (Ubuntu 20.04 / 22.04 / 24.04, and other distros) |
| `p40-miner-hiveos-1.2.0.tar.gz` | HiveOS custom miner |

## Usage (Windows / Linux)

Windows:
```
p40-miner.exe --wallet prl1YOURWALLET --worker rig1
```

Linux:
```
tar xzf p40-miner-linux-x64.tar.gz
./p40-miner-linux-x64/p40-miner --wallet prl1YOURWALLET --worker rig1
```

### Options

| Flag       | Default                             | Description                       |
|------------|-------------------------------------|-----------------------------------|
| `--wallet` | _(required)_                        | Your Pearl payout address         |
| `--worker` | `p40`                               | Worker name shown on the pool     |
| `--pool`   | `pearl-cpu-eu1.luckypool.io:3370`   | Stratum `host:port`               |
| `--devices`| _(auto-detect all)_                 | GPUs to use, e.g. `0,1,2` or `all`|
| `--region` | `4096`                              | Sub-output search size            |
| `--solo`   | _(off)_                             | Solo mine to a local pearl-gateway `HOST:PORT` (no pool/wallet/dev fee) |

## HiveOS

Download `p40-miner-hiveos-1.2.0.tar.gz` and install it as a **Custom** miner:

1. **Flight Sheet → Miner = Custom.**
2. Set the **Installation URL** to the tarball, *or* `scp` it to the rig and run
   `tar -C /hive/miners/custom -xzf p40-miner-hiveos-1.2.0.tar.gz`.
3. **Wallet and worker:** your Pearl wallet `prl1...` (worker name auto-appended).
4. **Pool URL:** `pearl-cpu-eu1.luckypool.io:3370` (default LuckyPool).
5. **Extra config arguments** (optional): `--devices 0,1`, `--region 4096`, etc.

The miner reports per-GPU TH/s and accepted shares to the HiveOS dashboard.
Built against glibc 2.31, so it runs on both the *focal* (20.04) and *jammy* (22.04)
HiveOS images and newer.

## Multi-GPU

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

## Dev fee

A transparent **2%** dev fee is included: for 2% of cumulative mining time the miner
mines to the developer's address. This is disclosed at startup and logged on every
switch, so you can always see exactly when it is active. (Solo mode via `--solo` has
no dev fee — block rewards go to your node's configured address.)

Thank you for supporting development!

## Notes

- Closed-source binary distribution.
- The bundled CUDA runtime means the only external dependency is your GPU driver.
