# Pascal Pearl Miner

A high-performance **Pearl (PRL)** proof-of-work miner for **NVIDIA Pascal GPUs** —
Tesla P40, GTX 1070 / 1080 (and other `sm_61`, DP4A-capable cards).

No Python, CUDA toolkit, or PyTorch required — the CUDA runtime is bundled. Just an
NVIDIA driver and the standalone binary.

## Performance

- **~7.0 TH/s** sustained on a single Tesla P40.
- **Continuous mining** — no idle time waiting between pool jobs.
- **Background proof submission** — finding a share never stalls the search.

## Requirements

- An NVIDIA Pascal GPU (Tesla P40, GTX 1070/1080, or other `sm_61`)
- An up-to-date NVIDIA driver (the CUDA runtime is bundled — no CUDA install needed)
- Windows x64

## Download

Get the latest **`p40-miner-windows-x64.zip`** from the
[Releases](https://github.com/Muskwak/Pascal-Pearl-Miner/releases) page and extract it.

## Usage

```
p40-miner.exe --wallet prl1YOURWALLET --worker rig1
```

### Options

| Flag       | Default                             | Description                       |
|------------|-------------------------------------|-----------------------------------|
| `--wallet` | _(required)_                        | Your Pearl payout address         |
| `--worker` | `p40`                               | Worker name shown on the pool     |
| `--pool`   | `pearl-cpu-eu1.luckypool.io:3370`   | Stratum `host:port`               |
| `--region` | `4096`                              | Sub-output search size            |

## Pool

Built for **LuckyPool**'s Pearl stratum. By default the miner connects to:

```
pearl-cpu-eu1.luckypool.io:3370
```

Use `--pool HOST:PORT` to select another LuckyPool region or port:

```
p40-miner.exe --wallet prl1YOURWALLET --pool pearl-cpu-eu1.luckypool.io:3370
```

Currently only LuckyPool's stratum protocol is supported. **Support for additional
pools will be added in the next update.**

## Dev fee

A transparent **2%** dev fee is included: for 2% of cumulative mining time the miner
mines to the developer's address. This is disclosed at startup and logged on every
switch, so you can always see exactly when it is active.

Thank you for supporting development!

## Notes

- Closed-source binary distribution.
- The bundled CUDA runtime means the only external dependency is your GPU driver.
