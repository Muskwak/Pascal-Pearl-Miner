"""Transparent logging TCP proxy to capture a pool's wire protocol.

Run the OFFICIAL pearl/alpha miner against this proxy (it forwards to the real
pool and logs every line both directions). Point the official miner at
`stratum+tcp://<this-host>:<LISTEN_PORT>`. You only need to run the official
miner on a SUPPORTED GPU (sm_70+) long enough to capture one `mining.submit`
(plus the subscribe/authorize handshake and the challenge/job format). Then we
replicate that exact format in pool_miner.py and mine on the P40.

Usage:
    python pool_sniffer.py --listen 0.0.0.0:5566 --pool us2.alphapool.tech:5566
    # then run the official miner: ... --pool stratum+tcp://<host>:5566 --address prl1...

Captured traffic is printed and appended to pool_capture.log (pretty-printed
JSON where possible, so the seed->header mapping and submit format are readable).
"""
import argparse
import json
import socket
import threading
import time

LOG = "pool_capture.log"


def log(direction: str, data: bytes):
    ts = time.strftime("%H:%M:%S")
    for raw in data.split(b"\n"):
        raw = raw.strip()
        if not raw:
            continue
        try:
            pretty = json.dumps(json.loads(raw), indent=None)
        except Exception:
            pretty = raw.decode("replace")
        line = f"[{ts}] {direction} {pretty}"
        print(line, flush=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def pump(src: socket.socket, dst: socket.socket, direction: str):
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            log(direction, data)
            dst.sendall(data)
    except OSError:
        pass
    finally:
        for s in (src, dst):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def handle(client: socket.socket, pool_host: str, pool_port: int):
    peer = client.getpeername()
    print(f"\n=== miner {peer} connected; dialing pool {pool_host}:{pool_port} ===", flush=True)
    try:
        upstream = socket.create_connection((pool_host, pool_port), timeout=30)
    except OSError as e:
        print(f"pool connect failed: {e}", flush=True)
        client.close()
        return
    t1 = threading.Thread(target=pump, args=(client, upstream, "miner->POOL"), daemon=True)
    t2 = threading.Thread(target=pump, args=(upstream, client, "POOL->miner"), daemon=True)
    t1.start(); t2.start()
    t1.join(); t2.join()
    print(f"=== session {peer} closed ===", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", default="0.0.0.0:5566")
    ap.add_argument("--pool", required=True, help="real pool host:port")
    args = ap.parse_args()
    lhost, lport = args.listen.rsplit(":", 1)
    phost, pport = args.pool.rsplit(":", 1)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((lhost, int(lport)))
    srv.listen(5)
    print(f"sniffer listening on {args.listen} -> {args.pool}  (logging to {LOG})", flush=True)
    while True:
        client, _ = srv.accept()
        threading.Thread(target=handle, args=(client, phost, int(pport)), daemon=True).start()


if __name__ == "__main__":
    main()
