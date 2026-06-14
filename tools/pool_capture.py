"""Transparent logging TCP proxy for reverse-engineering a Pearl pool's protocol.

Listens locally, forwards to the real pool, and logs every line in both
directions (C->P = miner->pool, P->C = pool->miner) to a file. Point a real
miner at this proxy instead of the pool, mine until it submits a share, and the
submit message + accept response are captured.

    python pool_capture.py [LISTEN_PORT] [UPSTREAM_HOST:PORT] [LOGFILE]
    python pool_capture.py 5566 eu1.alphapool.tech:5566 alphapool_capture.log
"""
import datetime
import socket
import sys
import threading

LISTEN_PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 5566
_up = sys.argv[2] if len(sys.argv) > 2 else "eu1.alphapool.tech:5566"
UP_HOST, UP_PORT = _up.rsplit(":", 1)
UP_PORT = int(UP_PORT)
LOGFILE = sys.argv[3] if len(sys.argv) > 3 else "pool_capture.log"

_lock = threading.Lock()


def log(direction, data: bytes):
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    with _lock, open(LOGFILE, "a", encoding="utf-8") as f:
        for line in data.split(b"\n"):
            if line.strip():
                f.write(f"{ts} {direction} {line.decode('utf-8', 'replace')}\n")


def pipe(src, dst, direction):
    try:
        while True:
            d = src.recv(65536)
            if not d:
                break
            log(direction, d)
            dst.sendall(d)
    except OSError:
        pass
    finally:
        for s in (src, dst):
            try:
                s.close()
            except OSError:
                pass


def handle(client, addr):
    print(f"client connected: {addr}", flush=True)
    up = None
    for attempt in range(12):  # flaky path: retry the upstream connect
        try:
            up = socket.create_connection((UP_HOST, UP_PORT), timeout=5)
            break
        except OSError:
            continue
    if up is None:
        print("upstream connect failed after retries", flush=True)
        client.close()
        return
    print(f"upstream connected (attempt {attempt + 1})", flush=True)
    threading.Thread(target=pipe, args=(client, up, "C->P"), daemon=True).start()
    threading.Thread(target=pipe, args=(up, client, "P->C"), daemon=True).start()


def main():
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", LISTEN_PORT))
    srv.listen(8)
    print(f"proxy 0.0.0.0:{LISTEN_PORT} -> {UP_HOST}:{UP_PORT} | log -> {LOGFILE}",
          flush=True)
    while True:
        c, addr = srv.accept()
        threading.Thread(target=handle, args=(c, addr), daemon=True).start()


if __name__ == "__main__":
    main()
