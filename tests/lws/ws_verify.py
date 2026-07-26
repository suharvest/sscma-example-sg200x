#!/usr/bin/env python3
"""debug_stream ws_transport refactor verification. Raw sockets, no deps."""
import base64, os, socket, struct, sys, time

HOST, PORT = "192.168.42.1", 8001


def handshake(path, timeout=5):
    """Open a WS connection. Returns (sock, status_code, headers_blob)."""
    s = socket.create_connection((HOST, PORT), timeout=timeout)
    key = base64.b64encode(os.urandom(16)).decode()
    req = (f"GET {path} HTTP/1.1\r\nHost: {HOST}:{PORT}\r\n"
           f"Upgrade: websocket\r\nConnection: Upgrade\r\n"
           f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n")
    s.sendall(req.encode())
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = s.recv(4096)
        if not chunk:
            break
        buf += chunk
    head, _, rest = buf.partition(b"\r\n\r\n")
    status = int(head.split(b" ")[1]) if b" " in head else 0
    return s, status, head, rest


def read_frame(s, carry=b"", timeout=8):
    """Read one unmasked server->client WS frame. Returns (opcode, payload, carry)."""
    s.settimeout(timeout)
    buf = carry

    def need(n):
        nonlocal buf
        while len(buf) < n:
            c = s.recv(65536)
            if not c:
                raise ConnectionError("closed")
            buf += c

    need(2)
    op = buf[0] & 0x0F
    ln = buf[1] & 0x7F
    off = 2
    if ln == 126:
        need(4); ln = struct.unpack(">H", buf[2:4])[0]; off = 4
    elif ln == 127:
        need(10); ln = struct.unpack(">Q", buf[2:10])[0]; off = 10
    need(off + ln)
    payload = buf[off:off + ln]
    return op, payload, buf[off + ln:]


def nal_type(b):
    if len(b) >= 4 and b[:4] == b"\x00\x00\x00\x01":
        return b[4] & 0x1F
    if len(b) >= 3 and b[:3] == b"\x00\x00\x01":
        return b[3] & 0x1F
    return -1


results = []
def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  |  {detail}" if detail else ""))


# ---------------------------------------------------------------- T1 video
print("\n--- T1: /  video upgrade + frame format ---")
s1, st, head, carry = handshake("/")
check("T1.1 video path returns 101", st == 101, f"status={st}")
op, payload, carry = read_frame(s1, carry)
check("T1.2 first message is binary", op == 2, f"opcode={op}")
nt = nal_type(payload)
check("T1.3 first frame is a keyframe (SPS=7 or IDR=5)", nt in (7, 5), f"nal_type={nt}")
ts = struct.unpack("<Q", payload[-8:])[0]
now = int(time.time() * 1000)
check("T1.4 8-byte LE unix-ms tail is sane", abs(now - ts) < 60000,
      f"tail={ts} now={now} delta={now-ts}ms")

n, tot = 0, 0
t0 = time.time()
while time.time() - t0 < 3.0:
    try:
        op, payload, carry = read_frame(s1, carry, timeout=3)
        n += 1; tot += len(payload)
    except Exception:
        break
check("T1.5 frames keep flowing", n >= 10, f"{n} frames / {tot} bytes in 3s")

# ---------------------------------------------------------------- T2 results
print("\n--- T2: /results upgrade ---")
s2, st2, _, _ = handshake("/results")
check("T2.1 results path returns 101", st2 == 101, f"status={st2}")

# ---------------------------------------------------------------- T3 404
print("\n--- T3: unknown path ---")
s3, st3, head3, rest3 = handshake("/definitely-not-a-path")
check("T3.1 unknown path returns 404", st3 == 404, f"status={st3}")
# The message must be conveyed; the framing is backend-specific (mongoose
# writes the body verbatim, lws wraps it in a generated HTML page). See the
# on_upgrade contract in ws_transport.h.
check("T3.2 refusal body carries the message", b"not found" in rest3,
      f"body={rest3[:24]!r}")
s3.close()

# ---------------------------------------------------------------- T4 limit
print("\n--- T4: video client limit (max 2) ---")
sA, stA, _, _ = handshake("/")          # s1 already holds slot 1, this is slot 2
check("T4.1 second video client accepted", stA == 101, f"status={stA}")
sB, stB, headB, restB = handshake("/")  # third -> should be refused
check("T4.2 third video client refused with 503", stB == 503, f"status={stB}")
check("T4.3 refusal body mentions the limit", b"limit reached" in restB,
      f"body={restB[:40]!r}")
sB.close()

# ---------------------------------------------------------------- T5 counts
print("\n--- T5: client count released on close ---")
sA.close(); s1.close()
time.sleep(1.5)
sC, stC, _, _ = handshake("/")
check("T5.1 slot freed after disconnect", stC == 101, f"status={stC}")

# ---------------------------------------------------------------- T6 backpressure
print("\n--- T6: slow client backpressure (the critical one) ---")
# sC is our SLOW client: handshake done, then never read. Its send buffer fills,
# the server should mark it DS_NEEDS_IDR and skip frames rather than buffering
# without bound or blocking the VENC callback thread.
slow_sock = sC
slow_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)

# A healthy client on the other slot must keep receiving throughout.
sH, stH, _, carryH = handshake("/")
check("T6.1 healthy client alongside slow one", stH == 101, f"status={stH}")

print("     starving the slow client for 20s ...")
t0 = time.time()
hn, hbytes, gaps, last = 0, 0, [], time.time()
while time.time() - t0 < 20.0:
    try:
        op, payload, carryH = read_frame(sH, carryH, timeout=5)
        now_t = time.time()
        gaps.append(now_t - last); last = now_t
        hn += 1; hbytes += len(payload)
    except Exception as e:
        check("T6.x healthy client stayed connected", False, f"died: {e}")
        break

check("T6.2 healthy client kept streaming while peer starved",
      hn >= 40, f"{hn} frames / {hbytes} bytes in 20s")
if gaps:
    worst = max(gaps)
    check("T6.3 no long stall on the healthy client (<2s worst gap)",
          worst < 2.0, f"worst gap={worst:.2f}s avg={sum(gaps)/len(gaps):.3f}s")

# Server must still accept new work after all that.
slow_sock.close(); time.sleep(1.0)
sR, stR, _, _ = handshake("/results")
check("T6.4 server still healthy after backpressure episode", stR == 101, f"status={stR}")
sR.close(); sH.close(); s2.close()

# ---------------------------------------------------------------- summary
print("\n" + "=" * 60)
bad = [r for r in results if not r[1]]
print(f"{len(results)-len(bad)}/{len(results)} passed")
for name, ok, detail in bad:
    print(f"  FAILED: {name}  {detail}")
sys.exit(1 if bad else 0)
