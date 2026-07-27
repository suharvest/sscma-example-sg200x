#!/usr/bin/env python3
"""持续负载 + 定期采样。专打新代码路径：
   - HTTP 请求 → 连接注册表 attach/detach（最可疑的泄漏点）
   - WS 连/断     → LwsConn 分配释放
   - /snapshot.jpg → on_http 暂存路径
   - multipart 上传 → 自写解析器
"""
import base64, json, os, socket, subprocess, sys, time, urllib.request

HOST = "192.168.42.1"
DUR = int(sys.argv[1]) if len(sys.argv) > 1 else 900

def login():
    d = json.dumps({"userName":"recamera","password":"706d8af33b5d6edf491703e9afdb30e2"}).encode()
    r = urllib.request.Request(f"http://{HOST}/api/userMgr/login", data=d,
                               headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r, timeout=10))["data"]["token"]

def ws(path, hold=0.3):
    s = socket.create_connection((HOST, 8001), timeout=5)
    k = base64.b64encode(os.urandom(16)).decode()
    s.sendall(f"GET {path} HTTP/1.1\r\nHost: h\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
              f"Sec-WebSocket-Key: {k}\r\nSec-WebSocket-Version: 13\r\n\r\n".encode())
    s.recv(4096); time.sleep(hold); s.close()

def probe():
    return subprocess.run(["ssh","-o","StrictHostKeyChecking=no","-o","UserKnownHostsFile=/dev/null",
                           "-o","BatchMode=yes",f"recamera@{HOST}","/tmp/probe.sh"],
                          capture_output=True, text=True, timeout=30).stdout

tok = login()
t0 = time.time(); n = 0
print(probe(), flush=True)
next_probe = t0 + 180
while time.time() - t0 < DUR:
    try:
        req = urllib.request.Request(f"http://{HOST}/api/deviceMgr/getDeviceInfo",
                                     headers={"Authorization": tok})
        urllib.request.urlopen(req, timeout=10).read()
        urllib.request.urlopen(f"http://{HOST}/", timeout=10).read()          # 静态
        urllib.request.urlopen(f"http://{HOST}:8001/snapshot.jpg", timeout=10).read()
        ws("/"); ws("/results")
        if n % 10 == 0:  # multipart
            body = b"--B\r\nContent-Disposition: form-data; name=\"file\"; filename=\"s.txt\"\r\n\r\n" \
                   + f"soak-{n}".encode() + b"\r\n--B--\r\n"
            r = urllib.request.Request(f"http://{HOST}/api/fileMgr/upload", data=body,
                headers={"Authorization": tok, "Content-Type": "multipart/form-data; boundary=B"})
            urllib.request.urlopen(r, timeout=15).read()
        n += 1
    except Exception as e:
        print(f"  [{int(time.time()-t0)}s] 迭代 {n} 异常: {type(e).__name__}: {e}", flush=True)
    if time.time() >= next_probe:
        print(f"--- 已跑 {int(time.time()-t0)}s / {n} 轮 ---", flush=True)
        print(probe(), flush=True)
        next_probe += 180
    time.sleep(0.5)
print(f"=== 结束：{n} 轮 ===", flush=True)
print(probe(), flush=True)
