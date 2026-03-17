#!/usr/bin/env python3
"""
Retail Vision Debug Server
- Serves the debug HTML page
- Proxies RTSP → MJPEG via ffmpeg at /stream
"""

import argparse
import subprocess
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
import signal
import sys

ffmpeg_procs = []


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=directory or STATIC_DIR, **kwargs)

    def do_GET(self):
        if self.path.startswith("/stream"):
            self.handle_mjpeg_stream()
        else:
            super().do_GET()

    def handle_mjpeg_stream(self):
        # Parse query params
        from urllib.parse import urlparse, parse_qs
        query = parse_qs(urlparse(self.path).query)
        rtsp_url = query.get("url", [DEFAULT_RTSP])[0]

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        cmd = [
            "ffmpeg",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-probesize", "32",
            "-analyzeduration", "0",
            "-rtsp_transport", "tcp",
            "-i", rtsp_url,
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "-q:v", "5",
            "-r", "15",
            "-an",
            "-",
        ]

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        ffmpeg_procs.append(proc)

        buf = b""
        try:
            while True:
                chunk = proc.stdout.read(32768)
                if not chunk:
                    break
                buf += chunk

                while True:
                    start = buf.find(b"\xff\xd8")
                    if start < 0:
                        buf = b""
                        break
                    end = buf.find(b"\xff\xd9", start + 2)
                    if end < 0:
                        buf = buf[start:]
                        break
                    frame = buf[start : end + 2]
                    buf = buf[end + 2 :]
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(frame)}\r\n\r\n".encode()
                        )
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    except BrokenPipeError:
                        break
        except Exception:
            pass
        finally:
            proc.terminate()
            proc.wait()
            if proc in ffmpeg_procs:
                ffmpeg_procs.remove(proc)

    def log_message(self, format, *args):
        msg = str(args[0]) if args else ""
        if "/stream" not in msg:
            super().log_message(format, *args)


def cleanup(signum=None, frame=None):
    for p in ffmpeg_procs:
        try:
            p.terminate()
        except Exception:
            pass
    sys.exit(0)


STATIC_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RTSP = "rtsp://192.168.10.158:8554/live0"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retail Vision Debug Server")
    parser.add_argument("-p", "--port", type=int, default=8080)
    parser.add_argument("--rtsp", default=DEFAULT_RTSP, help="Default RTSP URL")
    args = parser.parse_args()

    DEFAULT_RTSP = args.rtsp

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Debug server: http://localhost:{args.port}")
    print(f"MJPEG proxy:  http://localhost:{args.port}/stream")
    print(f"Default RTSP: {DEFAULT_RTSP}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        cleanup()
