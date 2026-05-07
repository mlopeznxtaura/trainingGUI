"""
headless/server.py — Transport-layer server
"""
import json
import sys
import socket
import threading
import os
import pathlib
from http.server import BaseHTTPRequestHandler, HTTPServer


class HeadlessServer:

    def __init__(self, orchestrator, config: dict):
        self._orchestrator = orchestrator
        self._config = config
        self._running = False
        self._thread = None

    def start(self) -> None:
        transport = self._config.get("headless", {}).get("transport", "http")

        transport_map = {
            "stdio": self._run_stdio,
            "http": self._run_http,
            "unix_socket": self._run_unix_socket,
        }

        run_fn = transport_map.get(transport, self._run_http)
        self._running = True
        self._orchestrator.running = True
        self._thread = threading.Thread(target=run_fn, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._orchestrator.running = False

    def _run_stdio(self) -> None:
        orchestrator = self._orchestrator
        while self._running:
            try:
                line = sys.stdin.buffer.readline()
                if not line:
                    break
                payload = json.loads(line.decode("utf-8"))
                method = payload.get("method", "")

                if method == "infer":
                    result = orchestrator.infer(payload)
                    sys.stdout.write(json.dumps(result) + "\n")
                    sys.stdout.flush()
                elif method == "stream":
                    for token in orchestrator.stream(payload):
                        sys.stdout.write(json.dumps({"token": token}) + "\n")
                        sys.stdout.flush()
                    sys.stdout.write(json.dumps({"done": True}) + "\n")
                    sys.stdout.flush()
                elif method == "set_weight":
                    result = orchestrator.set_weight(payload["key"], payload["value"])
                    sys.stdout.write(json.dumps(result) + "\n")
                    sys.stdout.flush()
                elif method == "get_weights":
                    result = orchestrator.get_weights()
                    sys.stdout.write(json.dumps(result) + "\n")
                    sys.stdout.flush()
                else:
                    sys.stdout.write(json.dumps({"error": f"Unknown method: {method}", "code": 404}) + "\n")
                    sys.stdout.flush()
            except json.JSONDecodeError as e:
                sys.stdout.write(json.dumps({"error": str(e), "code": 400, "context": "json_decode"}) + "\n")
                sys.stdout.flush()
            except Exception:
                break

    def _run_http(self) -> None:
        orchestrator = self._orchestrator

        class _HTTPHandler(BaseHTTPRequestHandler):

            _orchestrator = orchestrator

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode())
                    return

                if self.path == "/infer":
                    result = self._orchestrator.infer(payload)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(result).encode())

                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Transfer-Encoding", "chunked")
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    for token in self._orchestrator.stream(payload):
                        chunk = json.dumps({"token": token}).encode()
                        self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")

                elif self.path == "/weights":
                    result = self._orchestrator.get_weights()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(result).encode())

                elif self.path == "/set_weight":
                    result = self._orchestrator.set_weight(payload.get("key"), payload.get("value"))
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(result).encode())

                else:
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Not found"}).encode())

            def log_message(self, format, *args):
                pass  # suppress default access logs

        port = self._config.get("headless", {}).get("port", 8765)
        server = HTTPServer(("0.0.0.0", port), _HTTPHandler)
        server.serve_forever()

    def _run_unix_socket(self) -> None:
        if sys.platform == "win32":
            # TCP fallback on port+1 (gap_04)
            port = self._config.get("headless", {}).get("port", 8765) + 1
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("localhost", port))
        else:
            platform_key = "darwin" if sys.platform == "darwin" else "linux"
            socket_dir = self._config.get("os_paths", {}).get(platform_key, {}).get("socket_dir", "/tmp")
            socket_name = self._config.get("headless", {}).get("unix_socket_name", "aig.sock")
            socket_path = pathlib.Path(socket_dir) / socket_name

            if socket_path.exists():
                socket_path.unlink()

            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.bind(str(socket_path))

        sock.listen(5)
        orchestrator = self._orchestrator

        while self._running:
            try:
                conn, _ = sock.accept()
                data = b""
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                    if b"\n" in data:
                        break

                payload = json.loads(data.decode("utf-8").strip())
                method = payload.get("method", "")

                if method == "infer":
                    result = orchestrator.infer(payload)
                elif method == "get_weights":
                    result = orchestrator.get_weights()
                elif method == "set_weight":
                    result = orchestrator.set_weight(payload["key"], payload["value"])
                else:
                    result = {"error": f"Unknown method: {method}", "code": 404}

                conn.sendall((json.dumps(result) + "\n").encode("utf-8"))
                conn.close()
            except Exception:
                pass
