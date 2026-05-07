"""
headless/server.py — Transport-layer server (gaps 4+5+7 fixed)
"""
import json, sys, socket, threading, os, pathlib
from http.server import BaseHTTPRequestHandler, HTTPServer


class HeadlessServer:

    def __init__(self, orchestrator, config: dict):
        self._orchestrator = orchestrator
        self._config = config
        self._running = False
        self._thread = None

    def start(self) -> None:
        transport = self._config.get("headless", {}).get("transport", "http")
        run_fn = {"stdio": self._run_stdio, "http": self._run_http, "unix_socket": self._run_unix_socket}.get(transport, self._run_http)
        self._running = True
        self._orchestrator.running = True  # gap 7 fix: set running flag
        self._thread = threading.Thread(target=run_fn, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._orchestrator.running = False

    def _run_stdio(self) -> None:
        orch = self._orchestrator
        while self._running:
            try:
                line = sys.stdin.buffer.readline()
                if not line:
                    break
                payload = json.loads(line.decode("utf-8"))
                method = payload.get("method", "")
                if method == "infer":
                    sys.stdout.write(json.dumps(orch.infer(payload)) + "\n"); sys.stdout.flush()
                elif method == "stream":
                    for token_json in orch.stream(payload):
                        sys.stdout.write(token_json + "\n"); sys.stdout.flush()
                    sys.stdout.write(json.dumps({"done": True}) + "\n"); sys.stdout.flush()
                elif method == "set_weight":
                    sys.stdout.write(json.dumps(orch.set_weight(payload["key"], payload["value"])) + "\n"); sys.stdout.flush()
                elif method == "get_weights":
                    sys.stdout.write(json.dumps(orch.get_weights()) + "\n"); sys.stdout.flush()
                else:
                    sys.stdout.write(json.dumps({"error": f"Unknown method: {method}", "code": 404}) + "\n"); sys.stdout.flush()
            except json.JSONDecodeError as e:
                sys.stdout.write(json.dumps({"error": str(e), "code": 400}) + "\n"); sys.stdout.flush()
            except Exception:
                break

    def _run_http(self) -> None:
        orch = self._orchestrator

        class _HTTPHandler(BaseHTTPRequestHandler):
            _orch = orch

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                try:
                    payload = json.loads(self.rfile.read(length))
                except json.JSONDecodeError:
                    self.send_response(400); self.end_headers()
                    self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode()); return

                if self.path == "/infer":
                    result = self._orch.infer(payload)
                    self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
                    self.wfile.write(json.dumps(result).encode())

                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Transfer-Encoding", "chunked")
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    try:
                        for token_json in self._orch.stream(payload):
                            chunk = token_json.encode("utf-8")
                            self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                            self.wfile.flush()
                        self.wfile.write(b"0\r\n\r\n")
                        self.wfile.flush()
                    except Exception as e:
                        err = json.dumps({"error": str(e)}).encode()
                        self.wfile.write(f"{len(err):x}\r\n".encode() + err + b"\r\n")
                        self.wfile.write(b"0\r\n\r\n")

                elif self.path == "/weights":
                    result = self._orch.get_weights()
                    self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
                    self.wfile.write(json.dumps(result).encode())

                elif self.path == "/set_weight":
                    result = self._orch.set_weight(payload.get("key"), payload.get("value"))
                    self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
                    self.wfile.write(json.dumps(result).encode())

                else:
                    self.send_response(404); self.end_headers()
                    self.wfile.write(json.dumps({"error": "Not found"}).encode())

            def log_message(self, format, *args): pass

        port = self._config.get("headless", {}).get("port", 8765)
        HTTPServer(("0.0.0.0", port), _HTTPHandler).serve_forever()

    def _run_unix_socket(self) -> None:
        import logging
        logger = logging.getLogger(__name__)
        # gap 4 fix: Windows TCP fallback
        if sys.platform == "win32":
            port = self._config.get("headless", {}).get("port", 8765) + 1
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("localhost", port))
        else:
            platform_key = "darwin" if sys.platform == "darwin" else "linux"
            socket_dir = self._config.get("os_paths", {}).get(platform_key, {}).get("socket_dir", "/tmp")
            socket_path = pathlib.Path(socket_dir) / self._config.get("headless", {}).get("unix_socket_name", "aig.sock")
            if socket_path.exists():
                socket_path.unlink()
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.bind(str(socket_path))

        sock.listen(5)
        orch = self._orchestrator

        while self._running:
            try:
                sock.settimeout(1.0)
                try:
                    conn, _ = sock.accept()
                except socket.timeout:
                    continue
                data = b""
                while True:
                    chunk = conn.recv(4096)
                    if not chunk: break
                    data += chunk
                    if b"\n" in data: break
                payload = json.loads(data.decode("utf-8").strip())
                method = payload.get("method", "")
                if method == "infer":
                    result = orch.infer(payload)
                elif method == "get_weights":
                    result = orch.get_weights()
                elif method == "set_weight":
                    result = orch.set_weight(payload["key"], payload["value"])
                elif method == "stream":
                    tokens = [json.loads(t).get("token", t) for t in orch.stream(payload)]
                    result = {"tokens": tokens, "done": True}
                else:
                    result = {"error": f"Unknown method: {method}", "code": 404}
                conn.sendall((json.dumps(result) + "\n").encode("utf-8"))
                conn.close()
            except Exception as e:
                logger.error("unix socket error: %s", e)
