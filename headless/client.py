"""
headless/client.py — gap_04 (Windows TCP fallback) + gap_05 (RFC 7230 chunked parsing)
"""
import json
import urllib.request
import subprocess
import socket
import sys


class HeadlessClient:

    def __init__(self, transport: str, **kwargs):
        self._transport = transport
        self._kwargs = kwargs

    def _post(self, route: str, payload: dict) -> dict:
        host = self._kwargs.get("host", "localhost")
        port = self._kwargs.get("port", 8765)
        url = f"http://{host}:{port}{route}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def infer(self, payload: dict) -> dict:
        if self._transport == "http":
            return self._post("/infer", {"inputs": payload, "stream": False})
        elif self._transport == "stdio":
            return self._stdio_send({"method": "infer", "inputs": payload, "stream": False})
        elif self._transport == "unix_socket":
            return self._unix_send({"method": "infer", "inputs": payload, "stream": False})
        return {}

    def set_weight(self, key: str, value) -> dict:
        if self._transport == "http":
            return self._post("/set_weight", {"key": key, "value": value})
        elif self._transport == "stdio":
            return self._stdio_send({"method": "set_weight", "key": key, "value": value})
        elif self._transport == "unix_socket":
            return self._unix_send({"method": "set_weight", "key": key, "value": value})
        return {}

    def get_weights(self) -> list:
        if self._transport == "http":
            return self._post("/weights", {})
        elif self._transport == "stdio":
            return self._stdio_send({"method": "get_weights"})
        elif self._transport == "unix_socket":
            return self._unix_send({"method": "get_weights"})
        return []

    def stream(self, payload: dict):
        if self._transport == "http":
            yield from self._http_stream(payload)
        elif self._transport == "stdio":
            yield from self._stdio_stream(payload)
        elif self._transport == "unix_socket":
            yield from self._unix_stream(payload)

    def _http_stream(self, payload: dict):
        """gap_05: Parse RFC 7230 chunked transfer encoding correctly."""
        host = self._kwargs.get("host", "localhost")
        port = self._kwargs.get("port", 8765)
        url = f"http://{host}:{port}/stream"
        data = json.dumps({"inputs": payload, "stream": True}).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

        with urllib.request.urlopen(req) as resp:
            while True:
                # Read chunk size line (hex)
                size_line = b""
                while not size_line.endswith(b"\r\n"):
                    byte = resp.read(1)
                    if not byte:
                        return
                    size_line += byte

                size_str = size_line.strip().decode("utf-8")
                if not size_str:
                    continue

                # Handle chunk extensions (size;ext=val) — strip extension
                size_str = size_str.split(";")[0].strip()
                chunk_size = int(size_str, 16)

                # Terminal chunk
                if chunk_size == 0:
                    # Consume trailing CRLF after terminal chunk
                    resp.read(2)
                    return

                # Read exactly chunk_size bytes
                chunk_data = b""
                remaining = chunk_size
                while remaining > 0:
                    part = resp.read(remaining)
                    if not part:
                        return
                    chunk_data += part
                    remaining -= len(part)

                # Consume trailing CRLF after chunk data
                resp.read(2)

                try:
                    token_data = json.loads(chunk_data.decode("utf-8"))
                    if "token" in token_data:
                        yield token_data["token"]
                except json.JSONDecodeError:
                    pass

    def _stdio_send(self, payload: dict) -> dict:
        cmd = self._kwargs.get("cmd", [])
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        proc.stdin.write((json.dumps(payload) + "\n").encode())
        proc.stdin.flush()
        line = proc.stdout.readline()
        return json.loads(line.decode("utf-8"))

    def _stdio_stream(self, payload: dict):
        cmd = self._kwargs.get("cmd", [])
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        request = json.dumps({"method": "stream", "inputs": payload, "stream": True}) + "\n"
        proc.stdin.write(request.encode())
        proc.stdin.flush()
        for line in proc.stdout:
            data = json.loads(line.decode("utf-8"))
            if data.get("done"):
                break
            if "token" in data:
                yield data["token"]

    def _make_unix_socket(self):
        """gap_04: Use TCP fallback on Windows instead of AF_UNIX."""
        if sys.platform == "win32":
            port = self._kwargs.get("port", 8765) + 1
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(("localhost", port))
        else:
            socket_path = self._kwargs.get("socket_path", "/tmp/aig.sock")
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.connect(socket_path)
        return s

    def _unix_send(self, payload: dict) -> dict:
        s = self._make_unix_socket()
        s.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        data = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break
        s.close()
        return json.loads(data.decode("utf-8").strip())

    def _unix_stream(self, payload: dict):
        s = self._make_unix_socket()
        s.sendall((json.dumps({"method": "stream", "inputs": payload, "stream": True}) + "\n").encode("utf-8"))
        buf = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                data = json.loads(line.decode("utf-8"))
                if data.get("done"):
                    s.close()
                    return
                if "token" in data:
                    yield data["token"]
        s.close()
