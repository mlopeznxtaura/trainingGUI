"""
headless/client.py — Reference client for all transports
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
            cmd = self._kwargs.get("cmd", [])
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            request = json.dumps({"method": "infer", "inputs": payload, "stream": False}) + "\n"
            proc.stdin.write(request.encode())
            proc.stdin.flush()
            line = proc.stdout.readline()
            return json.loads(line.decode("utf-8"))

        elif self._transport == "unix_socket":
            return self._unix_send({"method": "infer", "inputs": payload, "stream": False})

        return {}

    def set_weight(self, key: str, value) -> dict:
        if self._transport == "http":
            return self._post("/set_weight", {"key": key, "value": value})

        elif self._transport == "stdio":
            cmd = self._kwargs.get("cmd", [])
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            request = json.dumps({"method": "set_weight", "key": key, "value": value}) + "\n"
            proc.stdin.write(request.encode())
            proc.stdin.flush()
            line = proc.stdout.readline()
            return json.loads(line.decode("utf-8"))

        elif self._transport == "unix_socket":
            return self._unix_send({"method": "set_weight", "key": key, "value": value})

        return {}

    def get_weights(self) -> list:
        if self._transport == "http":
            return self._post("/weights", {})

        elif self._transport == "stdio":
            cmd = self._kwargs.get("cmd", [])
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            request = json.dumps({"method": "get_weights"}) + "\n"
            proc.stdin.write(request.encode())
            proc.stdin.flush()
            line = proc.stdout.readline()
            return json.loads(line.decode("utf-8"))

        elif self._transport == "unix_socket":
            return self._unix_send({"method": "get_weights"})

        return []

    def stream(self, payload: dict):
        if self._transport == "http":
            host = self._kwargs.get("host", "localhost")
            port = self._kwargs.get("port", 8765)
            url = f"http://{host}:{port}/stream"
            data = json.dumps({"inputs": payload, "stream": True}).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req) as resp:
                while True:
                    size_line = resp.readline().decode("utf-8").strip()
                    if not size_line or size_line == "0":
                        break
                    size = int(size_line, 16)
                    if size == 0:
                        break
                    chunk = resp.read(size)
                    resp.readline()  # consume trailing CRLF
                    token_data = json.loads(chunk.decode("utf-8"))
                    if "token" in token_data:
                        yield token_data["token"]

        elif self._transport == "stdio":
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

        elif self._transport == "unix_socket":
            sock = self._make_unix_socket()
            request = json.dumps({"method": "stream", "inputs": payload, "stream": True}) + "\n"
            sock.sendall(request.encode("utf-8"))
            buf = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    data = json.loads(line.decode("utf-8"))
                    if data.get("done"):
                        sock.close()
                        return
                    if "token" in data:
                        yield data["token"]
            sock.close()

    def _make_unix_socket(self):
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
        request = json.dumps(payload) + "\n"
        s.sendall(request.encode("utf-8"))
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
