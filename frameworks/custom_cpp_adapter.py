"""
frameworks/custom_cpp_adapter.py — CustomCppAdapter
"""
import time
import os
import logging
from frameworks.base import BaseFrameworkAdapter

logger = logging.getLogger(__name__)


class CustomCppAdapter(BaseFrameworkAdapter):

    FRAMEWORK_KEY = "custom_cpp"

    def load_model(self, path: str, device: str) -> None:
        import ctypes
        import json
        import pathlib

        expanded = os.path.expanduser(path)
        self._model_path = expanded
        self._active_device = device

        self._lib = ctypes.CDLL(expanded)

        manifest_path = pathlib.Path(expanded).parent / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
        else:
            manifest = {"functions": [], "weight_keys": [], "input_format": "json_string"}

        self._weight_keys = manifest.get("weight_keys", [])
        self._input_format = manifest.get("input_format", "json_string")

        ctypes_map = {
            "c_float": ctypes.c_float,
            "c_double": ctypes.c_double,
            "c_int": ctypes.c_int,
            "c_char_p": ctypes.c_char_p,
            "c_void_p": ctypes.c_void_p,
            "c_bool": ctypes.c_bool,
        }

        for fn in manifest.get("functions", []):
            name = fn["name"]
            if hasattr(self._lib, name):
                func = getattr(self._lib, name)
                argtypes = [ctypes_map.get(a, ctypes.c_void_p) for a in fn.get("argtypes", [])]
                restype = ctypes_map.get(fn.get("restype", "c_char_p"), ctypes.c_char_p)
                func.argtypes = argtypes
                func.restype = restype

    def run_inference(self, inputs: dict) -> dict:
        import ctypes
        import json

        t0 = time.perf_counter()
        if self._input_format == "json_string":
            serialized = json.dumps(inputs).encode("utf-8")
        else:
            serialized = str(inputs).encode("utf-8")

        raw = self._lib.run_inference(serialized)
        if isinstance(raw, bytes):
            result = json.loads(raw.decode("utf-8"))
        else:
            result = raw

        latency = (time.perf_counter() - t0) * 1000
        return {"output": result, "latency_ms": latency}

    def get_weight_keys(self) -> list:
        if hasattr(self._lib, "get_weight_names"):
            raw = self._lib.get_weight_names()
            if raw:
                decoded = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                return [k.strip() for k in decoded.split("\n") if k.strip()]
        return list(self._weight_keys)

    def set_weight(self, key: str, value) -> None:
        import json
        import ctypes

        if hasattr(self._lib, "set_weight"):
            serialized = json.dumps(value).encode("utf-8")
            self._lib.set_weight(key.encode("utf-8"), serialized)
        else:
            logger.warning("No set_weight symbol in binary. Cannot set weight: %s", key)

    def stream_tokens(self, inputs: dict):
        if hasattr(self._lib, "stream_tokens"):
            import json
            import ctypes

            serialized = json.dumps(inputs).encode("utf-8")
            sentinel = b"__DONE__"
            while True:
                raw = self._lib.stream_tokens(serialized)
                if raw is None or raw == sentinel:
                    break
                token = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                yield token
        else:
            yield str(self.run_inference(inputs)["output"])

    def shutdown(self) -> None:
        if hasattr(self._lib, "shutdown"):
            self._lib.shutdown()
        del self._lib
        self._lib = None
