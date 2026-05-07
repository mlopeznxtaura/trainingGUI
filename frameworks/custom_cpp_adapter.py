"""
frameworks/custom_cpp_adapter.py — gap_06: Full manifest schema + dynamic ctypes loader
"""
import time, os, logging
from frameworks.base import BaseFrameworkAdapter
logger = logging.getLogger(__name__)


class CustomCppAdapter(BaseFrameworkAdapter):
    FRAMEWORK_KEY = "custom_cpp"

    def _resolve_ctype(self, type_str: str):
        import ctypes
        return {
            "c_float": ctypes.c_float, "c_double": ctypes.c_double, "c_int": ctypes.c_int,
            "c_long": ctypes.c_long, "c_char_p": ctypes.c_char_p, "c_void_p": ctypes.c_void_p,
            "c_bool": ctypes.c_bool, "c_uint": ctypes.c_uint,
        }.get(type_str)

    def load_model(self, path: str, device: str) -> None:
        import ctypes, json, pathlib
        expanded = os.path.expanduser(path)
        self._model_path = expanded
        self._active_device = device
        self._lib = ctypes.CDLL(expanded)

        manifest_path = pathlib.Path(expanded).parent / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
        else:
            logger.warning("No manifest.json found at %s", expanded)
            manifest = {"functions": [], "weight_keys": [], "input_format": "json_string", "output_format": "json_string"}

        self._weight_keys = manifest.get("weight_keys", [])
        self._input_format = manifest.get("input_format", "json_string")
        self._output_format = manifest.get("output_format", "json_string")

        for fn in manifest.get("functions", []):
            name = fn.get("name", "")
            if not name or not hasattr(self._lib, name):
                continue
            func = getattr(self._lib, name)
            func.argtypes = [self._resolve_ctype(a) or __import__("ctypes").c_void_p for a in fn.get("argtypes", [])]
            func.restype = self._resolve_ctype(fn.get("restype", "c_char_p"))

    def run_inference(self, inputs: dict) -> dict:
        import json
        t0 = time.perf_counter()
        serialized = json.dumps(inputs).encode("utf-8") if self._input_format == "json_string" else str(inputs).encode()
        raw = self._lib.run_inference(serialized)
        try:
            result = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else str(raw))
        except Exception:
            result = raw
        return {"output": result, "latency_ms": (time.perf_counter() - t0) * 1000}

    def get_weight_keys(self) -> list:
        """gap_06 task_03: binary symbol first, manifest fallback."""
        if hasattr(self._lib, "get_weight_names"):
            try:
                raw = self._lib.get_weight_names()
                if raw:
                    keys = [k.strip() for k in (raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)).split("\n") if k.strip()]
                    if keys:
                        return keys
            except Exception as e:
                logger.warning("get_weight_names() failed: %s", e)
        return list(self._weight_keys)

    def set_weight(self, key: str, value) -> None:
        import json
        if hasattr(self._lib, "set_weight"):
            self._lib.set_weight(key.encode("utf-8"), json.dumps(value).encode("utf-8"))
        else:
            logger.warning("No set_weight symbol in binary for key: %s", key)

    def stream_tokens(self, inputs: dict):
        if hasattr(self._lib, "stream_tokens"):
            import json
            serialized = json.dumps(inputs).encode("utf-8")
            while True:
                raw = self._lib.stream_tokens(serialized)
                if not raw or raw in (b"__DONE__",):
                    break
                token = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                if token == "__DONE__":
                    break
                yield token
        else:
            yield str(self.run_inference(inputs)["output"])

    def shutdown(self) -> None:
        if hasattr(self._lib, "shutdown"):
            try: self._lib.shutdown()
            except Exception: pass
        self._lib = None
