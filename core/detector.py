"""
core/detector.py — Framework auto-detection
"""
import importlib.util
import ctypes
import pathlib
import sys
import os
import glob


def _current_platform() -> str:
    if sys.platform.startswith("linux"):
        return "linux"
    elif sys.platform == "darwin":
        return "darwin"
    else:
        return "windows"


class FrameworkDetector:

    def __init__(self, config: dict):
        self._config = config

    def detect(self) -> str:
        cfg = self._config

        if cfg.get("framework", "auto") != "auto":
            return cfg["framework"]

        detected = []

        probe_map = {
            "torch": "pytorch",
            "tensorflow": "tensorflow",
            "jax": "jax",
            "onnxruntime": "onnx",
            "llama_cpp": "llamacpp",
        }

        for module_name, key in probe_map.items():
            if importlib.util.find_spec(module_name) is not None:
                detected.append(key)

        # Binary probe for llamacpp if not already found via python binding
        if "llamacpp" not in detected:
            platform = _current_platform()
            os_paths = cfg.get("os_paths", {})
            lib_search = os_paths.get(platform, {}).get("lib_search", [])
            patterns = ["libllama.so", "libllama.dylib", "llama.dll"]
            for search_dir in lib_search:
                expanded = os.path.expanduser(search_dir)
                for pattern in patterns:
                    matches = glob.glob(os.path.join(expanded, "**", pattern), recursive=True)
                    if matches:
                        detected.append("llamacpp")
                        break
                if "llamacpp" in detected:
                    break

        # Probe custom_cpp
        custom_path = cfg.get("custom_engine_path")
        if custom_path:
            try:
                expanded = os.path.expanduser(custom_path)
                ctypes.CDLL(expanded)
                detected.append("custom_cpp")
            except Exception:
                pass

        if not detected:
            probed = list(probe_map.values()) + ["llamacpp (binary)", "custom_cpp"]
            raise RuntimeError(
                f"No supported inference frameworks found. Probed: {', '.join(probed)}. "
                "Install at least one: torch, tensorflow, jax, onnxruntime, llama_cpp, or custom_cpp."
            )

        priority = cfg.get("framework_priority", [])
        if priority:
            for p in priority:
                if p in detected:
                    return p

        return detected[0]
