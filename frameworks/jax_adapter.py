"""
frameworks/jax_adapter.py — JAXAdapter
"""
import time
import os
from frameworks.base import BaseFrameworkAdapter


class JAXAdapter(BaseFrameworkAdapter):

    FRAMEWORK_KEY = "jax"

    def load_model(self, path: str, device: str) -> None:
        import jax
        import jax.numpy as jnp

        expanded = os.path.expanduser(path)
        self._model_path = expanded

        devices = jax.devices()
        if device == "auto":
            preferred = [d for d in devices if d.platform in ("gpu", "tpu")]
            self._active_device = preferred[0].platform if preferred else "cpu"
        else:
            self._active_device = device

        # Try orbax first, then pickle fallback
        try:
            import orbax.checkpoint as ocp
            checkpointer = ocp.PyTreeCheckpointer()
            self._params = checkpointer.restore(expanded)
        except Exception:
            import pickle
            with open(expanded, "rb") as f:
                data = pickle.load(f)
            self._params = data.get("params", data)

        # Try to load apply_fn from flax model if available
        try:
            import flax.linen as nn
            self._apply_fn = lambda params, inputs: params  # placeholder
        except Exception:
            self._apply_fn = lambda params, inputs: {"output": str(params)}

    def run_inference(self, inputs: dict) -> dict:
        t0 = time.perf_counter()
        output = self._apply_fn(self._params, inputs)
        latency = (time.perf_counter() - t0) * 1000
        return {"output": output, "latency_ms": latency}

    def get_weight_keys(self) -> list:
        import jax

        keys = []

        def _walk(node, prefix=""):
            if isinstance(node, dict):
                for k, v in node.items():
                    _walk(v, f"{prefix}.{k}" if prefix else k)
            else:
                keys.append(prefix)

        _walk(self._params)
        return keys

    def set_weight(self, key: str, value) -> None:
        """Rebuild immutable pytree with updated leaf."""
        parts = key.split(".")

        def _set_nested(d, keys, val):
            if len(keys) == 1:
                new = dict(d)
                new[keys[0]] = val
                return new
            new = dict(d)
            new[keys[0]] = _set_nested(d[keys[0]], keys[1:], val)
            return new

        if isinstance(self._params, dict):
            self._params = _set_nested(self._params, parts, value)

    def stream_tokens(self, inputs: dict):
        yield str(self.run_inference(inputs)["output"])

    def shutdown(self) -> None:
        del self._params
        del self._apply_fn
        self._params = None
        self._apply_fn = None
