"""
frameworks/jax_adapter.py — JAXAdapter with immutable pytree set_weight (gap_03)
"""
import time, os
from frameworks.base import BaseFrameworkAdapter


class JAXAdapter(BaseFrameworkAdapter):
    FRAMEWORK_KEY = "jax"

    def load_model(self, path: str, device: str) -> None:
        import jax
        expanded = os.path.expanduser(path)
        self._model_path = expanded
        devices = jax.devices()
        if device == "auto":
            preferred = [d for d in devices if d.platform in ("gpu", "tpu")]
            self._active_device = preferred[0].platform if preferred else "cpu"
        else:
            self._active_device = device
        try:
            import orbax.checkpoint as ocp
            self._params = ocp.PyTreeCheckpointer().restore(expanded)
        except Exception:
            try:
                import pickle
                with open(expanded, "rb") as f:
                    data = pickle.load(f)
                self._params = data.get("params", data)
            except Exception:
                self._params = {}
        self._original_params = dict(self._params) if isinstance(self._params, dict) else self._params
        self._apply_fn = lambda params, inputs: {"output": str(params)}

    def run_inference(self, inputs: dict) -> dict:
        t0 = time.perf_counter()
        output = self._apply_fn(self._params, inputs)
        return {"output": output, "latency_ms": (time.perf_counter() - t0) * 1000}

    def get_weight_keys(self) -> list:
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
        """gap_03: Rebuild frozen pytree — never mutates original."""
        def _set_nested(node, keys, val):
            if not keys:
                return val
            k = keys[0]
            if isinstance(node, dict):
                new = dict(node)
                new[k] = _set_nested(node.get(k, {}), keys[1:], val)
                return new
            return val
        self._params = _set_nested(self._params, key.split("."), value)

    def stream_tokens(self, inputs: dict):
        yield str(self.run_inference(inputs)["output"])

    def shutdown(self) -> None:
        self._params = None
        self._apply_fn = None
