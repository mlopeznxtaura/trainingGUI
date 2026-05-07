"""
core/session.py — Stateful inference session
"""
import threading
from datetime import datetime


class InferenceSession:

    def __init__(self, adapter, config: dict):
        self.adapter = adapter
        self.config = config
        self.history = []
        self._lock = threading.Lock()
        self._callbacks = {
            "on_output": [],
            "on_token": [],
            "on_error": [],
        }
        self.stream_mode = config.get("output", {}).get("stream", True)

    def run(self, inputs: dict) -> dict:
        try:
            result = self.adapter.run_inference(inputs)
        except Exception as e:
            error = {"error": str(e)}
            for fn in self._callbacks["on_error"]:
                try:
                    fn(error)
                except Exception:
                    pass
            return error

        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "inputs": inputs,
            "output": result.get("output"),
            "latency_ms": result.get("latency_ms", 0.0),
        }

        with self._lock:
            self.history.append(record)

        for fn in self._callbacks["on_output"]:
            try:
                fn(result)
            except Exception:
                pass

        return result

    def stream(self, inputs: dict):
        tokens = []
        try:
            gen = self.adapter.stream_tokens(inputs)
            for token in gen:
                tokens.append(token)
                for fn in self._callbacks["on_token"]:
                    try:
                        fn(token)
                    except Exception:
                        pass
                yield token
        except Exception as e:
            error_tok = str(e)
            for fn in self._callbacks["on_error"]:
                try:
                    fn({"error": error_tok})
                except Exception:
                    pass
            yield error_tok

        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "inputs": inputs,
            "output": "".join(tokens),
            "latency_ms": 0.0,
            "type": "stream_summary",
        }
        with self._lock:
            self.history.append(record)

    def apply_weight(self, key: str, value) -> None:
        self.adapter.set_weight(key, value)
        with self._lock:
            self.history.append({
                "type": "weight_set",
                "key": key,
                "value": str(value),
                "timestamp": datetime.utcnow().isoformat(),
            })

    def register_callback(self, event: str, fn) -> None:
        if event not in self._callbacks:
            raise ValueError(f"Unknown event '{event}'. Valid events: {list(self._callbacks.keys())}")
        self._callbacks[event].append(fn)

    def get_history(self) -> list:
        return list(self.history)
