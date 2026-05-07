"""
frameworks/tensorflow_adapter.py — TensorFlowAdapter
"""
import time
import os
from frameworks.base import BaseFrameworkAdapter


class TensorFlowAdapter(BaseFrameworkAdapter):

    FRAMEWORK_KEY = "tensorflow"

    def load_model(self, path: str, device: str) -> None:
        import tensorflow as tf

        expanded = os.path.expanduser(path)
        self._model_path = expanded

        gpus = tf.config.list_physical_devices("GPU")
        if device == "auto":
            device = "/GPU:0" if gpus else "/CPU:0"
        self._active_device = device

        with tf.device(device):
            try:
                self._model = tf.saved_model.load(expanded)
            except Exception:
                self._model = tf.keras.models.load_model(expanded)

    def run_inference(self, inputs: dict) -> dict:
        import tensorflow as tf

        t0 = time.perf_counter()
        try:
            result = self._model(inputs)
            output = result.numpy() if hasattr(result, "numpy") else result
        except Exception:
            result = self._model.predict(inputs)
            output = result
        latency = (time.perf_counter() - t0) * 1000
        return {"output": output, "latency_ms": latency}

    def get_weight_keys(self) -> list:
        if hasattr(self._model, "trainable_variables"):
            return [v.name for v in self._model.trainable_variables]
        return []

    def set_weight(self, key: str, value) -> None:
        if not hasattr(self._model, "trainable_variables"):
            return
        for var in self._model.trainable_variables:
            if var.name == key:
                var.assign(value)
                return

    def stream_tokens(self, inputs: dict):
        yield str(self.run_inference(inputs)["output"])

    def shutdown(self) -> None:
        del self._model
        self._model = None
