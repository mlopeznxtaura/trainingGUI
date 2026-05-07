"""
frameworks/onnx_adapter.py — ONNXAdapter
"""
import time
import os
import logging
from frameworks.base import BaseFrameworkAdapter

logger = logging.getLogger(__name__)


class ONNXAdapter(BaseFrameworkAdapter):

    FRAMEWORK_KEY = "onnx"

    def load_model(self, path: str, device: str) -> None:
        import onnxruntime as ort

        expanded = os.path.expanduser(path)
        self._model_path = expanded
        self._weight_overrides = {}

        available = ort.get_available_providers()
        if device == "auto":
            if "CUDAExecutionProvider" in available:
                provider = "CUDAExecutionProvider"
                self._active_device = "cuda"
            elif "CoreMLExecutionProvider" in available:
                provider = "CoreMLExecutionProvider"
                self._active_device = "mps"
            else:
                provider = "CPUExecutionProvider"
                self._active_device = "cpu"
        else:
            provider = "CPUExecutionProvider"
            self._active_device = device

        self._session = ort.InferenceSession(expanded, providers=[provider])

    def run_inference(self, inputs: dict) -> dict:
        import numpy as np

        t0 = time.perf_counter()
        input_names = [i.name for i in self._session.get_inputs()]
        feed = {}
        for name in input_names:
            if name in inputs:
                feed[name] = np.array(inputs[name])
            elif name in self._weight_overrides:
                feed[name] = np.array(self._weight_overrides[name])
        outputs = self._session.run(None, feed)
        latency = (time.perf_counter() - t0) * 1000
        return {"output": outputs, "latency_ms": latency}

    def get_weight_keys(self) -> list:
        input_names = [i.name for i in self._session.get_inputs()]
        output_names = [o.name for o in self._session.get_outputs()]
        return input_names + output_names + list(self._weight_overrides.keys())

    def set_weight(self, key: str, value) -> None:
        logger.warning("ONNXRuntime does not support in-place weight mutation. Storing as inference hint: %s", key)
        self._weight_overrides[key] = value

    def stream_tokens(self, inputs: dict):
        yield str(self.run_inference(inputs)["output"])

    def shutdown(self) -> None:
        del self._session
        self._session = None
