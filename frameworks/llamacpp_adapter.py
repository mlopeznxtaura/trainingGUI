"""
frameworks/llamacpp_adapter.py — LlamaCppAdapter
"""
import time
import os
import logging
from frameworks.base import BaseFrameworkAdapter

logger = logging.getLogger(__name__)


class LlamaCppAdapter(BaseFrameworkAdapter):

    FRAMEWORK_KEY = "llamacpp"

    def load_model(self, path: str, device: str) -> None:
        from llama_cpp import Llama

        expanded = os.path.expanduser(path)
        self._model_path = expanded
        self._sampler_params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 512,
        }

        if device == "auto":
            n_gpu_layers = -1  # full GPU offload, llama.cpp falls back to CPU if unavailable
            self._active_device = "gpu"
        else:
            n_gpu_layers = 0
            self._active_device = device

        self._model = Llama(
            model_path=expanded,
            n_gpu_layers=n_gpu_layers,
            n_ctx=4096,
        )

    def run_inference(self, inputs: dict) -> dict:
        t0 = time.perf_counter()
        prompt = inputs.get("prompt", "")
        result = self._model(prompt, stream=False, **self._sampler_params)
        latency = (time.perf_counter() - t0) * 1000
        return {"output": result["choices"][0]["text"], "latency_ms": latency}

    def get_weight_keys(self) -> list:
        return list(self._sampler_params.keys())

    def set_weight(self, key: str, value) -> None:
        if key in self._sampler_params:
            self._sampler_params[key] = value
        else:
            logger.warning("llama.cpp does not expose raw tensor weights at runtime. Unknown key: %s", key)

    def stream_tokens(self, inputs: dict):
        prompt = inputs.get("prompt", "")
        generator = self._model(prompt, stream=True, **self._sampler_params)
        for chunk in generator:
            token = chunk["choices"][0]["text"]
            yield token

    def shutdown(self) -> None:
        del self._model
        self._model = None
