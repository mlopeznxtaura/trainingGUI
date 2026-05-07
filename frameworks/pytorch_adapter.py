"""
frameworks/pytorch_adapter.py — PyTorchAdapter
"""
import time
import os
from frameworks.base import BaseFrameworkAdapter


class PyTorchAdapter(BaseFrameworkAdapter):

    FRAMEWORK_KEY = "pytorch"

    def load_model(self, path: str, device: str) -> None:
        import torch

        expanded = os.path.expanduser(path)
        self._model_path = expanded

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self._active_device = device
        self._device_obj = torch.device(device)

        try:
            self._model = torch.load(expanded, map_location=self._device_obj)
        except Exception:
            # Fallback: try as state dict or huggingface model
            try:
                from transformers import AutoModel
                self._model = AutoModel.from_pretrained(expanded)
                self._model = self._model.to(self._device_obj)
            except Exception:
                self._model = None

        if self._model is not None and hasattr(self._model, "eval"):
            self._model.eval()

    def run_inference(self, inputs: dict) -> dict:
        import torch

        t0 = time.perf_counter()
        with torch.no_grad():
            if self._model is None:
                result = {"error": "model not loaded"}
            else:
                try:
                    result = self._model(inputs)
                except Exception as e:
                    result = {"error": str(e)}
        latency = (time.perf_counter() - t0) * 1000
        return {"output": result, "latency_ms": latency}

    def get_weight_keys(self) -> list:
        if self._model is None:
            return []
        return [name for name, _ in self._model.named_parameters()]

    def set_weight(self, key: str, value) -> None:
        import torch

        if self._model is None:
            return
        parts = key.split(".")
        obj = self._model
        for part in parts[:-1]:
            obj = getattr(obj, part)
        param = getattr(obj, parts[-1])
        with torch.no_grad():
            if hasattr(param, "data"):
                param.data.copy_(torch.tensor(value))
            else:
                setattr(obj, parts[-1], value)

    def stream_tokens(self, inputs: dict):
        if self._model is None:
            yield str(self.run_inference(inputs)["output"])
            return

        if hasattr(self._model, "generate"):
            try:
                from transformers import TextIteratorStreamer
                import threading

                streamer = TextIteratorStreamer(self._model.config.tokenizer if hasattr(self._model, "config") else None, skip_prompt=True)
                thread = threading.Thread(target=self._model.generate, kwargs={"streamer": streamer, **inputs})
                thread.start()
                for token in streamer:
                    yield token
                thread.join()
                return
            except Exception:
                pass

        yield str(self.run_inference(inputs)["output"])

    def shutdown(self) -> None:
        import torch

        del self._model
        self._model = None
        if self._active_device == "cuda":
            torch.cuda.empty_cache()
