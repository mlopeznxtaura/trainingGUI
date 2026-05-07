"""
frameworks/pytorch_adapter.py — PyTorchAdapter (gap 9 fixed: safe tokenizer in stream_tokens)
"""
import time, os, logging
from frameworks.base import BaseFrameworkAdapter
logger = logging.getLogger(__name__)


class PyTorchAdapter(BaseFrameworkAdapter):
    FRAMEWORK_KEY = "pytorch"

    def load_model(self, path: str, device: str) -> None:
        import torch
        expanded = os.path.expanduser(path)
        self._model_path = expanded
        if device == "auto":
            if torch.cuda.is_available(): device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): device = "mps"
            else: device = "cpu"
        self._active_device = device
        self._device_obj = torch.device(device)
        try:
            # gap 24 fix: weights_only=True avoids arbitrary code execution
            self._model = torch.load(expanded, map_location=self._device_obj, weights_only=True)
        except Exception:
            try:
                from transformers import AutoModel
                self._model = AutoModel.from_pretrained(expanded).to(self._device_obj)
            except Exception:
                self._model = None
        if self._model is not None and hasattr(self._model, "eval"):
            self._model.eval()

    def run_inference(self, inputs: dict) -> dict:
        import torch
        t0 = time.perf_counter()
        with torch.no_grad():
            try:
                result = self._model(inputs) if self._model else {"error": "model not loaded"}
            except Exception as e:
                result = {"error": str(e)}
        return {"output": result, "latency_ms": (time.perf_counter() - t0) * 1000}

    def get_weight_keys(self) -> list:
        return [name for name, _ in self._model.named_parameters()] if self._model else []

    def set_weight(self, key: str, value) -> None:
        import torch
        if not self._model: return
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
        """gap 9 fix: safely load tokenizer before using TextIteratorStreamer."""
        if self._model is not None and hasattr(self._model, "generate"):
            try:
                from transformers import AutoTokenizer, TextIteratorStreamer
                import threading
                tokenizer = AutoTokenizer.from_pretrained(self._model_path)
                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                gen_kwargs = {"streamer": streamer}
                if isinstance(inputs, dict) and "input_ids" in inputs:
                    gen_kwargs.update(inputs)
                thread = threading.Thread(target=self._model.generate, kwargs=gen_kwargs)
                thread.start()
                for token in streamer:
                    yield token
                thread.join()
                return
            except Exception as e:
                logger.warning("TextIteratorStreamer unavailable: %s. Falling back to batch.", e)
        yield str(self.run_inference(inputs)["output"])

    def shutdown(self) -> None:
        import torch
        if self._model: del self._model
        self._model = None
        if self._active_device == "cuda":
            torch.cuda.empty_cache()
