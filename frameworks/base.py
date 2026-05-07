"""
frameworks/base.py — Abstract base class for all framework adapters
"""
import time
from abc import ABC, abstractmethod


class BaseFrameworkAdapter(ABC):

    FRAMEWORK_KEY: str = ""

    def __init__(self):
        self._active_device: str = "cpu"
        self._model_path: str = ""

    @abstractmethod
    def load_model(self, path: str, device: str) -> None:
        """Expand ~ via os.path.expanduser. Resolve 'auto' device using framework-native API."""
        pass

    @abstractmethod
    def run_inference(self, inputs: dict) -> dict:
        """Measure latency with time.perf_counter(). Return at minimum: {output, latency_ms}."""
        pass

    @abstractmethod
    def get_weight_keys(self) -> list:
        """Walk model parameter/variable tree. Return dot-notation strings."""
        pass

    @abstractmethod
    def set_weight(self, key: str, value) -> None:
        """Resolve dot-notation key to nested attribute. Apply value in-place."""
        pass

    @abstractmethod
    def stream_tokens(self, inputs: dict):
        """Yield string tokens one at a time."""
        pass

    def get_metadata(self) -> dict:
        return {
            "framework": self.FRAMEWORK_KEY,
            "device": self._active_device,
            "model_path": self._model_path,
            "adapter_class": self.__class__.__name__,
        }

    def shutdown(self) -> None:
        pass
