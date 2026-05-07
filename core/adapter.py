"""
core/adapter.py — Adapter factory
"""
import importlib


class AdapterLoadError(Exception):
    def __init__(self, framework_key: str, path: str, cause: Exception = None):
        self.framework_key = framework_key
        self.path = path
        super().__init__(f"Failed to load adapter for '{framework_key}' from path '{path}': {cause}")
        if cause:
            self.__cause__ = cause


class AdapterFactory:

    _registry = {
        "pytorch":      "frameworks.pytorch_adapter.PyTorchAdapter",
        "tensorflow":   "frameworks.tensorflow_adapter.TensorFlowAdapter",
        "jax":          "frameworks.jax_adapter.JAXAdapter",
        "onnx":         "frameworks.onnx_adapter.ONNXAdapter",
        "llamacpp":     "frameworks.llamacpp_adapter.LlamaCppAdapter",
        "custom_cpp":   "frameworks.custom_cpp_adapter.CustomCppAdapter",
        "llm_provider": "frameworks.llm_provider_adapter.LLMProviderAdapter",
    }

    @staticmethod
    def build(framework_key: str, config: dict):
        if framework_key not in AdapterFactory._registry:
            raise AdapterLoadError(framework_key, config.get("model_path", ""), KeyError(framework_key))

        class_path = AdapterFactory._registry[framework_key]
        module_path, class_name = class_path.rsplit(".", 1)

        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            instance = cls()
            instance.load_model(config["model_path"], config.get("device", "auto"))
            # gap 2 fix: call configure() for adapters that need post-load config (e.g. LLMProviderAdapter)
            if hasattr(instance, "configure"):
                instance.configure(config.get("headless_llm", {}))
            return instance
        except Exception as e:
            raise AdapterLoadError(framework_key, config.get("model_path", ""), e) from e
