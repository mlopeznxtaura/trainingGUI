"""
frameworks/llm_provider_adapter.py — gap_07: Provider-agnostic LLM adapter
Supports Ollama, llama-server, any OpenAI-compatible local endpoint.

Add to config.json:
  "headless_llm": {
    "enabled": true,
    "base_url": "http://localhost:11434",
    "model": "llama3",
    "api_key_env": ""
  }
"""
import time, os, json
import urllib.request, urllib.error
from frameworks.base import BaseFrameworkAdapter


class LLMProviderAdapter(BaseFrameworkAdapter):
    FRAMEWORK_KEY = "llm_provider"

    def load_model(self, path: str, device: str) -> None:
        self._model_path = path
        self._active_device = device
        self._base_url = ""
        self._model = ""
        self._api_key = None
        self._sampler_params = {"temperature": 0.7, "top_p": 0.9, "max_tokens": 512}

    def configure(self, llm_config: dict) -> None:
        self._base_url = llm_config.get("base_url", "http://localhost:11434").rstrip("/")
        self._model = llm_config.get("model", "llama3")
        env_key = llm_config.get("api_key_env", "")
        self._api_key = os.environ.get(env_key, "") if env_key else llm_config.get("api_key", "")

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    def _post(self, endpoint: str, payload: dict) -> dict:
        url = f"{self._base_url}{endpoint}"
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=self._headers(), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            return {"error": str(e), "code": e.code}

    def run_inference(self, inputs: dict) -> dict:
        prompt = inputs.get("prompt", inputs.get("messages", ""))
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": str(prompt)}]
        t0 = time.perf_counter()

        result = self._post("/v1/chat/completions", {
            "model": self._model, "messages": messages, "stream": False,
            "temperature": self._sampler_params["temperature"],
            "top_p": self._sampler_params["top_p"],
            "max_tokens": self._sampler_params["max_tokens"],
        })

        if "choices" in result:
            output = result["choices"][0]["message"]["content"]
        else:
            ollama = self._post("/api/generate", {
                "model": self._model, "prompt": str(prompt), "stream": False,
                "options": {"temperature": self._sampler_params["temperature"],
                            "top_p": self._sampler_params["top_p"],
                            "num_predict": self._sampler_params["max_tokens"]},
            })
            output = ollama.get("response", ollama.get("error", str(ollama)))

        return {"output": output, "latency_ms": (time.perf_counter() - t0) * 1000}

    def get_weight_keys(self) -> list:
        return list(self._sampler_params.keys())

    def set_weight(self, key: str, value) -> None:
        if key in self._sampler_params:
            self._sampler_params[key] = value

    def stream_tokens(self, inputs: dict):
        prompt = inputs.get("prompt", inputs.get("messages", ""))
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": str(prompt)}]
        payload = {"model": self._model, "messages": messages, "stream": True,
                   "temperature": self._sampler_params["temperature"],
                   "top_p": self._sampler_params["top_p"],
                   "max_tokens": self._sampler_params["max_tokens"]}
        try:
            url = f"{self._base_url}/v1/chat/completions"
            req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=self._headers(), method="POST")
            with urllib.request.urlopen(req, timeout=60) as r:
                for line in r:
                    line = line.decode().strip()
                    if not line or line == "data: [DONE]": continue
                    if line.startswith("data: "): line = line[6:]
                    try:
                        token = json.loads(line)["choices"][0]["delta"].get("content", "")
                        if token: yield token
                    except Exception: pass
            return
        except Exception:
            pass
        try:
            url = f"{self._base_url}/api/generate"
            payload2 = {"model": self._model, "prompt": str(prompt), "stream": True,
                        "options": {"temperature": self._sampler_params["temperature"],
                                    "top_p": self._sampler_params["top_p"],
                                    "num_predict": self._sampler_params["max_tokens"]}}
            req = urllib.request.Request(url, data=json.dumps(payload2).encode(), headers=self._headers(), method="POST")
            with urllib.request.urlopen(req, timeout=60) as r:
                for line in r:
                    try:
                        chunk = json.loads(line.decode().strip())
                        token = chunk.get("response", "")
                        if token: yield token
                        if chunk.get("done"): break
                    except Exception: pass
        except Exception:
            yield str(self.run_inference(inputs)["output"])

    def get_metadata(self) -> dict:
        meta = super().get_metadata()
        meta.update({"base_url": self._base_url, "model": self._model})
        return meta

    def shutdown(self) -> None:
        pass
