"""
core/headless.py — Transport-agnostic orchestration layer
"""
import json


class HeadlessOrchestrator:

    def __init__(self, session, config: dict):
        self.session = session
        self.config = config
        self.running = False  # set to True by HeadlessServer.start()

    def infer(self, payload: dict) -> dict:
        from headless.protocols import validate
        ok, errors = validate("INFER_REQUEST", payload)
        if not ok:
            return {"error": "; ".join(errors), "code": 400, "context": "validate"}
        try:
            result = self.session.run(payload["inputs"])
            metadata = self.session.adapter.get_metadata()
            result["framework"] = metadata.get("framework", "unknown")
            result.setdefault("tokens_generated", 0)
            return result
        except Exception as e:
            return {"error": str(e), "code": 500, "context": "infer"}

    def stream(self, payload: dict):
        """gap 8 fix: yield {token: ...} dicts, not raw strings."""
        from headless.protocols import validate
        ok, errors = validate("INFER_REQUEST", payload)
        if not ok:
            yield json.dumps({"error": "; ".join(errors), "code": 400})
            return
        try:
            for token in self.session.stream(payload["inputs"]):
                yield json.dumps({"token": token})
        except Exception as e:
            yield json.dumps({"error": str(e), "code": 500})

    def set_weight(self, key: str, value) -> dict:
        try:
            self.session.apply_weight(key, value)
            return {"ok": True, "key": key}
        except Exception as e:
            return {"error": str(e), "code": 500}

    def get_weights(self) -> list:
        return self.session.adapter.get_weight_keys()
