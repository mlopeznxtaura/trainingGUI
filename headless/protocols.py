"""
headless/protocols.py — Message schema definitions and validator
"""

INFER_REQUEST = {"inputs": dict, "stream": bool, "model_id": str, "options": dict}
INFER_RESPONSE = {"output": ..., "latency_ms": float, "tokens_generated": int, "framework": str}
WEIGHT_SET_REQUEST = {"key": str, "value": ..., "dtype": str}
ERROR_RESPONSE = {"error": str, "code": int, "context": str}
STREAM_TOKEN = {"token": str}
STREAM_DONE = {"done": True}

_REQUIRED_KEYS = {
    "INFER_REQUEST":      ["inputs", "stream"],
    "INFER_RESPONSE":     ["output", "latency_ms", "framework"],
    "WEIGHT_SET_REQUEST": ["key", "value"],
    "ERROR_RESPONSE":     ["error", "code"],
    "STREAM_TOKEN":       ["token"],
    "STREAM_DONE":        ["done"],
}


def validate(schema_name: str, payload: dict) -> tuple:
    if schema_name not in _REQUIRED_KEYS:
        return (False, [f"Unknown schema: {schema_name}"])

    required = _REQUIRED_KEYS[schema_name]
    errors = [f"Missing required key: '{k}'" for k in required if k not in payload]

    if errors:
        return (False, errors)
    return (True, [])
