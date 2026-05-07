# Adaptive Modular Inference GUI

Real-time adaptive inference GUI. 50/50 split layout. Framework-agnostic. OS-agnostic. Config-driven scaling from single GPU to multi-node cluster.

---

## Installation

```bash
pip install nicegui
```

All other dependencies are framework-specific and must be installed by the user. The app detects whatever is present.

---

## Running

```bash
python main.py
```

Optional: point to a custom config file:

```bash
python main.py --config /path/to/config.json
```

---

## Configuration

`config.json` is the single scaling and behavior lever. No code changes are needed to switch frameworks, devices, or deployment scale.

### Key fields

| Field | Purpose |
|---|---|
| `framework` | `"auto"` detects all installed frameworks. Set to a specific key to lock. |
| `model_path` | OS-agnostic path. Use `~` or forward slashes. |
| `device` | `"auto"`, `"cpu"`, `"cuda"`, `"mps"` |
| `scaling.nodes` | Number of cluster nodes |
| `scaling.gpus_per_node` | GPUs per node |
| `buttons` | Array of UI controls — all rendered from config, no hardcoding |
| `headless.enabled` | Start headless server alongside GUI |
| `headless.transport` | `"stdio"`, `"http"`, `"unix_socket"` |
| `output.refresh_ms` | Right panel polling interval |

### Scaling example — single GPU to cluster

Change only `config.json`:

```json
"scaling": {
  "nodes": 8,
  "gpus_per_node": 4,
  "sharding_strategy": "input_split"
}
```

No other files change.

---

## Adding a New Framework

1. Create `frameworks/myframework_adapter.py`
2. Implement `BaseFrameworkAdapter` from `frameworks/base.py` — all abstract methods required
3. Add entry to `AdapterFactory._registry` dict in `core/adapter.py`
4. Add detection logic in `core/detector.py` using `importlib.util.find_spec()` or path probe

---

## Headless Mode

Enable in config:

```json
"headless": {
  "enabled": true,
  "transport": "http",
  "port": 8765
}
```

Example infer request (HTTP):

```bash
curl -X POST http://localhost:8765/infer \
  -H "Content-Type: application/json" \
  -d '{"inputs": {"prompt": "hello"}, "stream": false}'
```

Example stream request:

```bash
curl -X POST http://localhost:8765/stream \
  -H "Content-Type: application/json" \
  -d '{"inputs": {"prompt": "hello"}, "stream": true}'
```

Use `headless/client.py` for programmatic access:

```python
from headless.client import HeadlessClient
client = HeadlessClient(transport="http", port=8765)
print(client.infer({"prompt": "hello"}))
for token in client.stream({"prompt": "hello"}):
    print(token, end="", flush=True)
```

---

## OS Notes

All paths use `os.path.expanduser` and `pathlib.Path`. No OS-specific code exists in core or framework modules. Platform differences are isolated to `config.json["os_paths"]` and the Unix socket fallback in `headless/server.py`. The app runs identically on Linux, macOS, and Windows.

---

## Gap Tasks

See `agents.json` for all known gaps with tasks and tests. Each gap is self-contained and can be implemented without touching core files.
