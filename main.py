"""
main.py — Entrypoint for Adaptive Modular Inference GUI
"""
import argparse
import json
import sys
import traceback
import atexit

from nicegui import ui

from core.detector import FrameworkDetector
from core.adapter import AdapterFactory, AdapterLoadError
from core.session import InferenceSession
from core.sharded_session import ShardedSession
from core.router import ButtonRouter
from core.headless import HeadlessOrchestrator
from headless.server import HeadlessServer
from ui.layout import build_layout


def _build_session(adapter_or_adapters, config):
    """gap 3+4: Use ShardedSession when nodes > 1 or sharding_strategy is set."""
    scaling = config.get("scaling", {})
    nodes = scaling.get("nodes", 1)
    strategy = scaling.get("sharding_strategy", "none")

    if nodes > 1 or strategy not in ("none", ""):
        if isinstance(adapter_or_adapters, list):
            adapters = adapter_or_adapters
        else:
            # Build N copies of the same adapter for multi-node
            framework_key = adapter_or_adapters.FRAMEWORK_KEY
            adapters = [adapter_or_adapters]
            for _ in range(nodes - 1):
                try:
                    adapters.append(AdapterFactory.build(framework_key, config))
                except Exception:
                    break
        return ShardedSession(adapters, config)

    return InferenceSession(adapter_or_adapters, config)


def main():
    parser = argparse.ArgumentParser(description="Adaptive Modular Inference GUI")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    args = parser.parse_args()

    try:
        with open(args.config) as f:
            config = json.load(f)
    except Exception as e:
        print(f"Failed to load config from '{args.config}': {e}", file=sys.stderr)
        sys.exit(1)

    try:
        framework_key = FrameworkDetector(config).detect()
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    try:
        adapter = AdapterFactory.build(framework_key, config)
    except AdapterLoadError as e:
        print(f"AdapterLoadError: framework='{e.framework_key}', path='{e.path}'", file=sys.stderr)
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    session = _build_session(adapter, config)
    router = ButtonRouter(session, config)

    orchestrator = None
    server = None
    if config.get("headless", {}).get("enabled", False):
        orchestrator = HeadlessOrchestrator(session, config)
        server = HeadlessServer(orchestrator, config)
        server.start()
        # gap 6 fix: register clean shutdown
        atexit.register(server.stop)

    build_layout(session, router, config, orchestrator=orchestrator)
    ui.run(title="Adaptive Inference GUI", port=8080, reload=False)


if __name__ == "__main__":
    main()
