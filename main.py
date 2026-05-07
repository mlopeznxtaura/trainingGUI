"""
main.py — Entrypoint for Adaptive Modular Inference GUI
"""
import argparse
import json
import sys
import traceback
import threading

from nicegui import ui

from core.detector import FrameworkDetector
from core.adapter import AdapterFactory, AdapterLoadError
from core.session import InferenceSession
from core.router import ButtonRouter
from core.headless import HeadlessOrchestrator
from headless.server import HeadlessServer
from ui.layout import build_layout


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
        print(f"AdapterLoadError: framework='{e.framework_key}', path='{e.path}': {e}", file=sys.stderr)
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    session = InferenceSession(adapter, config)
    router = ButtonRouter(session, config)

    orchestrator = None
    if config.get("headless", {}).get("enabled", False):
        orchestrator = HeadlessOrchestrator(session, config)
        server = HeadlessServer(orchestrator, config)
        server.start()

    build_layout(session, router, config, orchestrator=orchestrator)

    ui.run(title="Adaptive Inference GUI", port=8080, reload=False)


if __name__ == "__main__":
    main()
