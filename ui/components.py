"""
ui/components.py — Reusable NiceGUI sub-components
"""
from nicegui import ui


def status_dot(label: str, state_fn) -> None:
    with ui.row().classes("items-center gap-2 mt-2"):
        dot = ui.label(" ").classes("w-3 h-3 rounded-full bg-gray-500")
        ui.label(label).classes("text-sm")
    ui.timer(1.0, callback=lambda: dot.style(
        "background-color: #22c55e" if state_fn() else "background-color: #6b7280"))


def metadata_card(data: dict) -> None:
    with ui.card().classes("w-full mt-2 p-2"):
        ui.label("Framework Info").classes("font-semibold text-sm mb-1")
        for k, v in data.items():
            with ui.row().classes("justify-between w-full"):
                ui.label(str(k)).classes("text-xs text-gray-400")
                ui.label(str(v)).classes("text-xs")


def token_log(max_lines: int = 500):
    log = ui.log(max_lines=max_lines)
    log.classes("w-full flex-1 font-mono text-sm bg-gray-900 rounded p-2")
    return log


def run_stats_bar(stats: dict) -> None:
    with ui.row().classes("w-full justify-between mt-1 text-xs text-gray-400"):
        stats["_labels"] = [
            ui.label(f"Latency: {stats['latency_ms']:.1f}ms"),
            ui.label(f"Tokens/sec: {stats['tokens_per_sec']:.1f}"),
            ui.label(f"Runs: {stats['run_count']}"),
        ]


def update_stats_bar(stats: dict) -> None:
    labels = stats.get("_labels", [])
    if len(labels) == 3:
        labels[0].set_text(f"Latency: {stats['latency_ms']:.1f}ms")
        labels[1].set_text(f"Tokens/sec: {stats['tokens_per_sec']:.1f}")
        labels[2].set_text(f"Runs: {stats['run_count']}")
