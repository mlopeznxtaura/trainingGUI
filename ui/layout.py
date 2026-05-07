"""
ui/layout.py — Top-level NiceGUI layout
"""
from nicegui import ui
from ui import left_panel, right_panel


def build_layout(session, router, config, orchestrator=None) -> None:
    with ui.row().classes("w-full h-screen flex flex-row overflow-hidden"):
        with ui.column().classes("w-1/2 h-full overflow-y-auto border-r border-gray-700 p-4"):
            left_panel.build(session, router, config, orchestrator=orchestrator)

        with ui.column().classes("w-1/2 h-full overflow-hidden flex flex-col p-4"):
            right_panel.build(session, config)
