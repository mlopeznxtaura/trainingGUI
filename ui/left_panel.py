"""
ui/left_panel.py — Control panel (left half)
"""
import collections
from nicegui import ui
from ui import components


def build(session, router, config, orchestrator=None) -> None:
    ui.label("Controls").classes("text-xl font-bold mb-2")

    configs = router.get_button_configs()
    groups = collections.defaultdict(list)
    for btn in configs:
        groups[btn.get("group", "default")].append(btn)

    for group_name in sorted(groups.keys()):
        ui.label(group_name).classes("text-sm font-semibold uppercase text-gray-400 mt-4")
        for btn in groups[group_name]:
            btn_type = btn.get("type", "trigger")
            try:
                if btn_type == "toggle":
                    sw = ui.switch(btn["label"], value=btn.get("default", False))
                    sw.on("update:model-value", lambda e, b=btn: router.dispatch(b["id"], e.args))
                elif btn_type == "trigger":
                    ui.button(btn["label"]).on("click", lambda b=btn: router.dispatch(b["id"], None))
                elif btn_type == "slider":
                    ui.label(btn["label"]).classes("text-sm mt-2")
                    sl = ui.slider(
                        min=btn["min"],
                        max=btn["max"],
                        step=btn["step"],
                        value=btn.get("default", btn["min"]),
                    )
                    sl.on("update:model-value", lambda e, b=btn: router.dispatch(b["id"], e.args))
            except Exception as e:
                ui.notify(str(e), type="negative")
        ui.separator().classes("my-2")

    ui.separator()
    components.metadata_card(session.adapter.get_metadata())

    headless_enabled = config.get("headless", {}).get("enabled", False)
    if headless_enabled and orchestrator is not None:
        components.status_dot("Headless Server", lambda: orchestrator.running)
    else:
        components.status_dot("Headless Server (disabled)", lambda: False)

    ui.separator()
    with ui.expansion("Weight Inspector").classes("w-full"):
        ui.label("See gap_02 in agents.json").classes("text-gray-400 text-sm")
