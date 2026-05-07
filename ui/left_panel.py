"""
ui/left_panel.py — Control panel with Weight Inspector (gap_02)
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
            try:
                if btn["type"] == "toggle":
                    sw = ui.switch(btn["label"], value=btn.get("default", False))
                    sw.on("update:model-value", lambda e, b=btn: router.dispatch(b["id"], e.args))
                elif btn["type"] == "trigger":
                    ui.button(btn["label"]).on("click", lambda b=btn: router.dispatch(b["id"], None))
                elif btn["type"] == "slider":
                    ui.label(btn["label"]).classes("text-sm mt-2")
                    sl = ui.slider(min=btn["min"], max=btn["max"], step=btn["step"], value=btn.get("default", btn["min"]))
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

    # gap_02: Weight Inspector fully implemented
    ui.separator()
    with ui.expansion("Weight Inspector").classes("w-full"):
        weight_keys = session.adapter.get_weight_keys()
        if not weight_keys:
            ui.label("No weights exposed by this adapter.").classes("text-gray-400 text-sm")
        else:
            ui.label(f"{len(weight_keys)} weights").classes("text-xs text-gray-400 mb-2")
            for key in weight_keys:
                with ui.row().classes("w-full items-center gap-2 my-1"):
                    ui.label(key).classes("text-xs font-mono flex-1 truncate text-gray-300")
                    inp = ui.input(placeholder="value").classes("w-24 text-xs")
                    def _apply(k=key, i=inp):
                        try:
                            val = float(i.value) if "." in i.value else int(i.value)
                        except ValueError:
                            val = i.value
                        try:
                            session.apply_weight(k, val)
                            ui.notify(f"Set {k} = {val}", type="positive")
                        except Exception as e:
                            ui.notify(str(e), type="negative")
                    ui.button("set", on_click=_apply).classes("text-xs px-2 py-1")
