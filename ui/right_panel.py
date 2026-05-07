"""
ui/right_panel.py — Live output panel (right half)
"""
import asyncio
import time
from nicegui import ui
from ui import components


def build(session, config) -> None:
    prompt_input = ui.textarea(placeholder="Enter prompt...").classes("w-full")

    log = components.token_log(max_lines=500)
    stats = {"latency_ms": 0.0, "tokens_per_sec": 0.0, "run_count": 0}

    async def _run_handler():
        try:
            log.clear()
            prompt = prompt_input.value

            if mode_select.value == "stream":
                t0 = time.perf_counter()
                token_count = 0
                gen = session.stream({"prompt": prompt})

                loop = asyncio.get_event_loop()

                def _next_token():
                    try:
                        return next(gen)
                    except StopIteration:
                        return None

                while True:
                    token = await loop.run_in_executor(None, _next_token)
                    if token is None:
                        break
                    log.push(token)
                    token_count += 1
                    await asyncio.sleep(0)

                elapsed_ms = (time.perf_counter() - t0) * 1000
                stats["latency_ms"] = elapsed_ms
                stats["tokens_per_sec"] = token_count / (elapsed_ms / 1000) if elapsed_ms > 0 else 0.0

            elif mode_select.value == "batch":
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, session.run, {"prompt": prompt})
                log.push(str(result.get("output", "")))
                stats["latency_ms"] = result.get("latency_ms", 0.0)
                stats["tokens_per_sec"] = 0.0

            stats["run_count"] += 1
            components.update_stats_bar(stats)

        except Exception as e:
            ui.notify(str(e), type="negative")

    with ui.row().classes("w-full items-center gap-2 my-2"):
        mode_select = ui.select(["stream", "batch"], value="stream").classes("w-32")
        ui.button("Run").on("click", lambda: ui.run_async(_run_handler()))
        ui.button("Clear").on("click", lambda: log.clear())

    def _refresh_run_count():
        h = len(session.get_history())
        if h != stats["run_count"]:
            stats["run_count"] = h
            components.update_stats_bar(stats)

    components.run_stats_bar(stats)

    refresh_interval = config.get("output", {}).get("refresh_ms", 100) / 1000
    ui.timer(refresh_interval, callback=_refresh_run_count)
