"""
core/router.py — Button-to-session dispatcher
"""
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class ButtonRouter:

    def __init__(self, session, config: dict):
        self._button_configs = config.get("buttons", [])
        self._handlers = {}

        for btn in self._button_configs:
            btn_id = btn["id"]
            btn_type = btn["type"]

            if btn_type == "toggle":
                self._handlers[btn_id] = lambda value, b=btn: session.apply_weight(b["target"], value)
            elif btn_type == "trigger":
                self._handlers[btn_id] = lambda _: session.run({})
            elif btn_type == "slider":
                self._handlers[btn_id] = lambda value, b=btn: session.apply_weight(b["target"], value)

    def dispatch(self, button_id: str, value) -> None:
        if button_id not in self._handlers:
            logger.warning("Unknown button id: %s", button_id)
            return
        try:
            self._handlers[button_id](value)
        except Exception as e:
            logger.error("Error dispatching button %s: %s", button_id, e)

    def get_button_configs(self) -> list:
        return self._button_configs
