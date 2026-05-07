"""
tests/test_layout.py — ButtonRouter tests (no NiceGUI needed)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import MagicMock

from core.router import ButtonRouter


BUTTON_CONFIG = [
    {"id": "toggle_stream", "label": "Stream Tokens", "type": "toggle", "target": "session.stream_mode", "default": True, "group": "inference"},
    {"id": "btn_run", "label": "Run Inference", "type": "trigger", "target": "session.run", "group": "inference"},
    {"id": "slider_temp", "label": "Temperature", "type": "slider", "target": "model.temperature", "min": 0.0, "max": 2.0, "step": 0.05, "default": 0.7, "group": "weights"},
]


class TestButtonRouter(unittest.TestCase):

    def setUp(self):
        self.session = MagicMock()
        self.config = {"buttons": BUTTON_CONFIG}
        self.router = ButtonRouter(self.session, self.config)

    def test_toggle_dispatches_apply_weight(self):
        self.router.dispatch("toggle_stream", True)
        self.session.apply_weight.assert_called_once_with("session.stream_mode", True)

    def test_trigger_dispatches_run(self):
        self.router.dispatch("btn_run", None)
        self.session.run.assert_called_once_with({})

    def test_slider_dispatches_apply_weight(self):
        self.router.dispatch("slider_temp", 1.2)
        self.session.apply_weight.assert_called_once_with("model.temperature", 1.2)

    def test_unknown_id_does_not_raise(self):
        try:
            self.router.dispatch("nonexistent_button", None)
        except Exception as e:
            self.fail(f"dispatch raised unexpectedly: {e}")

    def test_get_button_configs_returns_all_buttons(self):
        configs = self.router.get_button_configs()
        self.assertEqual(len(configs), 3)

    def test_dispatch_error_does_not_propagate(self):
        self.session.run.side_effect = RuntimeError("internal error")
        try:
            self.router.dispatch("btn_run", None)
        except Exception as e:
            self.fail(f"dispatch let exception escape: {e}")


if __name__ == "__main__":
    unittest.main()
