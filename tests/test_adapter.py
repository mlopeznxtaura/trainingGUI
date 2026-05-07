"""
tests/test_adapter.py — AdapterFactory + InferenceSession tests
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import MagicMock, patch

from core.adapter import AdapterFactory, AdapterLoadError
from core.session import InferenceSession


BASE_CONFIG = {
    "model_path": "/tmp/test_model",
    "device": "cpu",
    "output": {"stream": True, "refresh_ms": 100},
}


class TestAdapterFactory(unittest.TestCase):

    def test_unknown_key_raises_adapter_load_error(self):
        with self.assertRaises(AdapterLoadError):
            AdapterFactory.build("totally_fake_framework", BASE_CONFIG)

    def test_known_key_returns_correct_class(self):
        mock_adapter = MagicMock()
        mock_adapter.load_model = MagicMock(return_value=None)

        mock_module = MagicMock()
        mock_module.PyTorchAdapter = MagicMock(return_value=mock_adapter)

        with patch("core.adapter.importlib.import_module", return_value=mock_module):
            result = AdapterFactory.build("pytorch", BASE_CONFIG)
        self.assertEqual(result, mock_adapter)

    def test_load_model_failure_raises_adapter_load_error(self):
        mock_adapter = MagicMock()
        mock_adapter.load_model.side_effect = RuntimeError("no model found")

        mock_module = MagicMock()
        mock_module.PyTorchAdapter = MagicMock(return_value=mock_adapter)

        with patch("core.adapter.importlib.import_module", return_value=mock_module):
            with self.assertRaises(AdapterLoadError):
                AdapterFactory.build("pytorch", BASE_CONFIG)


class TestInferenceSession(unittest.TestCase):

    def setUp(self):
        self.adapter = MagicMock()
        self.adapter.run_inference.return_value = {"output": "result", "latency_ms": 5.0}
        self.adapter.stream_tokens.return_value = iter(["a", "b", "c"])
        self.adapter.set_weight.return_value = None
        self.session = InferenceSession(self.adapter, BASE_CONFIG)

    def test_run_appends_to_history(self):
        self.session.run({"prompt": "hello"})
        self.assertEqual(len(self.session.history), 1)

    def test_callback_fires_on_output(self):
        called_with = []
        self.session.register_callback("on_output", lambda r: called_with.append(r))
        self.session.run({"prompt": "hello"})
        self.assertEqual(len(called_with), 1)
        self.assertIn("output", called_with[0])

    def test_apply_weight_calls_adapter(self):
        self.session.apply_weight("model.temperature", 0.9)
        self.adapter.set_weight.assert_called_once_with("model.temperature", 0.9)

    def test_get_history_returns_copy(self):
        self.session.run({"prompt": "hi"})
        h = self.session.get_history()
        h.clear()
        self.assertEqual(len(self.session.history), 1)

    def test_stream_yields_tokens(self):
        tokens = list(self.session.stream({"prompt": "hi"}))
        self.assertEqual(tokens, ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
