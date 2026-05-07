"""
tests/test_headless.py — HeadlessOrchestrator + protocol validator tests
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import MagicMock, patch

from core.headless import HeadlessOrchestrator
from headless.protocols import validate


class TestHeadlessOrchestrator(unittest.TestCase):

    def setUp(self):
        self.adapter = MagicMock()
        self.adapter.get_metadata.return_value = {"framework": "pytorch", "device": "cpu", "model_path": "/tmp/m", "adapter_class": "PyTorchAdapter"}
        self.adapter.get_weight_keys.return_value = ["layer.0.weight"]
        self.session = MagicMock()
        self.session.adapter = self.adapter
        self.session.run.return_value = {"output": "hello", "latency_ms": 10.0}
        self.session.stream.return_value = iter(["tok1", "tok2", "tok3"])
        self.config = {"headless": {"enabled": False}}
        self.orch = HeadlessOrchestrator(self.session, self.config)

    def test_infer_returns_output_and_framework_key(self):
        result = self.orch.infer({"inputs": {"prompt": "hi"}, "stream": False})
        self.assertIn("output", result)
        self.assertIn("framework", result)

    def test_infer_error_returns_error_dict_not_raise(self):
        self.session.run.side_effect = RuntimeError("boom")
        result = self.orch.infer({"inputs": {}, "stream": False})
        self.assertIn("error", result)
        self.assertEqual(result.get("code"), 500)

    def test_stream_yields_tokens(self):
        tokens = list(self.orch.stream({"inputs": {"prompt": "hi"}, "stream": True}))
        self.assertEqual(tokens, ["tok1", "tok2", "tok3"])

    def test_get_weights_delegates_to_adapter(self):
        result = self.orch.get_weights()
        self.session.adapter.get_weight_keys.assert_called_once()
        self.assertIsInstance(result, list)

    def test_set_weight_returns_ok_dict(self):
        result = self.orch.set_weight("layer.0.weight", 0.5)
        self.assertTrue(result.get("ok"))
        self.assertEqual(result.get("key"), "layer.0.weight")


class TestProtocolValidator(unittest.TestCase):

    def test_valid_infer_request(self):
        ok, errors = validate("INFER_REQUEST", {"inputs": {}, "stream": False})
        self.assertTrue(ok)
        self.assertEqual(errors, [])

    def test_missing_required_key(self):
        ok, errors = validate("INFER_REQUEST", {"stream": False})
        self.assertFalse(ok)
        self.assertTrue(any("inputs" in e for e in errors))

    def test_unknown_schema_returns_false(self):
        ok, errors = validate("MADE_UP_SCHEMA", {})
        self.assertFalse(ok)

    def test_weight_set_request_valid(self):
        ok, errors = validate("WEIGHT_SET_REQUEST", {"key": "lr", "value": 0.01})
        self.assertTrue(ok)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
