"""
tests/test_detector.py — FrameworkDetector tests
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import patch, MagicMock

from core.detector import FrameworkDetector


BASE_CONFIG = {
    "framework": "auto",
    "framework_priority": [],
    "os_paths": {
        "linux": {"lib_search": [], "socket_dir": "/tmp"},
        "darwin": {"lib_search": [], "socket_dir": "/tmp"},
        "windows": {"lib_search": [], "socket_dir": None},
    },
    "custom_engine_path": None,
}


class TestFrameworkDetector(unittest.TestCase):

    def test_explicit_framework_skips_detection(self):
        config = dict(BASE_CONFIG)
        config["framework"] = "pytorch"
        detector = FrameworkDetector(config)
        result = detector.detect()
        self.assertEqual(result, "pytorch")

    def test_auto_detects_available_framework(self):
        config = dict(BASE_CONFIG)

        def mock_find_spec(name):
            return MagicMock() if name == "torch" else None

        with patch("core.detector.importlib.util.find_spec", side_effect=mock_find_spec):
            detector = FrameworkDetector(config)
            result = detector.detect()
        self.assertEqual(result, "pytorch")

    def test_priority_list_respected(self):
        config = dict(BASE_CONFIG)
        config["framework_priority"] = ["jax", "pytorch"]

        def mock_find_spec(name):
            return MagicMock() if name in ("torch", "jax") else None

        with patch("core.detector.importlib.util.find_spec", side_effect=mock_find_spec):
            detector = FrameworkDetector(config)
            result = detector.detect()
        self.assertEqual(result, "jax")

    def test_no_framework_raises(self):
        config = dict(BASE_CONFIG)

        with patch("core.detector.importlib.util.find_spec", return_value=None):
            detector = FrameworkDetector(config)
            with self.assertRaises(RuntimeError):
                detector.detect()

    def test_custom_cpp_detected_via_path(self):
        config = dict(BASE_CONFIG)
        config["custom_engine_path"] = "/tmp/fake_engine.so"

        with patch("core.detector.importlib.util.find_spec", return_value=None), \
             patch("core.detector.ctypes.CDLL", return_value=MagicMock()):
            detector = FrameworkDetector(config)
            result = detector.detect()
        self.assertEqual(result, "custom_cpp")


if __name__ == "__main__":
    unittest.main()
