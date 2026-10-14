"""Startup failures and headless Python exports must work without a GPU."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.check_environment import check

ROOT = Path(__file__).resolve().parents[1]


class PortabilityTests(unittest.TestCase):
    def test_optional_gpu_does_not_block_cpu_but_required_gpu_fails(self):
        unavailable = {"available": False, "simulator": False, "reason": "Test: no GPU"}
        with patch("life.gpu.cuda_status", return_value=unavailable), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(check(web=True), 0)
            self.assertEqual(check(web=True, require_cuda=True), 1)
        self.assertIn("CPU mode is ready", output.getvalue())
        self.assertIn("ERROR: Test: no GPU", output.getvalue())

    def test_startup_help_needs_no_display_and_invalid_flag_fails(self):
        help_run = subprocess.run([sys.executable, str(ROOT / "app.py"), "--help"], capture_output=True, text=True)
        self.assertEqual(help_run.returncode, 0, help_run.stderr)
        self.assertIn("--check", help_run.stdout)
        bad_run = subprocess.run([sys.executable, str(ROOT / "app.py"), "--require-cuda"], capture_output=True, text=True)
        self.assertEqual(bad_run.returncode, 2)
        self.assertIn("must be used with --check", bad_run.stderr)

    def test_headless_experiment_exports_to_path_with_spaces(self):
        try:
            from tools.run_experiment import main
        except ImportError:
            self.skipTest("Desktop export dependencies not installed")
        with tempfile.TemporaryDirectory(prefix="life lab ") as folder:
            path = Path(folder) / "my results" / "statistics.csv"
            with contextlib.redirect_stdout(io.StringIO()):
                result = main(["--mode", "cpu", "--iterations", "4", "--output", str(path)])
            self.assertEqual(result, 0)
            report = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(report["settings"]["cpu_backend"], "python")
            self.assertEqual(report["engines"]["cpu"]["generations"], 4)
            self.assertNotIn("cuda", report["engines"])
            self.assertIsNone(report["speedup"])
            self.assertTrue(path.exists())
            self.assertTrue(path.with_name("statistics_generations.csv").exists())


if __name__ == "__main__":
    unittest.main()
