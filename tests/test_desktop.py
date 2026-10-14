"""Controller tests run without a window. GPU cases require actual CUDA hardware."""
import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import numpy as np
from desktop.models import Settings, Measurement
from desktop.runner import Control, compare_results, execute
from life.core import Config, make_grid, reference_step
from life.cpu import CPUEngine
from life.python_cpu import PythonEngine
from life.gpu import cuda_status

HARDWARE = {"cpu": "Test CPU", "cuda": {"available": False}, "cuda_runtime": "Unavailable"}


class DesktopTests(unittest.TestCase):
    def test_python_loops_against_reference_and_numpy(self):
        for boundary in ("wrap", "dead"):
            for shape in ((1, 1), (2, 3), (7, 11)):
                initial = (np.random.default_rng(42).random(shape) < .3).astype(np.uint8)
                engine = PythonEngine(initial, boundary)
                expected = initial
                for _ in range(4):
                    expected = reference_step(expected, boundary)
                    np.testing.assert_array_equal(engine.step(), expected)
                np.testing.assert_array_equal(expected, CPUEngine(initial, boundary).advance(4))
                self.assertFalse(np.shares_memory(initial, engine.current))

    def test_settings_validate_user_inputs(self):
        for kwargs in ({"size": 32}, {"iterations": 0}, {"iterations": True}, {"iterations": 100001},
                       {"seed": -1}, {"density": .51}, {"density": float("nan")},
                       {"cell_size": 0}, {"block": 24}, {"mode": "fake"}, {"cpu_backend": "fake"}):
            with self.assertRaises(ValueError):
                Settings(**kwargs)
        self.assertEqual(Settings(iterations=5000, block=32).grid_config().block_x, 32)

    def test_metrics_formula_and_sample_retention(self):
        m = Measurement("python", 4096, setup_ms=2, output_ms=1, durations_ms=[1, 3])
        data = m.summary(True)
        self.assertEqual(data["total_ms"], 7)
        self.assertEqual(data["average_ms"], 3.5)
        self.assertEqual(data["min_ms"], 1)
        self.assertEqual(data["max_ms"], 3)
        self.assertAlmostEqual(data["cells_per_second"], 4096 * 2 * 1000 / 7)
        self.assertEqual(data["series"][-1], [2, 4, 1/3])

    def test_cpu_only_has_no_gpu_results_or_speedup(self):
        s = Settings(mode="cpu", display="benchmark", iterations=4, pattern="blinker")
        report, finals = execute(s, hardware=HARDWARE)
        self.assertEqual(report["status"], "complete")
        self.assertNotIn("cuda", report["engines"])
        self.assertIsNone(report["speedup"])
        self.assertIsNone(report["results_match"])
        np.testing.assert_array_equal(finals["cpu"], make_grid(s.grid_config()))
        self.assertEqual(report["engines"]["cpu"]["final_alive"], 3)

    def test_stop_keeps_partial_results_without_comparison(self):
        control = Control()
        def publish(packet):
            if packet["state"] == "running":
                control.stop()
        s = Settings(mode="cpu", display="live", iterations=100, cpu_backend="numpy")
        report, _ = execute(s, control, publish, HARDWARE)
        self.assertEqual(report["status"], "stopped")
        self.assertEqual(report["engines"]["cpu"]["generations"], 1)
        self.assertIsNone(report["speedup"])

    def test_stop_before_setup_returns_valid_empty_report(self):
        control = Control()
        control.stop()
        report, frames = execute(Settings(mode="cpu"), control, hardware=HARDWARE)
        self.assertEqual(report["status"], "stopped")
        self.assertEqual(report["engines"], {})
        self.assertEqual(frames, {})

    def test_pause_blocks_and_stop_wakes_the_worker(self):
        control = Control()
        control.pause()
        reached = threading.Event()
        finished = threading.Event()
        result = []
        def worker():
            result.append(control.checkpoint(reached.set))
            finished.set()
        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(reached.wait(1))
        self.assertFalse(finished.wait(.04))
        control.stop()
        thread.join(1)
        self.assertEqual(result, [False])

    def test_pause_resume_continues(self):
        control = Control()
        control.pause()
        reached = threading.Event()
        result = []
        thread = threading.Thread(target=lambda: result.append(control.checkpoint(reached.set)))
        thread.start()
        self.assertTrue(reached.wait(1))
        control.pause(False)
        thread.join(1)
        self.assertEqual(result, [True])

    def test_mismatch_and_incomplete_runs_withhold_speedup(self):
        settings = Settings(iterations=1)
        summaries = {k: {"generations": 1, "total_ms": t} for k, t in (("cpu", 20), ("cuda", 2))}
        a = np.zeros((64, 64), dtype=np.uint8)
        match, speedup = compare_results(settings, summaries, {"cpu": a, "cuda": a.copy()}, "complete")
        self.assertTrue(match)
        self.assertEqual(speedup, 10)
        self.assertEqual(compare_results(settings, summaries, {"cpu": a, "cuda": 1-a}, "complete"), (False, None))
        self.assertEqual(compare_results(settings, summaries, {}, "stopped"), (None, None))

    @patch("desktop.runner.cuda_status", return_value={"available": False, "reason": "No GPU"})
    def test_gpu_unavailable_never_emulates_measurements(self, _):
        for mode in ("both", "cuda"):
            with self.assertRaisesRegex(RuntimeError, "CPU only"):
                execute(Settings(mode=mode), hardware=HARDWARE)

    def test_csv_bundle_contains_raw_samples_and_metadata(self):
        try:
            from desktop.export import export_csv, export_bundle
        except ImportError:
            self.skipTest("Desktop export libraries not installed")
        report, _ = execute(Settings(mode="cpu", display="benchmark", iterations=3, cpu_backend="numpy"), hardware=HARDWARE)
        with tempfile.TemporaryDirectory() as folder:
            paths = export_csv(report, Path(folder) / "run.csv")
            self.assertTrue(all(p.exists() for p in paths))
            self.assertEqual(json.loads(paths[2].read_text())["settings"]["seed"], 42)
            with paths[1].open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[-1]["generation"], "3")
            bundle = export_bundle(report, folder, "s")
            self.assertTrue((bundle / "comparison.pdf").exists())
            self.assertTrue((bundle / "throughput.svg").exists())


@unittest.skipUnless(cuda_status()["available"], "Real CUDA device required")
class DesktopGPUTests(unittest.TestCase):
    def test_all_blocks_boundaries_and_100_generations(self):
        for block in (8, 16, 32):
            for boundary in ("wrap", "dead"):
                settings = Settings(size=64, iterations=100, block=block, boundary=boundary,
                                    cpu_backend="numpy", display="benchmark")
                report, _ = execute(settings, hardware=HARDWARE)
                self.assertIs(report["results_match"], True)
                gpu = report["engines"]["cuda"]
                self.assertEqual(gpu["generations"], 100)
                self.assertEqual(len(gpu["kernel_generation_ms"]), 100)
                self.assertGreater(gpu["kernel_ms"], 0)
                self.assertGreater(gpu["transfer_ms"], 0)
                self.assertEqual(gpu["buffer_bytes"], 2*64*64)
                self.assertEqual(gpu["preview_transfer_ms"], 0)

    def test_live_python_cuda_and_gpu_only_modes(self):
        settings = Settings(iterations=3, display="live", pattern="glider")
        report, grids = execute(settings, hardware=HARDWARE)
        self.assertIs(report["results_match"], True)
        self.assertGreater(report["engines"]["cuda"]["preview_transfer_ms"], 0)
        solo, solo_grid = execute(replace(settings, mode="cuda", display="benchmark"), hardware=HARDWARE)
        self.assertNotIn("cpu", solo["engines"])
        self.assertIsNone(solo["results_match"])
        np.testing.assert_array_equal(grids["cuda"], solo_grid["cuda"])


if __name__ == "__main__":
    unittest.main()
