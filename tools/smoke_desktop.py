"""Exercise our own Tk widgets/controllers on a display; no external UI automation.

Run from the project root: python -m tools.smoke_desktop
"""
import argparse
import json
from pathlib import Path
import time
import tkinter as tk
import numpy as np
from desktop.application import Application
from desktop.export import export_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true", help="Exercise the GUI without an NVIDIA GPU")
    args = parser.parse_args()
    root = tk.Tk()
    callback_errors = []
    root.report_callback_exception = lambda *args: callback_errors.append(str(args[1]))
    app = Application(root)
    def wait(predicate, timeout=20):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            root.update()
            if callback_errors:
                raise AssertionError(callback_errors)
            if predicate():
                return
            time.sleep(.01)
        raise AssertionError("Timed out waiting for UI state")
    try:
        wait(lambda: app.hardware is not None)
        if args.cpu_only:
            if not app.hardware["cuda"]["available"]:
                assert app.variables["mode"].get() == "CPU only"
            app.variables["mode"].set("CPU only")
        else:
            assert app.hardware["cuda"]["available"], app.hardware
        for size in (64, 128, 256, 512, 1024):
            app.variables["size"].set(str(size))
            wait(lambda: app.settings is not None and app.settings.size == size and app.preview_job is None)
            assert app.preview_grid.shape == (size, size)
        app.variables["cell_size"].set("12")
        wait(lambda: app.preview.cell_size == 12 and app.preview.pending is None)
        # The backing image is a viewport tile, not a giant 12288² bitmap.
        assert app.preview.photo.width() <= app.preview.canvas.winfo_width() + 36
        assert app.preview.canvas.canvasx(0) > 0
        app.variables["size"].set("64")
        app.variables["cell_size"].set("4")
        for pattern, alive in (("Glider", 5), ("Glider Gun", 36), ("R-pentomino", 5), ("Blinker", 3)):
            app.variables["pattern"].set(pattern)
            wait(lambda: app.preview_job is None)
            assert app.preview_grid.sum() == alive
        app.variables["pattern"].set("Random")
        app.variables["seed"].set("")
        wait(lambda: app.preview_job is None)
        assert app.settings is None
        assert app.run_button.instate(["disabled"])
        app.variables["seed"].set("42")
        app.variables["iterations"].set("20")
        wait(lambda: app.preview_job is None)
        assert not app.run_button.instate(["disabled"]), (app.validation_text.get(), {k:v.get() for k,v in app.variables.items()})
        app.run_button.invoke()
        assert not callback_errors, callback_errors
        window = app.comparison
        assert window.state == "ready"
        window.start_button.invoke()
        wait(lambda: window.latest_engines.get("cpu", {}).get("generations", 0) >= 2)
        window.pause_button.invoke()
        wait(lambda: window.state == "paused")
        paused_at = window.latest_engines["cpu"]["generations"]
        until = time.monotonic() + .2
        wait(lambda: time.monotonic() >= until)
        assert window.latest_engines["cpu"]["generations"] == paused_at
        window.pause_button.invoke()
        wait(lambda: window.latest_engines["cpu"]["generations"] > paused_at)
        window.stop_button.invoke()
        wait(lambda: window.state == "stopped" and not window.is_running())
        assert window.report["speedup"] is None
        assert window.report["status"] == "stopped"
        window.reset_button.invoke()
        assert window.report is None and window.state == "ready"
        np.testing.assert_array_equal(window.grids["cpu"].grid_data, app.preview_grid)
        window.start_button.invoke()
        wait(lambda: window.latest_engines.get("cpu", {}).get("generations", 0) >= 2)
        window.reset_button.invoke()
        wait(lambda: window.state == "ready" and not window.is_running())
        assert window.report is None
        window.start_button.invoke()
        wait(lambda: window.state == "complete" and not window.is_running())
        assert window.report["results_match"] is (None if args.cpu_only else True)
        window.unit.set("s")
        assert window.axes.get_xlabel().endswith("(s)")
        assert window.axes.get_ylabel() == "Iterations / s"
        window.close()

        app.variables["display"].set("Benchmark (no drawing)")
        app.variables["iterations"].set("100")
        app.variables["block"].set("32×32")
        wait(lambda: app.preview_job is None)
        app.run_button.invoke()
        window = app.comparison
        window.start_button.invoke()
        wait(lambda: window.state in ("complete", "error") and not window.is_running(), 40)
        assert window.state == "complete", window.status_text.get()
        assert window.report["results_match"] is (None if args.cpu_only else True)
        if not args.cpu_only:
            assert window.report["engines"]["cuda"]["preview_transfer_ms"] == 0
        folder = export_bundle(window.report, Path("results") / "desktop-validation")
        print(json.dumps({"gui_smoke": "PASS", "export": str(folder),
                          "hardware": app.hardware["cuda"].get("name", "CPU only"),
                          "cpu_ms": window.report["engines"]["cpu"]["total_ms"],
                          "gpu_ms": window.report["engines"].get("cuda", {}).get("total_ms"),
                          "results_match": window.report["results_match"]}, indent=2), flush=True)
        window.close()
        root.update()
        assert not callback_errors, callback_errors
        # A finished worker may still have an undelivered final packet. Reset must discard it.
        app.variables["mode"].set("CPU only")
        app.variables["cpu_backend"].set("NumPy optimized")
        app.variables["iterations"].set("1")
        wait(lambda: app.preview_job is None)
        app.run_button.invoke()
        window = app.comparison
        window.start_button.invoke()
        window.worker.join(5)
        assert not window.is_running()
        window.reset_button.invoke()
        root.update()
        assert window.state == "ready" and window.report is None
        window.close()
    finally:
        app.close()


if __name__ == "__main__":
    main()
