"""Check installed dependencies and actual execution without opening a window."""
import importlib
import importlib.metadata
import platform
import sys


def check(*, web=False, require_cuda=False):
    print(f"Python {platform.python_version()} on {platform.system()} ({platform.machine()})")
    print(f"Interpreter: {sys.executable}")
    if not (3, 10) <= sys.version_info[:2] <= (3, 13):
        print("ERROR: Supported Python versions are 3.10-3.13. Use Python 3.12 for a new installation.")
        return 1
    required = [("numpy", "numpy")]
    if not web:
        required += [("PIL", "Pillow"), ("matplotlib", "matplotlib"), ("tkinter", None)]
    missing = False
    for module, package in required:
        try:
            imported = importlib.import_module(module)
            version = importlib.metadata.version(package) if package else imported.TkVersion
            print(f"OK: {package or module} {version}")
        except Exception as exc:
            print(f"ERROR: {module}: {exc}")
            missing = True
    if missing:
        print("Install the requirements and Tcl/Tk described in docs/INSTALL.md.")
        return 1
    from life.core import Config, make_grid, reference_step
    from life.python_cpu import PythonEngine
    from life.gpu import CUDAEngine, cuda, cuda_status
    import numpy as np
    initial = make_grid(Config(size=64, pattern="glider"))
    expected = reference_step(initial, "wrap")
    try:
        np.testing.assert_array_equal(PythonEngine(initial, "wrap").step(), expected)
        if not web:
            importlib.import_module("desktop.application")
        print("OK: Python CPU execution" + (" and desktop imports" if not web else ""))
    except Exception as exc:
        print(f"ERROR: CPU/desktop check failed: {exc}")
        return 1
    status = cuda_status()
    if status["available"]:
        try:
            engine = CUDAEngine(initial, "wrap", (16, 16))
            engine.advance(1)
            cuda.synchronize()
            np.testing.assert_array_equal(engine.result(), expected)
            print(f"OK: real CUDA kernel compiled, executed and matched Python on {status['name']}")
        except Exception as exc:
            status = {"available": False, "reason": f"CUDA compilation/execution failed: {exc}"}
    if not status["available"]:
        print(f"{'ERROR' if require_cuda else 'OPTIONAL'}: {status['reason']}")
        if require_cuda:
            return 1
        print("CPU mode is ready. Choose CPU only in the app.")
    if not web:
        print("GUI display not opened by this check; launch app.py to test the desktop display.")
    return 0
