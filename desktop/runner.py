"""UI-independent execution, cancellation and timing. No Tk calls in workers."""
from datetime import datetime, timezone
import hashlib
import threading
from time import perf_counter
import numpy as np
from life.benchmark import environment
from life.core import make_grid
from life.cpu import CPUEngine
from life.python_cpu import PythonEngine
from life import gpu
from life.gpu import CUDAEngine, cuda, cuda_status, warmup_cuda
from .models import METHODOLOGY, Measurement, Settings


def hardware_info():
    info = environment()
    info["cuda_runtime"] = "Unavailable"
    if info["cuda"]["available"]:
        try:
            info["cuda_runtime"] = ".".join(map(str, cuda.runtime.get_version()))
            info["cuda"]["max_threads_per_block"] = int(cuda.get_current_device().MAX_THREADS_PER_BLOCK)
        except Exception:
            info["cuda_runtime"] = "Version unavailable"
    return info


class Control:
    def __init__(self):
        self.condition = threading.Condition()
        self.paused = False
        self.stopped = False

    def pause(self, paused=True):
        with self.condition:
            self.paused = paused
            self.condition.notify_all()

    def stop(self):
        with self.condition:
            self.stopped = True
            self.condition.notify_all()

    def checkpoint(self, on_pause=None):
        with self.condition:
            if self.paused and not self.stopped and on_pause:
                on_pause()
            while self.paused and not self.stopped:
                self.condition.wait()
            return not self.stopped

    def pace(self, seconds):
        with self.condition:
            if not self.stopped and not self.paused:
                self.condition.wait(timeout=seconds)


def compare_results(settings, summaries, finals, status):
    both_done = (status == "complete" and all(k in summaries for k in ("cpu", "cuda"))
                 and all(summaries[k]["generations"] == settings.iterations for k in ("cpu", "cuda")))
    match = bool(np.array_equal(finals["cpu"], finals["cuda"])) if both_done else None
    gpu_total = summaries.get("cuda", {}).get("total_ms", 0)
    speedup = summaries["cpu"]["total_ms"] / gpu_total if match and gpu_total > 0 else None
    return match, speedup


def execute(settings: Settings, control=None, publish=None, hardware=None):
    """Return (JSON-compatible report, final grids). Publication carries immutable snapshots."""
    control = control or Control()
    publish = publish or (lambda packet: None)
    hardware = hardware or hardware_info()
    initial = make_grid(settings.grid_config())
    engines, metrics, finals = {}, {}, {}
    last_publish = 0.0
    selected = [key for key in ("cpu", "cuda") if settings.mode in ("both", key)]
    if "cuda" in selected:
        status = cuda_status()
        if not status["available"]:
            raise RuntimeError(status["reason"] + " Select CPU only to continue.")
        limit = int(cuda.get_current_device().MAX_THREADS_PER_BLOCK)
        if settings.block ** 2 > limit:
            raise RuntimeError(f"This GPU allows at most {limit} threads per block.")

    def snapshot(state="running", force=False, frames=None, include_samples=False):
        nonlocal last_publish
        now = perf_counter()
        if not force and now - last_publish < .12:
            return
        last_publish = now
        publish({"state": state, "engines": {k: m.summary(include_samples) for k, m in metrics.items()},
                 "frames": frames or {}})

    publish({"state": "warming", "engines": {}, "frames": {}})
    cpu_class = PythonEngine if settings.cpu_backend == "python" else CPUEngine
    if "cpu" in selected:
        cpu_class(initial, settings.boundary).step()
    if "cuda" in selected and not control.stopped:
        warmup_cuda(initial, settings.boundary, (settings.block, settings.block))

    def setup(key):
        t = perf_counter()
        if key == "cpu":
            engine = cpu_class(initial, settings.boundary)
        else:
            engine = CUDAEngine(initial, settings.boundary, (settings.block, settings.block))
            cuda.synchronize()
        m = Measurement(settings.cpu_backend if key == "cpu" else "cuda", initial.nbytes)
        m.setup_ms = (perf_counter() - t) * 1000
        m.buffer_bytes = (sum(v.nbytes for v in vars(engine).values() if isinstance(v, np.ndarray))
                          if key == "cpu" else int(engine.current.nbytes + engine.next.nbytes))
        if key == "cuda":
            m.h2d_ms = engine.input_transfer_ms
        engines[key], metrics[key] = engine, m

    # Event objects are allocated outside measured intervals and reused in batches.
    events = [(cuda.event(timing=True), cuda.event(timing=True)) for _ in range(32)] if "cuda" in selected else []

    def step_cpu():
        t = perf_counter()
        engines["cpu"].step()
        metrics["cpu"].durations_ms.append((perf_counter() - t) * 1000)

    def step_gpu(count):
        engine, m = engines["cuda"], metrics["cuda"]
        wall_samples = []
        for i in range(count):
            begin, end = events[i]
            t = perf_counter()
            begin.record()
            gpu.life_kernel[engine.blocks, engine.block](engine.current, engine.next, engine.wrap)
            end.record()
            engine.current, engine.next = engine.next, engine.current
            wall_samples.append((perf_counter() - t) * 1000)
        t = perf_counter()
        events[count - 1][1].synchronize()
        sync_share = (perf_counter() - t) * 1000 / count
        m.durations_ms.extend(v + sync_share for v in wall_samples)
        m.kernel_durations_ms.extend(float(cuda.event_elapsed_time(*events[i])) for i in range(count))

    def frames_now():
        frames = {}
        for key, engine in engines.items():
            if key == "cpu":
                frames[key] = engine.current.copy()
            else:
                t = perf_counter()
                frames[key] = engine.result()
                metrics[key].preview_transfer_ms += (perf_counter() - t) * 1000
        return frames

    def checkpoint():
        return control.checkpoint(lambda: snapshot("paused", force=True))

    if settings.display == "live":
        for key in selected:
            if not checkpoint():
                break
            setup(key)
        for generation in range(settings.iterations):
            if not checkpoint():
                break
            # Alternate within each generation; both grids are published at the same generation.
            for key in selected if generation % 2 == 0 else selected[::-1]:
                step_cpu() if key == "cpu" else step_gpu(1)
            snapshot(force=True, frames=frames_now())
            if generation + 1 < settings.iterations:
                control.pace(.06)
    else:
        # Sequential runs avoid CPU baseline / GPU launch-thread contention.
        # No intermediate grid copies or canvas/graph drawing in benchmark mode.
        for key in selected:
            if not checkpoint():
                break
            setup(key)
            while len(metrics[key].durations_ms) < settings.iterations:
                if not checkpoint():
                    break
                if key == "cpu":
                    step_cpu()
                else:
                    step_gpu(min(32, settings.iterations - len(metrics[key].durations_ms)))
                snapshot()

    for key, engine in engines.items():
        t = perf_counter()
        if key == "cpu":
            final = engine.current.copy()
        else:
            final = np.empty_like(initial)
            copy_start = perf_counter()
            engine.result(final)
            metrics[key].d2h_ms = (perf_counter() - copy_start) * 1000
        output_ms = (perf_counter() - t) * 1000
        m = metrics[key]
        m.output_ms = output_ms
        m.final_alive = int(final.sum())
        m.final_sha256 = hashlib.sha256(final.tobytes()).hexdigest()
        finals[key] = final
    complete = (len(metrics) == len(selected)
                and all(len(m.durations_ms) == settings.iterations for m in metrics.values()))
    state = "complete" if complete else "stopped"
    summaries = {k: m.summary(include_samples=True) for k, m in metrics.items()}
    match, speedup = compare_results(settings, summaries, finals, state)
    report = {"schema_version": 2, "created_at": datetime.now(timezone.utc).isoformat(),
              "settings": settings.to_dict(), "hardware": hardware, "status": state,
              "initial_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
              "engines": summaries, "results_match": match, "speedup": speedup,
              "methodology": METHODOLOGY}
    publish({"state": state, "engines": summaries, "frames": finals, "report": report})
    return report, finals
