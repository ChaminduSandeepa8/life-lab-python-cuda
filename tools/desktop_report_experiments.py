"""Collect repeatable evidence from the actual desktop controller without opening Tk.

python -m tools.desktop_report_experiments
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time
import numpy as np
from desktop.models import Settings
from desktop.runner import execute, hardware_info

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "desktop-report"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hardware = hardware_info()
    if not hardware["cuda"]["available"]:
        raise RuntimeError("This report requires a real CUDA GPU")
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "hardware": hardware,
              "iterations": 100, "repeats": 3, "runs": [], "primary": [], "blocks": [], "edges": [],
              "source_hashes": {}, "method": "desktop.runner.execute, no GUI; separate engine runs; cyclic order per repetition; medians of three runs; full array equality checked per repeat"}
    for folder in ("life", "desktop"):
        for path in sorted((ROOT / folder).glob("*.py")):
            report["source_hashes"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    def save():
        (OUT / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    def trial(size, engine, repeat, block=16, boundary="wrap", study="primary"):
        settings = Settings(size=size, iterations=100, mode="cuda" if engine == "cuda" else "cpu",
                            cpu_backend="numpy" if engine == "numpy" else "python", display="benchmark",
                            block=block, boundary=boundary)
        started = time.perf_counter()
        result, grids = execute(settings, hardware=hardware)
        key = "cuda" if engine == "cuda" else "cpu"
        report["runs"].append({"study": study, "repeat": repeat, "engine": engine, "report": result})
        save()
        print(f"{study} {size} {boundary} {block}x{block} repeat {repeat} {engine}: "
              f'{result["engines"][key]["total_ms"]:.3f} ms; wall {time.perf_counter()-started:.1f}s', flush=True)
        return result["engines"][key], grids[key], result["initial_sha256"]
    def aggregate(samples):
        fields = ("total_ms", "compute_ms", "setup_ms", "output_ms", "average_ms", "min_ms", "max_ms",
                  "cells_per_second", "h2d_ms", "d2h_ms", "transfer_ms", "preview_transfer_ms", "kernel_ms")
        out = {}
        for field in fields:
            values = [s[field] for s in samples if s[field] is not None]
            out[field] = statistics.median(values) if values else None
        out.update(total_ms_min=min(s["total_ms"] for s in samples), total_ms_max=max(s["total_ms"] for s in samples),
                   grid_bytes=samples[0]["grid_bytes"], buffer_bytes=samples[0]["buffer_bytes"],
                   final_alive=samples[0]["final_alive"], final_sha256=samples[0]["final_sha256"])
        return out
    names = ["python", "numpy", "cuda"]
    for size in (64, 128, 256, 512, 1024):
        samples = {name: [] for name in names}
        match = True
        for repeat in range(3):
            grids, inputs = {}, []
            for engine in names[repeat:] + names[:repeat]:
                data, grids[engine], initial_hash = trial(size, engine, repeat+1)
                samples[engine].append(data)
                inputs.append(initial_hash)
            equal = len(set(inputs)) == 1 and np.array_equal(grids["python"], grids["numpy"]) and np.array_equal(grids["python"], grids["cuda"])
            match = match and equal
            if not equal:
                raise AssertionError(f"Mismatch at {size}, repetition {repeat+1}")
        row = {"size": size, "verified": match, **{k: aggregate(v) for k,v in samples.items()}}
        row["speedup_python"] = row["python"]["total_ms"] / row["cuda"]["total_ms"]
        row["speedup_numpy"] = row["numpy"]["total_ms"] / row["cuda"]["total_ms"]
        report["primary"].append(row)
        save()
        print(f"COMPLETE primary {size}: Python speedup {row['speedup_python']:.2f}x; NumPy speedup {row['speedup_numpy']:.2f}x", flush=True)
    for block in (8, 16, 32):
        samples = []
        hashes = []
        for repeat in range(3):
            data, grid, _ = trial(1024, "cuda", repeat+1, block=block, study="blocks")
            samples.append(data)
            hashes.append(hashlib.sha256(grid.tobytes()).hexdigest())
        report["blocks"].append({"block": block, **aggregate(samples), "verified": all(h == report["primary"][-1]["numpy"]["final_sha256"] for h in hashes)})
        save()
    for boundary in ("wrap", "dead"):
        samples = {"numpy": [], "cuda": []}
        matched = True
        for repeat in range(3):
            grids = {}
            for engine in (("numpy", "cuda") if repeat % 2 == 0 else ("cuda", "numpy")):
                data, grids[engine], _ = trial(512, engine, repeat+1, boundary=boundary, study="edges")
                samples[engine].append(data)
            matched = matched and bool(np.array_equal(grids["numpy"], grids["cuda"]))
        report["edges"].append({"boundary": boundary, "verified": matched, **{k: aggregate(v) for k,v in samples.items()}})
        save()
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    save()
    print("Saved " + str(OUT / "benchmark.json"), flush=True)


if __name__ == "__main__":
    main()
