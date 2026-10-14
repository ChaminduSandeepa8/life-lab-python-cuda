"""Repeated, reproducible experiments and portable JSON/CSV evidence."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import numpy as np
from .core import Config, make_grid
from .cpu import CPUEngine, measure_cpu
from .gpu import cuda_status, warmup_cuda, measure_cuda

def environment():
    packages = {}
    for name in ("numpy", "numba", "numba-cuda"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                cpu = line.split(':', 1)[1].strip()
                break
    except OSError:
        pass
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                 "--format=csv,noheader"], capture_output=True, text=True,
                                timeout=5).stdout.strip() or "Unavailable"
    except (OSError, subprocess.TimeoutExpired):
        driver = "Unavailable"
    return {"python": platform.python_version(), "os": platform.platform(),
            "cpu": cpu, "logical_cpus": os.cpu_count(), "packages": packages,
            "nvidia_smi": driver, "cuda": cuda_status()}

def summarize(samples):
    result = {"samples": samples}
    for field in ("compute_ms", "total_ms"):
        values = [s[field] for s in samples]
        result[field] = statistics.median(values)
        result[field + "_min"] = min(values)
        result[field + "_max"] = max(values)
    return result

def benchmark(config, mode="both"):
    if mode not in ("cpu", "both", "cuda"):
        raise ValueError("Mode must be cpu, both or cuda")
    initial = make_grid(config)
    block = (config.block_x, config.block_y)
    status = cuda_status()
    if mode == "cuda" and not status["available"]:
        raise RuntimeError(status["reason"])
    result = {"config": config.to_dict(), "mode": mode, "cpu": None, "cuda": None,
              "verified": None, "speedup_compute": None, "speedup_total": None,
              "initial_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
              "cuda_status": status, "note": "", "final_alive": None}
    # Warmups are excluded. A failed GPU warmup never becomes an emulated result.
    CPUEngine(initial, config.boundary).advance(1)
    use_gpu = mode != "cpu" and status["available"]
    if use_gpu:
        try:
            warmup_cuda(initial, config.boundary, block)
        except Exception as exc:
            raise RuntimeError(f"CUDA warmup/compilation failed: {exc}") from exc
    elif mode != "cpu":
        result["note"] = status["reason"]
    cpu_samples, gpu_samples = [], []
    final_cpu = final_gpu = None
    verified = True
    for repeat in range(config.repeats):
        # Alternate order to reduce a fixed-order thermal/cache bias.
        order = ("cpu", "cuda") if repeat % 2 == 0 else ("cuda", "cpu")
        for backend in order:
            if backend == "cpu" and mode != "cuda":
                final_cpu, times = measure_cpu(initial, config.iterations, config.boundary)
                cpu_samples.append(times)
            elif backend == "cuda" and use_gpu:
                final_gpu, times = measure_cuda(initial, config.iterations, config.boundary, block)
                gpu_samples.append(times)
        if final_cpu is not None and final_gpu is not None:
            verified = verified and bool(np.array_equal(final_cpu, final_gpu))
    if cpu_samples:
        result["cpu"] = summarize(cpu_samples)
    if gpu_samples:
        result["cuda"] = summarize(gpu_samples)
    if cpu_samples and gpu_samples:
        result["verified"] = verified
        if not verified:
            result["note"] = "CPU/CUDA mismatch: speedups withheld; investigate before using results."
        else:
            for metric in ("compute", "total"):
                denom = result["cuda"][metric + "_ms"]
                result["speedup_" + metric] = result["cpu"][metric + "_ms"] / denom if denom > 0 else None
    final = final_cpu if final_cpu is not None else final_gpu
    result["final_alive"] = int(final.sum())
    result["final_sha256"] = hashlib.sha256(final.tobytes()).hexdigest()
    return result

def suite(configs, mode="both", progress=None):
    results = []
    for index, config in enumerate(configs):
        if progress:
            progress(index, len(configs), config.size)
        results.append(benchmark(config, mode))
    return {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "environment": environment(), "results": results,
            "methodology": "Warmup excluded; identical initial grid per repetition; median of repeated runs. "
            "CPU compute excludes setup. CUDA compute uses events around the launch sequence. "
            "Total includes setup and output copies; CUDA total includes H2D/D2H. No rendering. "
            "Simulator timings are never reported."}

CSV_FIELDS = ["size", "iterations", "repeats", "boundary", "seed", "density", "pattern",
              "cpu_compute_ms", "cpu_total_ms", "cuda_compute_ms", "cuda_total_ms",
              "speedup_compute", "speedup_total", "verified", "note"]

def write_csv(report, stream):
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for row in report["results"]:
        flat = {k: row["config"][k] for k in CSV_FIELDS if k in row["config"]}
        for engine in ("cpu", "cuda"):
            for metric in ("compute_ms", "total_ms"):
                flat[engine + "_" + metric] = row[engine][metric] if row[engine] else ""
        flat.update({key: row[key] for key in ("speedup_compute", "speedup_total", "verified", "note")})
        writer.writerow(flat)

def save_report(report, output):
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    with (folder / "benchmark.csv").open("w", newline="") as stream:
        write_csv(report, stream)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[128,256,512,1024])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--density", type=float, default=.25)
    parser.add_argument("--boundary", choices=["wrap", "dead"], default="wrap")
    parser.add_argument("--block", choices=["8x8", "16x16", "32x8", "32x16", "32x32"], default="32x8")
    parser.add_argument("--mode", choices=["cpu", "cuda", "both"], default="both")
    parser.add_argument("--output", default="results/latest")
    args = parser.parse_args()
    bx, by = map(int, args.block.split('x'))
    try:
        configs = [Config(size=n, iterations=args.iterations, repeats=args.repeats, seed=args.seed,
                          density=args.density, boundary=args.boundary, block_x=bx, block_y=by)
                   for n in args.sizes]
        report = suite(configs, args.mode, lambda i,t,n: print(f"[{i+1}/{t}] {n} x {n}", flush=True))
        save_report(report, args.output)
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    for row in report["results"]:
        cpu = f'{row["cpu"]["compute_ms"]:.3f} ms' if row['cpu'] else 'not run'
        gpu = f'{row["cuda"]["compute_ms"]:.3f} ms' if row['cuda'] else 'unavailable'
        print(f'{row["config"]["size"]}: CPU {cpu}; CUDA {gpu}; verified={row["verified"]}')
    print(f"Saved JSON and CSV to {args.output}")
    if any(r["verified"] is False for r in report["results"]):
        raise SystemExit(2)

if __name__ == "__main__":
    main()
