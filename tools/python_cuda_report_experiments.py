"""Report evidence from actual Python loops and CUDA only. No vectorized CPU trials."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
from desktop.models import Settings
from desktop.runner import execute, hardware_info

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "python-cuda-report"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hardware = hardware_info()
    if not hardware["cuda"]["available"]:
        raise RuntimeError("A real CUDA GPU is required")
    data = {"created_at": datetime.now(timezone.utc).isoformat(), "hardware": hardware,
            "iterations": 100, "repeats": 3, "runs": [], "primary": [], "blocks": [], "dead": None,
            "source_hashes": {}, "method": "Actual desktop runner, no GUI; PythonEngine versus CUDAEngine only; alternating order; medians of three runs; full final bytes compared each repetition."}
    for folder in ("life", "desktop"):
        for path in sorted((ROOT / folder).glob("*.py")):
            data["source_hashes"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    def save():
        temp = OUT / "benchmark.tmp"
        temp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        temp.replace(OUT / "benchmark.json")
    def trial(size, engine, repetition, block=16, boundary="wrap", study="primary"):
        settings = Settings(size=size, iterations=100, mode="cpu" if engine == "python" else "cuda",
                            cpu_backend="python", display="benchmark", block=block, boundary=boundary)
        progress = [0]
        def publish(packet):
            item = packet.get("engines", {}).get("cpu", {})
            n = item.get("generations", 0)
            if n >= progress[0] + 25:
                progress[0] = n
                print(f"Progress {study} {size} Python repeat {repetition}: {n}/100", flush=True)
        result, grids = execute(settings, publish=publish, hardware=hardware)
        key = "cpu" if engine == "python" else "cuda"
        sample = result["engines"][key]
        assert sample["backend"] == engine and result["status"] == "complete"
        data["runs"].append({"study": study, "repeat": repetition, "engine": engine, "report": result})
        save()
        print(f"DONE {study} {size} {boundary} {block}x{block} repeat {repetition} {engine}: {sample['total_ms']:.3f} ms", flush=True)
        return sample, grids[key], result["initial_sha256"]
    def aggregate(samples):
        out = {}
        fields = ("total_ms", "compute_ms", "setup_ms", "output_ms", "average_ms", "min_ms", "max_ms",
                  "cells_per_second", "h2d_ms", "d2h_ms", "transfer_ms", "kernel_ms", "preview_transfer_ms")
        for field in fields:
            values = [s[field] for s in samples if s[field] is not None]
            out[field] = statistics.median(values) if values else None
        out.update(total_ms_min=min(s["total_ms"] for s in samples), total_ms_max=max(s["total_ms"] for s in samples),
                   generation_min_ms=min(min(s["generation_ms"]) for s in samples),
                   generation_max_ms=max(max(s["generation_ms"]) for s in samples),
                   grid_bytes=samples[0]["grid_bytes"], buffer_bytes=samples[0]["buffer_bytes"],
                   final_alive=samples[0]["final_alive"], final_sha256=samples[0]["final_sha256"])
        return out
    def compare(size, boundary="wrap", study="primary"):
        samples = {"python": [], "cuda": []}
        for repeat in range(1, 4):
            finals, hashes = {}, []
            for engine in (("python", "cuda") if repeat % 2 else ("cuda", "python")):
                sample, finals[engine], input_hash = trial(size, engine, repeat, boundary=boundary, study=study)
                samples[engine].append(sample)
                hashes.append(input_hash)
            if hashes[0] != hashes[1] or finals["python"].tobytes() != finals["cuda"].tobytes():
                raise AssertionError(f"CPU/CUDA mismatch at {size} {boundary} repeat {repeat}")
        row = {"size": size, "boundary": boundary, "verified": True,
               **{key: aggregate(values) for key, values in samples.items()}}
        row["speedup"] = row["python"]["total_ms"] / row["cuda"]["total_ms"]
        return row
    for size in (64, 128, 256, 512, 1024):
        row = compare(size)
        data["primary"].append(row)
        save()
        print(f"VERIFIED {size}: {row['speedup']:.2f}x", flush=True)
    for block in (8, 16, 32):
        samples = []
        for repeat in range(1, 4):
            sample, final, _ = trial(1024, "cuda", repeat, block=block, study="blocks")
            assert hashlib.sha256(final.tobytes()).hexdigest() == data["primary"][-1]["python"]["final_sha256"]
            samples.append(sample)
        data["blocks"].append({"block": block, "verified": True, **aggregate(samples)})
        save()
    data["dead"] = compare(256, boundary="dead", study="dead")
    data["completed_at"] = datetime.now(timezone.utc).isoformat()
    save()
    print("COMPLETE " + str(OUT / "benchmark.json"), flush=True)


if __name__ == "__main__":
    main()
