"""Run the desktop Python/CUDA engines without a graphical display."""
import argparse
from pathlib import Path

from desktop.models import Settings
from desktop.runner import execute
from desktop.export import export_csv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "both", "cuda"), default="cpu")
    parser.add_argument("--size", type=int, choices=(64, 128, 256, 512, 1024), default=64)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--density", type=float, default=.25)
    parser.add_argument("--boundary", choices=("wrap", "dead"), default="wrap")
    parser.add_argument("--block", type=int, choices=(8, 16, 32), default=16)
    parser.add_argument("--output", type=Path, default=Path("results/experiment/statistics.csv"))
    args = parser.parse_args(argv)
    try:
        settings = Settings(mode=args.mode, size=args.size, iterations=args.iterations,
                            seed=args.seed, density=args.density, boundary=args.boundary,
                            block=args.block, display="benchmark", cpu_backend="python")
        report, _ = execute(settings)
        paths = export_csv(report, args.output)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Experiment failed: {exc}\n")
    for engine, measurement in report["engines"].items():
        print(f"{engine}: {measurement['generations']} generations, {measurement['total_ms']:.3f} ms active time")
    print(f"Results match: {report['results_match']}; speedup: {report['speedup']}")
    for path in paths:
        print(f"Saved {path}")
    return 2 if report["results_match"] is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
