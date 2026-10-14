"""Save numeric evidence and publication-ready graphs without screen capture."""
import csv
from datetime import datetime
import json
from pathlib import Path
import textwrap
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from .presentation import draw_plot, stats_rows, cpu_label


def export_csv(report, path):
    path = Path(path)
    if path.suffix.lower() != ".csv":
        path = path.with_suffix(".csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Metric", cpu_label(report["settings"]), "CUDA"])
        writer.writerows(stats_rows(report["engines"], report))
        writer.writerow([])
        writer.writerow(["Run status", report["status"]])
        for key, value in report["settings"].items():
            writer.writerow(["setting." + key, value])
        for key in ("cpu", "cuda_runtime", "python", "os", "nvidia_smi"):
            writer.writerow(["hardware." + key, report["hardware"].get(key)])
        writer.writerow(["initial_sha256", report["initial_sha256"]])
        writer.writerow(["methodology", report["methodology"]])
    samples_path = path.with_name(path.stem + "_generations.csv")
    with samples_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["engine", "generation", "compute_ms", "cumulative_compute_ms",
                         "iterations_per_ms", "gpu_kernel_event_ms"])
        for key, data in report["engines"].items():
            cumulative = 0.0
            for i, duration in enumerate(data["generation_ms"]):
                cumulative += duration
                kernel = data["kernel_generation_ms"][i] if key == "cuda" else ""
                writer.writerow([key, i + 1, duration, cumulative, 1/duration if duration else "", kernel])
    json_path = path.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return [path, samples_path, json_path]


def export_bundle(report, folder, unit="ms"):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    destination = folder / f"life-lab-{stamp}"
    destination.mkdir()
    files = export_csv(report, destination / "statistics.csv")
    chart = Figure(figsize=(10, 4.8), layout="constrained")
    FigureCanvasAgg(chart)
    axes = chart.subplots()
    draw_plot(axes, report["engines"], unit, dark=False)
    axes.set_title("Conway's Game of Life · CPU / CUDA", loc="left", pad=16)
    for ext in ("png", "svg"):
        name = destination / f"throughput.{ext}"
        chart.savefig(name, dpi=180)
        files.append(name)

    page = Figure(figsize=(11.7, 14.5), facecolor="white")
    FigureCanvasAgg(page)
    page.text(.07, .96, "LIFE LAB / PERFORMANCE COMPARISON", fontsize=18, weight="bold", color="#17253d")
    settings = report["settings"]
    subtitle = (f'{settings["size"]} × {settings["size"]} · {settings["iterations"]} requested generations · '
                f'{settings["display"]} · {report["status"]}\n'
                f'{cpu_label(settings)} vs CUDA · {settings["block"]} × {settings["block"]} threads\n'
                f'{settings["pattern"]} · seed {settings["seed"]} · density {settings["density"]:.0%} · '
                f'{settings["boundary"]} edges')
    page.text(.07, .903, subtitle, fontsize=10, linespacing=1.6)
    axes = page.add_axes((.10, .64, .83, .22))
    draw_plot(axes, report["engines"], unit, dark=False)
    table_axes = page.add_axes((.07, .28, .86, .30))
    table_axes.axis("off")
    rows = stats_rows(report["engines"], report)
    # ASCII substitutions keep export fonts portable across machines.
    rows = [[str(v).replace("✓", "yes").replace("¹", " [1]").replace("²", " [2]")
             .replace("³", " [3]").replace("⁴", " [4]") for v in row] for row in rows]
    table = table_axes.table(cellText=rows, colLabels=["Metric", "CPU", "CUDA"],
                            colWidths=[.57, .215, .215], bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#dbe3ed")
        if row == 0:
            cell.set_facecolor("#17253d")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f1f5f9" if row % 2 else "white")
        if col == 0:
            cell.set_text_props(ha="left")
    hardware = report["hardware"]
    info = (f'CPU: {hardware.get("cpu", "Unknown")}\n'
            f'GPU: {hardware.get("cuda", {}).get("name", "Unavailable")}  |  '
            f'CUDA runtime: {hardware.get("cuda_runtime", "Unknown")}\n'
            f'Python {hardware.get("python", "Unknown")}  |  Measured {report["created_at"]}')
    page.text(.07, .219, info, fontsize=9, linespacing=1.6)
    notes = ("[1] Average uses total; min/max use generation compute samples. "
             "[2] CUDA event intervals include possible dispatch gaps. "
             "[3] Preview copies and pauses are excluded from total. "
             "[4] Engine array payload, not total process or driver memory. "
             "Benchmark GPU wall time assigns batch synchronization equally to its generations. "
             "Single-run instrumented results; repeat benchmarks before drawing general conclusions. "
             "Both final grids must match before speedup is reported. Raw samples and full timing definitions accompany this figure.")
    page.text(.07, .085, textwrap.fill(notes, 128), fontsize=8, linespacing=1.5, color="#42526b")
    for ext in ("png", "pdf"):
        name = destination / f"comparison.{ext}"
        page.savefig(name, dpi=160)
        files.append(name)
    (destination / "methodology.txt").write_text(report["methodology"] + "\n", encoding="utf-8")
    return destination
