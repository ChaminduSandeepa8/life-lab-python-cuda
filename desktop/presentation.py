"""One statistics definition for the window, CSV and saved figures."""
from .models import CPU_LABELS

CPU_COLOR = "#14b8a6"
GPU_COLOR = "#f59e0b"


def fmt(value, digits=3):
    return "—" if value is None else f"{value:,.{digits}f}"


def stats_rows(engines, report=None):
    cpu, gpu = engines.get("cpu", {}), engines.get("cuda", {})
    rows = []
    def add(label, field, digits=3):
        rows.append((label, fmt(cpu.get(field), digits), fmt(gpu.get(field), digits)))
    add("Completed generations", "generations", 0)
    add("Total active time (ms)", "total_ms")
    add("Average ms / generation¹", "average_ms")
    add("Min / generation (compute ms)", "min_ms")
    add("Max / generation (compute ms)", "max_ms")
    add("Throughput (cells / second)", "cells_per_second", 0)
    speedup = report.get("speedup") if report else None
    rows.append(("Speedup (CPU total / GPU total)", "1.00×" if speedup else "—",
                 f"{speedup:.2f}×" if speedup else "—"))
    rows.append(("GPU transfer: H2D + final D2H (ms)", "N/A", fmt(gpu.get("transfer_ms"))))
    rows.append(("GPU kernel events (ms)²", "N/A", fmt(gpu.get("kernel_ms"))))
    rows.append(("Live preview transfers (ms)³", "N/A", fmt(gpu.get("preview_transfer_ms"))))
    add("One grid (bytes)", "grid_bytes", 0)
    add("Engine buffers (RAM / VRAM bytes)⁴", "buffer_bytes", 0)
    add("Final alive cells", "final_alive", 0)
    match = report.get("results_match") if report else None
    status = "MATCH ✓" if match is True else "MISMATCH" if match is False else "Not compared"
    rows.append(("Full final-grid equality", status, status))
    return rows


def draw_plot(axes, engines, unit="ms", dark=True):
    axes.clear()
    background, foreground, grid = ("#111e32", "#b8c9df", "#2b3b52") if dark else ("white", "#24334c", "#dbe3ed")
    axes.set_facecolor(background)
    axes.figure.set_facecolor(background)
    scale = 1000 if unit == "s" else 1
    for key, color, label in (("cpu", CPU_COLOR, "CPU"), ("cuda", GPU_COLOR, "CUDA")):
        points = engines.get(key, {}).get("series", [])
        if points:
            axes.plot([p[1] / scale for p in points], [p[2] * scale for p in points],
                      color=color, linewidth=1.4, marker="." if len(points) < 3 else None, label=label)
    axes.set_xlabel(f"Cumulative active compute time ({unit})", color=foreground, fontsize=8)
    axes.set_ylabel(f"Iterations / {unit}", color=foreground, fontsize=8)
    axes.tick_params(colors=foreground, labelsize=8)
    for spine in axes.spines.values():
        spine.set_color(grid)
    axes.grid(alpha=.35, color=grid)
    axes.set_xlim(left=0)
    axes.set_ylim(bottom=0)
    if any(m.get("series") for m in engines.values()):
        legend = axes.legend(loc="upper right", fontsize=8, facecolor=background, edgecolor=grid)
        for text in legend.get_texts():
            text.set_color(foreground)


def cpu_label(settings):
    return CPU_LABELS[settings["cpu_backend"]]
