"""Two-window Tk application; UI state stays on the main thread."""
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from life.core import make_grid
from .export import export_bundle, export_csv
from .models import Settings, PATTERNS, CPU_LABELS
from .presentation import CPU_COLOR, GPU_COLOR, draw_plot, stats_rows
from .runner import Control, execute, hardware_info
from .widgets import BG, PANEL, TEXT, MUTED, GridCanvas, ScrollableFrame

ROOT = Path(__file__).resolve().parents[1]


def theme(root):
    root.configure(bg=BG)
    root.option_add("*Font", ("DejaVu Sans", 10))
    root.option_add("*TCombobox*Listbox.background", PANEL)
    root.option_add("*TCombobox*Listbox.foreground", TEXT)
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", background=BG, foreground=TEXT, font=("DejaVu Sans", 10))
    style.configure("TFrame", background=BG)
    style.configure("Panel.TFrame", background=PANEL)
    style.configure("TLabel", background=BG, foreground=TEXT)
    style.configure("Panel.TLabel", background=PANEL)
    style.configure("Muted.TLabel", foreground=MUTED, font=("DejaVu Sans", 9))
    style.configure("Small.TLabel", foreground=MUTED, font=("DejaVu Sans", 8))
    style.configure("Title.TLabel", font=("DejaVu Sans", 23, "bold"))
    style.configure("Heading.TLabel", font=("DejaVu Sans", 12, "bold"))
    style.configure("CPU.TLabel", foreground=CPU_COLOR, font=("DejaVu Sans", 12, "bold"))
    style.configure("GPU.TLabel", foreground=GPU_COLOR, font=("DejaVu Sans", 12, "bold"))
    style.configure("TButton", background="#263951", foreground=TEXT, padding=(13, 8), borderwidth=0)
    style.map("TButton", background=[("active", "#36516f"), ("disabled", "#18273b")],
              foreground=[("disabled", "#687d99")])
    style.configure("Accent.TButton", background="#0d9488", foreground="white")
    style.map("Accent.TButton", background=[("active", "#0f766e"), ("disabled", "#183c40")])
    style.configure("Stop.TButton", background="#8c3c45")
    style.configure("TEntry", fieldbackground=PANEL, foreground=TEXT, insertcolor=TEXT, padding=6)
    style.configure("TCombobox", fieldbackground=PANEL, background="#263951", foreground=TEXT, padding=5,
                    arrowcolor=TEXT)
    style.map("TCombobox", fieldbackground=[("readonly", PANEL), ("disabled", BG)],
              foreground=[("readonly", TEXT), ("disabled", "#687d99")], selectbackground=[("readonly", PANEL)])
    style.configure("Horizontal.TScale", background=BG, troughcolor=PANEL)
    style.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=TEXT,
                    rowheight=24, borderwidth=0, font=("DejaVu Sans", 9))
    style.configure("Treeview.Heading", background="#263951", foreground=TEXT, font=("DejaVu Sans", 9, "bold"))
    style.map("Treeview", background=[("selected", "#28435f")])
    style.configure("Horizontal.TProgressbar", troughcolor=PANEL, background=CPU_COLOR, borderwidth=0)


class Application:
    def __init__(self, root):
        self.root = root
        theme(root)
        root.title("Life Lab · Experiment setup")
        root.geometry(f"{min(1220, root.winfo_screenwidth()-60)}x{min(920, root.winfo_screenheight()-90)}+25+25")
        root.minsize(800, 650)
        self.settings = None
        self.preview_grid = None
        self.comparison = None
        self.retired_windows = []
        self.reap_job = None
        self.hardware = None
        self.preview_job = None
        self.hardware_queue = queue.Queue()
        self.hardware_poll = None
        self.variables = {}
        self.controls = {}
        self.closed = False
        self._build()
        self._update_preview()
        root.protocol("WM_DELETE_WINDOW", self.close)
        threading.Thread(target=self._probe, daemon=True, name="hardware-probe").start()
        self._poll_hardware()

    def _build(self):
        header = ttk.Frame(self.root, padding=(24, 17, 24, 10))
        header.pack(fill="x")
        ttk.Label(header, text="LIFE LAB", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="01 / SET UP YOUR EXPERIMENT     •     Python vs NVIDIA CUDA", style="Muted.TLabel").pack(anchor="w", pady=(2, 5))
        self.hardware_text = tk.StringVar(value="Checking CPU and CUDA hardware…")
        ttk.Label(header, textvariable=self.hardware_text, style="Small.TLabel", wraplength=1100).pack(anchor="w")

        footer = ttk.Frame(self.root, padding=(24, 10, 24, 16))
        footer.pack(side="bottom", fill="x")
        self.run_button = ttk.Button(footer, text="Run → Open comparison", style="Accent.TButton", command=self.open_comparison)
        self.run_button.pack(side="right")
        self.validation_text = tk.StringVar()
        ttk.Label(footer, textvariable=self.validation_text, style="Muted.TLabel", wraplength=670).pack(side="left")

        body = ScrollableFrame(self.root)
        body.pack(fill="both", expand=True, padx=24)
        form = ttk.Frame(body.content, padding=(0, 5, 0, 6))
        form.pack(fill="x")
        for col in range(4):
            form.columnconfigure(col, weight=1, uniform="settings")

        def field(key, label, value, options=None, hint="", slider=None):
            index = len(self.variables)
            frame = ttk.Frame(form, padding=(0, 4, 14, 8))
            frame.grid(row=index//4, column=index % 4, sticky="nsew")
            var = tk.StringVar(value=str(value))
            self.variables[key] = var
            ttk.Label(frame, text=label).pack(anchor="w", pady=(0, 4))
            if slider:
                line = ttk.Frame(frame)
                line.pack(fill="x")
                control = ttk.Scale(line, from_=slider[0], to=slider[1],
                                    command=lambda v: var.set(str(round(float(v)))))
                control.set(float(value))
                control.pack(side="left", fill="x", expand=True)
                ttk.Label(line, textvariable=var, width=3).pack(side="right")
            elif options:
                control = ttk.Combobox(frame, textvariable=var, values=options, state="readonly", width=17)
                control.pack(fill="x")
            else:
                control = ttk.Entry(frame, textvariable=var, width=17)
                control.pack(fill="x")
            self.controls[key] = control
            ttk.Label(frame, text=hint, style="Small.TLabel", wraplength=250).pack(anchor="w", pady=(4, 0))
            var.trace_add("write", self._schedule_preview)

        field("size", "Grid size", 64, [64, 128, 256, 512, 1024], "Width × height; one byte per cell")
        field("iterations", "Iterations", 100, hint="1–100,000; Stop ends a run early")
        field("pattern", "Initial pattern", "Random", list(PATTERNS), "The same input goes to both engines")
        field("density", "Random density (%)", 25, slider=(10, 50), hint="Used only for the Random pattern")
        field("seed", "Random seed", 42, hint="Repeatable input for a fair comparison")
        field("boundary", "Edge behavior", "Wrap-around", ["Wrap-around", "Dead edges"], "Opposite sides join / outside stays dead")
        field("mode", "Run mode", "Both", ["Both", "CPU only", "GPU only"], "GPU modes require an NVIDIA CUDA device")
        field("display", "Display mode", "Live view", ["Live view", "Benchmark (no drawing)"], "Live is paced; Benchmark skips drawing")
        field("cell_size", "Cell size (px)", 5, slider=(1, 12), hint="Scroll the canvas to inspect large grids")
        field("block", "CUDA block size", "16×16", ["8×8", "16×16", "32×32"], "64 / 256 / 1,024 threads per block")
        field("cpu_backend", "CPU implementation", "Normal Python", ["Normal Python", "NumPy optimized"], "Nested loops / vectorized 2D arrays")
        legend = ttk.Frame(form, padding=(0, 4, 14, 8))
        legend.grid(row=2, column=3, sticky="nsew")
        ttk.Label(legend, text="B3 / S23", style="Heading.TLabel").pack(anchor="w")
        ttk.Label(legend, text="Born with 3 neighbors.\nSurvives with 2 or 3.\nAll cells update together.", style="Small.TLabel").pack(anchor="w", pady=8)

        preview_header = ttk.Frame(body.content, padding=(0, 10, 0, 7))
        preview_header.pack(fill="x")
        ttk.Label(preview_header, text="Starting universe", style="Heading.TLabel").pack(side="left")
        self.preview_text = tk.StringVar()
        ttk.Label(preview_header, textvariable=self.preview_text, style="Muted.TLabel").pack(side="right")
        self.preview = GridCanvas(body.content, height=300)
        self.preview.pack(fill="both", expand=True, pady=(0, 12))

    def read_settings(self):
        v = {k: item.get() for k, item in self.variables.items()}
        return Settings(size=int(v["size"]), iterations=int(v["iterations"]), pattern=PATTERNS[v["pattern"]],
                        density=int(v["density"])/100, seed=int(v["seed"]),
                        boundary="wrap" if v["boundary"] == "Wrap-around" else "dead",
                        mode={"Both": "both", "CPU only": "cpu", "GPU only": "cuda"}[v["mode"]],
                        display="live" if v["display"] == "Live view" else "benchmark",
                        cell_size=int(v["cell_size"]), block=int(v["block"].split("×")[0]),
                        cpu_backend="python" if v["cpu_backend"] == "Normal Python" else "numpy")

    def _schedule_preview(self, *_):
        if self.preview_job:
            self.root.after_cancel(self.preview_job)
        # Invalid/incomplete input cannot launch the previous valid configuration.
        self.run_button.state(["disabled"])
        self.preview_job = self.root.after(140, self._update_preview)

    def _update_preview(self):
        self.preview_job = None
        try:
            settings = self.read_settings()
        except (ValueError, KeyError) as exc:
            self.settings = None
            self.validation_text.set(str(exc) if str(exc) else "Check the settings.")
            self.run_button.state(["disabled"])
            return
        self.settings = settings
        self.preview_grid = make_grid(settings.grid_config())
        self.preview.set_grid(self.preview_grid, settings.cell_size, reset=True)
        self.preview_text.set(f"{settings.size} × {settings.size}   ·   {int(self.preview_grid.sum()):,} alive   ·   Generation 0")
        random = settings.pattern == "random"
        self.controls["density"].state(["!disabled"] if random else ["disabled"])
        self.controls["seed"].state(["!disabled"] if random else ["disabled"])
        self.controls["block"].configure(state="disabled" if settings.mode == "cpu" else "readonly")
        self.controls["cpu_backend"].configure(state="disabled" if settings.mode == "cuda" else "readonly")
        if settings.mode != "cpu" and self.hardware is None:
            self.validation_text.set("Detecting CUDA… CPU only is available immediately.")
            self.run_button.state(["disabled"])
        elif settings.mode != "cpu" and not self.hardware["cuda"]["available"]:
            self.validation_text.set("CUDA unavailable. Select CPU only, or follow the CUDA setup in docs/INSTALL.md.")
            self.run_button.state(["disabled"])
        else:
            hint = "Ready. Run opens a separate window; press Start there."
            if settings.cpu_backend == "python" and settings.size >= 512 and settings.mode != "cuda":
                hint = "Large Python-loop runs take time. Pause / Stop remain available."
            self.validation_text.set(hint)
            self.run_button.state(["!disabled"])

    def _probe(self):
        try:
            self.hardware_queue.put(hardware_info())
        except Exception as exc:
            self.hardware_queue.put({"cpu": "Unknown", "cuda": {"available": False, "reason": str(exc)},
                                     "cuda_runtime": "Unavailable"})

    def _poll_hardware(self):
        try:
            self.hardware = self.hardware_queue.get_nowait()
        except queue.Empty:
            self.hardware_poll = self.root.after(100, self._poll_hardware)
            return
        self.hardware_poll = None
        gpu = self.hardware["cuda"]
        self.hardware_text.set(f'CPU: {self.hardware["cpu"]}     |     GPU: {gpu.get("name", "Unavailable")}     |     CUDA: {self.hardware["cuda_runtime"]}')
        # A fresh CPU-only installation should be usable immediately. Explicit GPU
        # requests made after detection still fail clearly instead of being emulated.
        if not gpu["available"] and self.variables["mode"].get() == "Both":
            self.variables["mode"].set("CPU only")
        if self.preview_job:
            self.root.after_cancel(self.preview_job)
        self._update_preview()

    def open_comparison(self):
        # Revalidate synchronously in case the debounce has not fired yet.
        try:
            settings = self.read_settings()
        except (ValueError, KeyError) as exc:
            messagebox.showerror("Check settings", str(exc), parent=self.root)
            return
        if self.comparison and self.comparison.window.winfo_exists():
            if self.comparison.is_running():
                self.comparison.window.lift()
                self.validation_text.set("Stop or finish the current comparison before opening another.")
                return
            self.comparison.close()
        self.comparison = ComparisonWindow(self, settings)

    def close(self):
        self.closed = True
        if self.comparison:
            self.comparison.close()
        for job in (self.preview_job, self.hardware_poll, self.reap_job):
            if job:
                self.root.after_cancel(job)
        self.root.destroy()

    def retire(self, window):
        # Retain Tk-owning objects until their worker exits; release on the UI thread.
        self.retired_windows.append(window)
        if not self.closed and self.reap_job is None:
            self.reap_job = self.root.after(100, self._reap)

    def _reap(self):
        self.retired_windows = [window for window in self.retired_windows if window.is_running()]
        self.reap_job = self.root.after(100, self._reap) if self.retired_windows else None


class ComparisonWindow:
    def __init__(self, app, settings):
        self.app, self.settings = app, settings
        self.window = tk.Toplevel(app.root)
        self.window.title("Life Lab · CPU / CUDA comparison")
        self.window.configure(bg=BG)
        self.window.geometry(f"{min(1270, app.root.winfo_screenwidth()-50)}x{min(1030, app.root.winfo_screenheight()-80)}+40+30")
        self.window.minsize(800, 650)
        self.queue = queue.Queue(maxsize=3)
        self.control = None
        self.worker = None
        self.report = None
        self.latest_engines = {}
        self.closed = False
        self.reset_pending = False
        self.poll_job = None
        self.state = "ready"
        self.run_serial = 0
        self.grids, self.population_labels = {}, {}
        self.initial = make_grid(settings.grid_config())
        self._build()
        self._reset_view()
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self._poll()

    def _build(self):
        header = ttk.Frame(self.window, padding=(22, 14, 22, 9))
        header.pack(fill="x")
        line = ttk.Frame(header)
        line.pack(fill="x")
        ttk.Label(line, text="02 / THE SAME UNIVERSE. TWO ENGINES.", style="Heading.TLabel").pack(side="left")
        self.stop_button = ttk.Button(line, text="Stop", style="Stop.TButton", command=self.stop)
        self.stop_button.pack(side="right", padx=(8, 0))
        self.start_button = ttk.Button(line, text="Start simulation", style="Accent.TButton", command=self.start)
        self.start_button.pack(side="right")
        s = self.settings
        description = (f"{s.size} × {s.size}  ·  {s.iterations:,} generations  ·  {s.pattern}  ·  {s.boundary} edges  ·  "
                       f"{'LIVE VIEW' if s.display == 'live' else 'BENCHMARK / NO DRAWING'}  ·  {s.block}×{s.block} CUDA block")
        ttk.Label(header, text=description, style="Muted.TLabel", wraplength=1100).pack(anchor="w", pady=(7, 0))
        self.status_text = tk.StringVar()
        ttk.Label(header, textvariable=self.status_text, style="Muted.TLabel", wraplength=1100).pack(anchor="w", pady=(5, 5))
        self.progress = ttk.Progressbar(header, maximum=s.iterations * (2 if s.mode == "both" else 1))
        self.progress.pack(fill="x")

        footer = ttk.Frame(self.window, padding=(22, 10, 22, 14))
        footer.pack(side="bottom", fill="x")
        self.pause_button = ttk.Button(footer, text="Pause", command=self.pause)
        self.pause_button.pack(side="left")
        self.reset_button = ttk.Button(footer, text="Reset", command=self.reset)
        self.reset_button.pack(side="left", padx=8)
        self.save_button = ttk.Button(footer, text="Save graphs + stat table", command=self.save_graphs)
        self.save_button.pack(side="right")
        self.csv_button = ttk.Button(footer, text="Export CSV", command=self.save_csv)
        self.csv_button.pack(side="right", padx=8)

        scroll = ScrollableFrame(self.window)
        scroll.pack(fill="both", expand=True, padx=22)
        content = scroll.content
        pair = ttk.Frame(content)
        pair.pack(fill="x", pady=(6, 8))
        for col, key in enumerate(("cpu", "cuda")):
            pair.columnconfigure(col, weight=1, uniform="engine")
            pane = ttk.Frame(pair, padding=(0, 0, 12 if col == 0 else 0, 0))
            pane.grid(row=0, column=col, sticky="nsew")
            title = "CPU / " + ("Normal Python" if s.cpu_backend == "python" else "NumPy optimized") if key == "cpu" else "GPU / NVIDIA CUDA"
            ttk.Label(pane, text=title, style="CPU.TLabel" if key == "cpu" else "GPU.TLabel").pack(anchor="w", pady=(0, 4))
            desc = "Sequential 2D-array update" if key == "cpu" else "Parallel kernel · one thread per cell"
            ttk.Label(pane, text=desc, style="Small.TLabel").pack(anchor="w", pady=(0, 6))
            grid = GridCanvas(pane, CPU_COLOR if key == "cpu" else GPU_COLOR, height=210)
            grid.pack(fill="x")
            self.grids[key] = grid
            label = tk.StringVar()
            self.population_labels[key] = label
            ttk.Label(pane, textvariable=label, style="Small.TLabel").pack(anchor="w", pady=(4, 0))

        plot_header = ttk.Frame(content, padding=(0, 5, 0, 4))
        plot_header.pack(fill="x")
        ttk.Label(plot_header, text="Iteration throughput", style="Heading.TLabel").pack(side="left")
        self.unit = tk.StringVar(value="ms")
        self.unit_control = ttk.Combobox(plot_header, textvariable=self.unit, values=["ms", "s"], state="readonly", width=4)
        self.unit_control.pack(side="right")
        ttk.Label(plot_header, text="Time unit  ", style="Small.TLabel").pack(side="right")
        self.unit.trace_add("write", lambda *_: self.update_plot())
        self.figure = Figure(figsize=(9, 2.15), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=.08, right=.98, top=.95, bottom=.25)
        self.chart = FigureCanvasTkAgg(self.figure, master=content)
        self.chart.get_tk_widget().configure(height=210)
        self.chart.get_tk_widget().pack(fill="x")
        ttk.Label(content, text="Rates use active compute time per engine. Live pacing and drawing are excluded. Benchmark draws after completion.",
                  style="Small.TLabel", wraplength=1100).pack(anchor="w", pady=(3, 10))

        ttk.Label(content, text="Measurements", style="Heading.TLabel").pack(anchor="w", pady=(0, 6))
        self.table = ttk.Treeview(content, columns=("metric", "cpu", "cuda"), show="headings", height=14, selectmode="none")
        for key, label, width in (("metric", "Metric", 440), ("cpu", "CPU", 220), ("cuda", "CUDA", 220)):
            self.table.heading(key, text=label)
            self.table.column(key, width=width, minwidth=120, anchor="w" if key == "metric" else "e")
        self.table.pack(fill="x")
        self.table.tag_configure("alternate", background="#16263d")
        notes = ("¹ Average = total / completed generations. Min/max cover compute only.  ² CUDA event intervals may include dispatch gaps.\n"
                 "³ Preview transfers are excluded from total.  ⁴ Array payload only; excludes Python objects, CUDA context and driver allocations.\n"
                 "Pause / Stop apply after the current generation (or a GPU batch of up to 32). Stopped runs are exported as partial results.")
        ttk.Label(content, text=notes, style="Small.TLabel", wraplength=1120).pack(anchor="w", pady=(8, 10))
        self.hardware_label = ttk.Label(content, style="Small.TLabel", wraplength=1120)
        self.hardware_label.pack(anchor="w", pady=(0, 12))
        self._show_hardware(self.app.hardware)

    def _show_hardware(self, info):
        if info:
            self.hardware_label.configure(text=f'CPU: {info.get("cpu", "Unknown")}\nGPU: {info.get("cuda", {}).get("name", "Unavailable")}  |  CUDA runtime: {info.get("cuda_runtime", "Unknown")}  |  Python: {info.get("python", "Unknown")}')
        else:
            self.hardware_label.configure(text="Hardware details will be included with the completed result.")

    def is_running(self):
        return self.worker is not None and self.worker.is_alive()

    def _publish(self, packet):
        # Keep only recent immutable snapshots; a fast worker cannot flood the UI.
        while True:
            try:
                self.queue.put_nowait(packet)
                return
            except queue.Full:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass

    def start(self):
        if self.is_running() or self.closed:
            return
        self._reset_view()
        self.control = Control()
        self.state = "warming"
        self.status_text.set("Warming engines and compiling CUDA… warmup is excluded from measurements.")
        self.start_button.state(["disabled"])
        self.pause_button.state(["!disabled"])
        self.stop_button.state(["!disabled"])
        if self.settings.display == "benchmark":
            self.unit_control.configure(state="disabled")
        self.window.update_idletasks()
        serial, control = self.run_serial, self.control
        def publish(packet):
            self._publish({**packet, "run_id": serial})
        def run():
            try:
                execute(self.settings, control, publish, self.app.hardware)
            except Exception as exc:
                publish({"state": "error", "message": f"{type(exc).__name__}: {exc}"})
        self.worker = threading.Thread(target=run, daemon=True, name="life-comparison")
        self.worker.start()

    def pause(self):
        if self.control and self.is_running():
            paused = not self.control.paused
            self.control.pause(paused)
            self.pause_button.configure(text="Resume" if paused else "Pause")
            self.status_text.set("Pause requested; finishing the current generation / batch…" if paused else "Resuming…")

    def stop(self):
        if self.control and self.is_running():
            self.control.stop()
            self.status_text.set("Stopping after the current generation / batch…")
            self.stop_button.state(["disabled"])
            self.pause_button.state(["disabled"])

    def reset(self):
        if self.is_running():
            self.reset_pending = True
            self.stop()
            self.status_text.set("Stopping, then restoring the identical initial universe…")
        else:
            self._reset_view()

    def _reset_view(self):
        self.run_serial += 1
        try:
            while True:
                self.queue.get_nowait()
        except queue.Empty:
            pass
        self.report, self.latest_engines = None, {}
        self.state = "ready"
        self.reset_pending = False
        self.progress["value"] = 0
        self.pause_button.configure(text="Pause")
        self.pause_button.state(["disabled"])
        self.stop_button.state(["disabled"])
        self.csv_button.state(["disabled"])
        self.save_button.state(["disabled"])
        self.start_button.state(["!disabled"])
        self.unit_control.configure(state="readonly")
        for key, grid in self.grids.items():
            selected = self.settings.mode in ("both", key)
            grid.set_grid(self.initial if selected else None, self.settings.cell_size, reset=True)
            self.population_labels[key].set(f"Generation 0  ·  {int(self.initial.sum()):,} alive" if selected else "Not selected for this run")
        self.status_text.set("Ready. Press Start simulation to begin.")
        self.update_table()
        self.update_plot()

    def _poll(self):
        if self.closed:
            return
        packet = None
        try:
            while True:
                candidate = self.queue.get_nowait()
                if candidate.get("run_id") == self.run_serial:
                    packet = candidate
        except queue.Empty:
            pass
        if packet:
            self._accept(packet)
        if not self.is_running() and self.state in ("complete", "stopped", "error"):
            self.start_button.state(["!disabled"])
        if self.reset_pending and not self.is_running():
            self._reset_view()
        self.poll_job = self.window.after(70, self._poll)

    def _accept(self, packet):
        self.state = packet["state"]
        if self.state == "error":
            self.status_text.set("Run failed: " + packet["message"])
            self.pause_button.state(["disabled"])
            self.stop_button.state(["disabled"])
            self.unit_control.configure(state="readonly")
            return
        self.latest_engines = packet["engines"]
        self.progress["value"] = sum(m["generations"] for m in self.latest_engines.values())
        for key, grid in packet.get("frames", {}).items():
            self.grids[key].set_grid(grid, self.settings.cell_size)
            self.population_labels[key].set(f'Generation {self.latest_engines[key]["generations"]:,}  ·  {int(grid.sum()):,} alive')
        if self.state in ("complete", "stopped"):
            self.report = packet["report"]
            self._show_hardware(self.report["hardware"])
            self.pause_button.state(["disabled"])
            self.stop_button.state(["disabled"])
            self.pause_button.configure(text="Pause")
            self.csv_button.state(["!disabled"])
            self.save_button.state(["!disabled"])
            self.unit_control.configure(state="readonly")
            match = self.report["results_match"]
            message = "Complete." if self.state == "complete" else "Stopped. Partial results retained; no speedup is claimed."
            if match is True:
                message += f' Final grids match. GPU speedup: {self.report["speedup"]:.2f}× (CPU / GPU total).'
            elif match is False:
                message += " FINAL GRIDS DIFFER — speedup withheld."
            self.status_text.set(message)
        elif self.state == "paused":
            self.status_text.set("Paused. Resume continues from this generation; paused time is excluded.")
        elif self.state != "warming":
            counts = "  ·  ".join(f'{key.upper()} {m["generations"]:,}/{self.settings.iterations:,}' for key, m in self.latest_engines.items())
            self.status_text.set(("Benchmarking without drawing…  " if self.settings.display == "benchmark" else "Running…  ") + counts)
        self.update_table()
        if self.settings.display == "live" or self.state in ("complete", "stopped"):
            self.update_plot()

    def update_table(self):
        rows = stats_rows(self.latest_engines, self.report)
        children = self.table.get_children()
        for index, row in enumerate(rows):
            if index < len(children):
                self.table.item(children[index], values=row)
            else:
                self.table.insert("", "end", values=row, tags=("alternate",) if index % 2 else ())

    def update_plot(self):
        draw_plot(self.axes, self.latest_engines, self.unit.get())
        self.chart.draw_idle()

    def save_csv(self):
        if not self.report:
            return
        path = filedialog.asksaveasfilename(parent=self.window, title="Export comparison statistics",
                                           initialdir=str(ROOT / "results"), initialfile="life-comparison.csv",
                                           defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            try:
                files = export_csv(self.report, path)
                self.status_text.set(f"Saved summary CSV, generation CSV and JSON to {files[0].parent}")
            except Exception as exc:
                messagebox.showerror("Export failed", str(exc), parent=self.window)

    def save_graphs(self):
        if not self.report:
            return
        folder = filedialog.askdirectory(parent=self.window, title="Choose a folder for graphs and statistics", initialdir=str(ROOT / "results"))
        if folder:
            try:
                destination = export_bundle(self.report, folder, self.unit.get())
                self.status_text.set(f"Saved PNG / SVG graphs, PNG / PDF comparison and CSV / JSON data: {destination}")
            except Exception as exc:
                messagebox.showerror("Save failed", str(exc), parent=self.window)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.control:
            self.control.stop()
        if self.poll_job:
            self.window.after_cancel(self.poll_job)
        # No worker accesses widgets. It may finish its current generation after closing.
        self.window.update_idletasks()
        self.window.destroy()
        self.app.retire(self)
        if self.app.comparison is self:
            self.app.comparison = None


def main():
    root = tk.Tk()
    Application(root)
    root.after_idle(lambda: print("Life Lab desktop is ready. Choose settings, then Run → Start simulation.", flush=True))
    root.mainloop()
