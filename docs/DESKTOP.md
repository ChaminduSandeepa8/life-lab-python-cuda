# Life Lab desktop guide

Follow [INSTALL.md](INSTALL.md) to create an environment on Windows, Linux or macOS. Run `run.bat` on Windows or `bash run.sh` on Linux/macOS. Alternatively, use the environment's interpreter directly: `python app.py`. This opens native Python/Tk windows; no localhost server or browser is needed. `python app.py --web` preserves the earlier browser version.

## Settings and preview

| Setting | Choices / behavior |
|---|---|
| Grid | 64, 128, 256, 512, 1024 square cells; default 64 |
| Iterations | 1–100,000; default 100; Stop can end the run early |
| Pattern | Random, Glider, Glider Gun, R-pentomino, Blinker |
| Density | 10–50%; default 25%; enabled for Random |
| Seed | 0–4,294,967,295; default 42; enabled for Random |
| Edges | Wrap-around or dead outside the grid |
| Run mode | Both, CPU only, GPU only |
| Display | Live view or Benchmark (no drawing during measurement) |
| Cell size | 1–12 px; default 5; grid scrollbars handle large views |
| CUDA block | 8×8, 16×16, 32×32; default 16×16 |
| CPU implementation | Normal Python loops (default), NumPy optimized |

Changing input settings regenerates generation zero. Rendering and execution settings update the configuration without changing its cell values. Run captures an immutable configuration and opens a comparison window; Start begins its experiment. To apply different settings, finish/stop and reopen the comparison. Both engines always receive the same seeded array and boundary. The initial SHA-256 is retained in exports.

CUDA availability is checked in the background. A machine without a usable CUDA device initially selects CPU only. Explicit GPU modes require real CUDA; the CPU simulator is rejected for performance measurements. Small grids may underutilize the GPU. Python loops at 1024² can take substantial time, so use a smaller live demonstration.

## Window controls

- **Start simulation** warms the selected implementations, resets both engines to the chosen input, then runs the configured generations.
- **Pause / Resume** waits at the next generation boundary (or CUDA batch boundary in benchmark mode); pause duration is excluded from active timing.
- **Stop** ends at that boundary and keeps partial results. Partial results have actual completed counts, no equality verdict, and no speedup.
- **Reset** first stops any active worker, then clears graphs and statistics and restores the identical starting grid. Press Start for a new run.
- **Export CSV** writes the chosen summary CSV plus sibling `_generations.csv` and `.json` files. The JSON is the authoritative machine-readable record, with unrounded values, configuration, hardware, hashes and timing definitions.
- **Save graphs + stat table** creates a unique timestamped directory with `throughput.png`, `throughput.svg`, `comparison.png`, `comparison.pdf`, CSV/JSON evidence and `methodology.txt`.

The bottom controls stay visible. Scroll the central panel to reach all statistics on smaller displays; each universe has its own scrollbars. The app renders only the visible cell tile, so zooming a 1024² grid does not allocate a 12288² image.

## What the measurements mean

This is a **single-run, instrumented comparison**. It measures the implementation selected in this window, including its launch and synchronization costs. A large CUDA speedup against interpreted loops does not imply the same gain against optimized NumPy.

| Statistic | Definition |
|---|---|
| Total active time | Measured input/buffer setup + active generation processing + final output copy. Excludes warmup/JIT, pauses, playback delay, drawing and preview copies. It is not the time the animation is visible. |
| Average ms/generation | Total active time / actually completed generations |
| Min/max ms/generation | Min/max individual compute samples, excluding input setup and final output |
| Throughput | Grid cells × completed generations / total active seconds |
| Speedup | CPU total / GPU total; available only for a complete Both run with equal final arrays |
| GPU transfer | Synchronous host-observed H2D input + final D2H output copy times; already included in total |
| GPU kernel events | Sum of CUDA-event intervals around each kernel. These can include dispatch/idle gaps; they are not profiler-isolated active kernel durations. |
| Preview transfer | Additional GPU copies used by live animation; reported separately and excluded from total |
| Grid bytes | One uint8 grid = size × size bytes |
| Engine buffer bytes | CPU arrays: two grids for Python loops, plus reusable work arrays for NumPy; GPU: two device grids. Excludes Python-object overhead, retained input/preview grids, CUDA context, allocator cache and driver memory. |
| Final alive / match | Population per final array; full byte-for-byte equality is checked, not merely equal populations |
| Hardware | CPU model, GPU model, CUDA runtime version, Python and environment metadata |

In **Live view**, engines alternate execution order each generation and their matching generation is displayed together, paced by an untimed 60 ms delay. The worker synchronizes each GPU generation for display. This is an educational animation; its wall duration is intentionally much longer than its compute time.

In **Benchmark mode**, CPU and GPU run sequentially to avoid competing for Python's execution thread. GPU launches are grouped in batches of up to 32, with CUDA event pairs around individual kernels. Host launch times are measured individually; each generation receives an equal share of its batch's final synchronization time. Therefore GPU min/max wall samples in benchmark mode are instrumented batch estimates. The raw kernel event durations are exported separately. There are no intermediate grid transfers or graph/canvas redraws while the benchmark runs. The progress bar and numeric statistics remain responsive.

The graph plots each engine's **per-generation rate** (`1 / generation_compute_ms`) against its **cumulative active compute time**. Switching to seconds divides x by 1000 and multiplies y by 1000. The axes do not describe shared animation wall time. More than 1200 points are decimated for display; every generation remains in CSV/JSON. A faster engine finishes further left and generally has a higher rate.

For repeatable report evidence, run several experiments under comparable power/thermal conditions. `python -m tools.run_experiment --mode both --size 256` runs the same normal-Python/CUDA engines without the GUI and exports raw timings. The older `python -m life.benchmark` CLI uses an optimized CPU baseline and different instrumentation; keep those datasets separate.

## Modules and extension points

```text
app.py → desktop.application
               ├── models: settings, validation, measurement definitions
               ├── widgets: viewport drawing and scrolling
               ├── runner: worker, cancellation, CPU/GPU scheduling
               │      └── life: patterns, Python loops, NumPy engine, CUDA kernel
               ├── presentation: graph and statistics formatting
               └── export: numeric evidence and standalone figures
```

All Tk operations happen on the main thread. Workers publish immutable snapshots into a bounded queue. Rendering cannot change an engine's arrays. GPU buffers remain on the device through the run; live copies are made only for display. Both implementations use old/new buffers to preserve simultaneous Game of Life updates.

Tests:

```bash
python -m unittest discover -s tests -v
python -m tools.smoke_desktop --cpu-only
```

The second command needs a desktop display. Omit `--cpu-only` to exercise both engines on real CUDA. It tests the app's own widget/controller methods, settings validation, large-grid rendering, Pause/Resume/Stop/Reset, both display modes and exports.

Implementation references: [Python Tkinter threading model](https://docs.python.org/3/library/tkinter.html#threading-model) and [Numba CUDA host/event API](https://nvidia.github.io/numba-cuda/reference/host.html).
