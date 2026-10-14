# Life Lab — Python CPU and CUDA Game of Life

An interactive Conway's Game of Life experiment with two native Python windows: configure an initial grid, then compare **normal Python loops on the CPU** with a **CUDA kernel on an NVIDIA GPU**. Both implementations start from the same grid and use the same rules.

## Platform support

| Platform | CPU simulation and exports | CUDA GPU simulation |
| --- | --- | --- |
| Windows 10/11, 64 bit | Supported setup | Compatible NVIDIA GPU and driver required |
| Linux, 64 bit | Supported setup; desktop mode needs Tcl/Tk and a display | Compatible NVIDIA GPU and driver required |
| macOS, Intel or Apple Silicon | Supported setup | Not supported by this CUDA implementation; use CPU only |
| Remote/headless machine | Browser mode or command-line experiment | Available only if real NVIDIA CUDA is accessible |

Use **Python 3.10–3.13**; **Python 3.12** is recommended for a new installation. GPU acceleration is optional. No particular Conda environment, home directory, or GPU model is hard-coded into the launchers. Internet is needed to clone and install dependencies; the installed application runs locally.

Linux CPU and RTX 3050 execution have been tested locally. The included GitHub Actions matrix runs CPU checks on Windows, Linux and macOS with Python 3.10, 3.12 and 3.13. Check the repository's **Actions** tab for the result on the current commit. Hosted CPU runners do not validate real GPU performance.

## Clone and install

Install [Git](https://git-scm.com/downloads) and a supported [Python](https://www.python.org/downloads/) first. On Windows, include Tcl/Tk and the Python launcher in the installer. On macOS, the Python.org installer is the simplest route to a Python build with Tk.

```text
git clone https://github.com/ChaminduSandeepa8/life-lab-python-cuda.git
cd life-lab-python-cuda
```

### Windows — PowerShell or Command Prompt

```powershell
py -3.12 tools/setup_environment.py
.venv\Scripts\python.exe app.py
```

After setup, `run.bat` also starts the app (`.\run.bat` in PowerShell). Activation is optional; the commands above use the environment directly and do not require changing PowerShell execution policy.

### Linux

On Ubuntu/Debian, install the system components once:

```bash
sudo apt update
sudo apt install python3 python3-venv python3-tk git
```

Use a Python 3.10–3.13 installation; distributions with a newer default Python need a supported interpreter or Conda (see the installation guide).

```bash
python3 tools/setup_environment.py
bash run.sh
```

### macOS

Install Python 3.12 from Python.org, then:

```bash
python3.12 tools/setup_environment.py
bash run.sh
```

Choose **CPU only**. The app selects it automatically when no usable CUDA device is detected. Other operating systems are not part of the supported test matrix.

Full manual setup, Conda instructions and troubleshooting: **[Installation guide](docs/INSTALL.md)**. Sinhala quick start: **[සිංහල උපදෙස්](docs/START-SI.md)**.

## Optional NVIDIA CUDA setup — Windows or Linux

First install a compatible NVIDIA driver and confirm `nvidia-smi` works. Then add the CUDA 12 dependencies to the same environment:

```text
python -m pip install -r requirements-cuda.txt
python app.py --check --require-cuda
```

Here `python` must be your activated environment's interpreter. Without activation, use `.venv\Scripts\python.exe` on Windows or `.venv/bin/python` on Linux. For a new installation, append `--cuda` to the setup command instead.

The check actually compiles a CUDA kernel, executes it and compares the result with Python. Installing packages alone does not install an NVIDIA driver or make an unsupported GPU compatible. AMD/Intel/Apple GPUs cannot execute this CUDA backend. See the [official CUDA Python installation instructions](https://nvidia.github.io/numba-cuda/user/installation.html) and [our CUDA troubleshooting steps](docs/INSTALL.md#cuda-troubleshooting).

An existing Conda environment also works: activate it, install the requirements, then run `python app.py`. `run.sh` and `run.bat` prefer an active environment, otherwise the project's `.venv`, then Python on PATH. `run-desktop.sh` is an alias for `run.sh`; `run-gpu.sh` verifies real CUDA before launch.

## Use the two windows

1. **Settings window:** choose the grid, iterations, pattern, density, seed, boundaries, run/display mode, cell size and CUDA block size. The starting preview changes automatically.
2. Keep **Normal Python** selected for the sequential CPU comparison. Choose **Both** only on a working CUDA machine, or **CPU only** elsewhere.
3. Click **Run → Open comparison**, then **Start simulation** in the second window. CPU is on the left; CUDA is on the right.
4. **Live view** animates matching generations. **Benchmark** skips drawing during computation for timing measurements. Use 64×64 for a short live demonstration; 1024×1024 Python-loop runs can take minutes.
5. **Pause / Resume**, **Stop** and **Reset** control the run. The graph switches between iterations/ms and iterations/s. Scroll down for statistics and hardware details.
6. **Export CSV** saves summary statistics, raw generation timings and JSON metadata. **Save graphs + stat table** additionally saves PNG/SVG graphs and a PNG/PDF comparison sheet.

The table includes total active time, average/min/max generation time, throughput, speedup, transfers, CUDA events, buffer memory, population and full final-grid equality. Incomplete or unmatched runs do not receive a speedup. CUDA is never silently replaced with CPU simulation.

[Detailed controls, architecture and measurement definitions](docs/DESKTOP.md)

## Verify a fresh installation

Run with your environment's Python from the cloned project directory:

```bash
python app.py --check
python -m unittest discover -s tests -v
python -m tools.run_experiment --mode cpu --size 64 --iterations 100
```

The last command uses **the same normal-Python engine as the desktop**, without opening a window. It writes `results/experiment/statistics.csv`, a generation CSV and JSON. No CUDA packages are needed for a CPU-only installation. GPU-specific tests skip when hardware is unavailable.

For a real comparison:

```bash
python app.py --check --require-cuda
python -m tools.run_experiment --mode both --size 256 --iterations 100 --block 16 --output results/comparison/statistics.csv
```

Use `--mode cuda` for GPU only, `--boundary dead` for dead edges, or `--block 8`, `16` or `32`. Repeat runs under similar system load before drawing conclusions. All timing figures describe the selected implementations and machine; they are not universal CPU/GPU ratios.

## Optional browser interface

```bash
python app.py --web
```

Keep that terminal running and open **http://127.0.0.1:8000** on the same computer. Use `--port 8001` if port 8000 is occupied. `ERR_CONNECTION_REFUSED` means the server is not listening; start it and check the terminal for an error. Stop it with Ctrl+C.

Browser-only setup: `python tools/setup_environment.py --web`. The earlier browser and `python -m life.benchmark` use a different optimized CPU implementation. Use the desktop or `tools.run_experiment` for the normal-Python/CUDA comparison; do not mix their timing datasets.

## Project layout

```text
app.py                     Desktop launcher, --web, --check
life/python_cpu.py         Sequential Python cell/neighbor loops
life/gpu.py                CUDA kernel, transfers and device buffers
life/core.py               Rules, initial patterns and validation
desktop/application.py     Settings/comparison windows and controls
desktop/runner.py          Background execution, pause/stop and timing
desktop/models.py          Validated configuration and measurements
desktop/widgets.py         Scrollable grid rendering
desktop/presentation.py    Charts and statistics definitions
desktop/export.py          CSV/JSON, images and PDF exports
tools/setup_environment.py Portable environment setup
tools/check_environment.py Dependency and execution diagnostics
tools/run_experiment.py    Headless normal-Python/CUDA experiment
tests/                     Correctness, startup, controller and export tests
web/ and web_server.py     Optional local browser interface
docs/                      Setup, user guide and presentation notes
.github/workflows/         Cross-platform automated CPU checks
```

Generated results, virtual environments, local assignment drafts and personal template documents are excluded from Git. Each user generates their own results on their own hardware. An assignment-specific repeated experiment is available as `python -m tools.python_cuda_report_experiments` (real CUDA required; the full study takes several minutes). Its results describe that run only.

## Rules and implementation sources

The game uses B3/S23: a dead cell is born with three live neighbors, a live cell survives with two or three, and all other cells die. Each generation reads the old grid and writes a separate new grid. Wrap-around and dead-outside are finite-board boundary choices.

- [Conway's Game of Life rules](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)
- [CUDA kernel programming](https://nvidia.github.io/numba-cuda/user/kernels.html)
- [CUDA events and host API](https://nvidia.github.io/numba-cuda/reference/host.html)
- [Python Tkinter](https://docs.python.org/3/library/tkinter.html)
- [Assignment reference provided for feature context](https://github.com/amantha2002/Conway-s-Game-of-Life-Python-and-cuda-)

The project implementation and interface were written independently. The reference project is acknowledged for assignment context.
