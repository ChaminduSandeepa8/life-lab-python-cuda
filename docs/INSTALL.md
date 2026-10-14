# Install and run Life Lab

The desktop application runs with Python 3.10–3.13, Tcl/Tk, Pillow, Matplotlib and the core requirements. Use Python 3.12 for a new installation. A CPU-only installation does not need CUDA. Windows and Linux with a compatible NVIDIA GPU can optionally install the CUDA backend. This backend is not supported on macOS.

## 1. Get the source

Install Git and a supported 64-bit Python interpreter. Clone to a folder you can write to:

```text
git clone https://github.com/ChaminduSandeepa8/life-lab-python-cuda.git
cd life-lab-python-cuda
```

Paths containing spaces are supported. Do not copy another computer's `.venv`; create one locally. Download ZIP from GitHub also works: extract it and open a terminal in the extracted project directory.

## 2. Install Python and Tcl/Tk

### Windows

Use the Python.org installer. Include the **Python launcher**, **pip**, and **Tcl/Tk and IDLE** components. In a new terminal, verify:

```powershell
py -3.12 --version
py -3.12 -m tkinter
```

The second command should open a small test window; close it before continuing. If `py` is unavailable, use the full path to the installed Python executable or `python` if it refers to Python 3.10–3.13.

### Linux

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install python3 python3-venv python3-tk git
python3 --version
python3 -m tkinter
```

Fedora: install `python3`, `python3-pip`, `python3-tkinter` and `git` with `dnf`. Other distributions use their equivalent packages. Tcl/Tk must belong to the same Python installation used to create the environment. If the distribution's Python is newer than 3.13, install a supported interpreter separately or use Conda below.

### macOS

Use the Python.org Python 3.12 installer for your Mac, then verify `python3.12 --version` and `python3.12 -m tkinter`. Select CPU only in the application. Apple Silicon, Intel and AMD graphics do not run this NVIDIA CUDA backend. Tk installation details are in the [Python Tkinter documentation](https://docs.python.org/3/library/tkinter.html).

## 3. Create the local environment

Automatic setup, run from the project directory:

```powershell
# Windows
py -3.12 tools/setup_environment.py
```

```bash
# Linux (use a supported python3)
python3 tools/setup_environment.py
# macOS with Python 3.12
python3.12 tools/setup_environment.py
```

The script creates `.venv`, installs the requirements and runs diagnostics. If `.venv` already exists, its interpreter is reused. The script does not modify system drivers or system Python packages. To change Python versions, close the app, rename the old `.venv`, and rerun setup with the desired interpreter.

Manual equivalent on Windows:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements-desktop.txt
.venv\Scripts\python.exe app.py --check
.venv\Scripts\python.exe app.py
```

Manual equivalent on Linux/macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-desktop.txt
.venv/bin/python app.py --check
.venv/bin/python app.py
```

Use `python3.12` instead of `python3` if needed. Activation is optional; using the explicit interpreter avoids PATH and PowerShell activation problems. On subsequent runs, use `run.bat` on Windows or `bash run.sh` on Linux/macOS.

### Conda alternative

Useful when a distribution lacks a supported Python/Tk combination:

```bash
conda create -n life-lab python=3.12 tk pip
conda activate life-lab
python -m pip install -r requirements-desktop.txt
python app.py --check
python app.py
```

An already existing compatible environment can be activated instead. The environment name is your choice. Launch scripts prefer the active virtual/Conda environment, then the project's `.venv`, then Python on PATH. Use an explicit interpreter when multiple environments are installed.

## 4. Optional real CUDA setup

1. Use Windows or Linux and an NVIDIA GPU supported by the CUDA 12 toolkit being installed.
2. Install a current, compatible NVIDIA driver using NVIDIA's installer or your distribution's driver manager. Reboot if requested by that installer.
3. Run `nvidia-smi` and confirm it lists the intended GPU.
4. Install the GPU requirements into **the same environment used to launch Life Lab**.

```bash
python -m pip install -r requirements-cuda.txt
python app.py --check --require-cuda
python app.py
```

Replace `python` with `.venv\Scripts\python.exe` on Windows or `.venv/bin/python` on Linux if the environment is not activated. Automatic setup also accepts `--cuda`:

```text
python tools/setup_environment.py --cuda
```

The CUDA 12 package extra installs the Python-side toolkit libraries. It does not install the GPU driver. The diagnostic launches a real kernel and checks its output; device detection alone is insufficient. Fresh installations use the separately maintained CUDA package. An existing working Numba CUDA environment can also run the project.

Official installation and driver/toolkit compatibility: [Numba CUDA installation](https://nvidia.github.io/numba-cuda/user/installation.html), [NVIDIA CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/).

### CUDA troubleshooting

| Symptom | What to check |
| --- | --- |
| `nvidia-smi` missing or cannot communicate with driver | Install/fix the host NVIDIA driver; pip cannot repair it. |
| App selects CPU only | Run `python app.py --check --require-cuda` in the exact environment used to launch the app. Read its device/compiler error. |
| Device is listed but compilation fails | Check driver/toolkit compatibility, GPU compute capability and the installed CUDA package versions. Update the driver where needed. |
| Driver/library/unsupported PTX error | Use the toolkit supported by the host driver; a working `nvidia-smi` alone does not guarantee a compatible toolkit. |
| A container or sandbox cannot see the GPU | Enable supported GPU passthrough or run from the normal host terminal. Do not use CPU simulation as performance evidence. |
| Simulator is enabled | Remove `NUMBA_ENABLE_CUDASIM` from the environment before measuring real CUDA. |
| AMD, Intel or Apple GPU | Use CPU only; these devices cannot run this CUDA implementation. |

On WSL2, use the Windows NVIDIA driver and NVIDIA's WSL instructions. A native Windows install is simpler for the desktop UI. WSL desktop mode needs working graphical forwarding; the browser or headless experiment is an alternative.

## 5. Validate

```bash
python -m unittest discover -s tests -v
python -m tools.run_experiment --mode cpu --iterations 100
# Only on a real CUDA machine:
python -m tools.run_experiment --mode both --size 256 --iterations 100
```

Tests validate rules, patterns, boundaries, Python/CUDA equality where hardware exists, controller behavior and exports. Without real CUDA, GPU tests skip. GitHub-hosted CPU checks must not be reported as GPU performance tests.

For a graphical session, run `python -m tools.smoke_desktop --cpu-only`. It opens and closes test windows, exercises controls and writes exports under `results/desktop-validation/`. Omit `--cpu-only` to exercise both engines on a real CUDA machine. Linux CI uses `xvfb-run -a` for a virtual display.

## 6. Common launch problems

| Problem | Solution |
| --- | --- |
| `python` / `py` not found | Install Python and reopen the terminal; use the interpreter's full path if necessary. |
| Missing `venv` / `ensurepip` on Linux | Install the distribution's matching Python venv package. |
| Missing `_tkinter` / `tkinter` | Install Tcl/Tk for that Python; it is not a normal pip package. |
| Missing Pillow or Matplotlib | Run `python -m pip install -r requirements-desktop.txt` with the same interpreter. |
| PowerShell blocks activation scripts | Use `.venv\Scripts\python.exe app.py`; activation is not required. |
| No `$DISPLAY` / cannot open display | Run inside a graphical session, use `python app.py --web`, or use the headless experiment. |
| `127.0.0.1 refused to connect` | Start `python app.py --web` and keep the terminal open. Native desktop mode does not start a web server. |
| Port 8000 already in use | Run `python app.py --web --port 8001`, then open that port. |
| Live animation looks equally fast | Live view intentionally displays matching generations; inspect measured active times or use Benchmark. |
| Large Python grid is slow | Start with 64 or 128; use Stop if needed. Pure Python loop work grows with cell count. |
| Exports fail | Choose a writable folder and close a file if another application has locked it. |

## Updating

Stop the app, run `git pull`, then reinstall the requirements in the existing environment. Keep your own exported results outside tracked source files. Generated results and environments are ignored by Git.

```bash
git pull
python -m pip install -r requirements-desktop.txt
python app.py --check
```

Reinstall `requirements-cuda.txt` only when updating the optional GPU packages. If an existing GPU environment works, record its package versions before changing it.
