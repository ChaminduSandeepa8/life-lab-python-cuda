# Screenshots for the desktop application

Capture your own current application and hardware. Use Normal Python for the CPU comparison. The repository contains source and instructions; local assignment documents and generated results are not included in a fresh clone.

| Screenshot | Capture | Suggested report location |
| --- | --- | --- |
| Settings and preview | Grid size, 100 iterations, pattern, seed, run mode, Normal Python selection and starting canvas | System design / settings |
| Live comparison | CPU and CUDA canvases at matching generations, generation labels and graph | Interactive execution |
| Hardware | `nvidia-smi`, `python --version` and the CPU model from system information | Hardware and software |
| Completed benchmark | Benchmark mode, completed generation count, statistics, speedup and full-grid match | Results |
| Tests | `python -m unittest discover -s tests -v` and its final summary | Correctness verification |

Start the native app with `run.bat` on Windows or `bash run.sh` on Linux/macOS. Native mode does not need a browser. Select 64×64 for a clear live demonstration. Use a real CUDA machine for a two-engine screenshot; CPU-only screenshots should be identified as CPU only.

Run `python app.py --check --require-cuda` to check a GPU setup. On Windows, use Task Manager or System Information for the CPU model; on Linux use `lscpu`; on macOS use About This Mac (CPU-only execution). The `nvidia-smi` CUDA version indicates driver capability, not necessarily the toolkit loaded by Python.

After a benchmark, export CSV/JSON evidence before Reset. A new single run can differ from the median of several report runs. Keep the screenshot and its exported data together; do not replace individual table values with numbers from a different experiment.

For Word, insert readable screenshots in line with text, preserve aspect ratio, add a descriptive caption, and review page breaks after insertion. Crop unrelated desktop content. Do not use another project's screenshots as evidence of this application.
