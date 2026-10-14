# Two-person presentation — approximately 2 minutes 30 seconds

Use 64×64, 100 iterations, Random, density 25%, seed 42, Both, Normal Python, Live view and a 16×16 block on a real CUDA machine. Run once before recording to avoid waiting for initial compilation. Spoken English is below; screen actions are italicized. Rehearse at a comfortable pace and adjust pauses to finish within 2:30.

## Person 1 — settings window — 0:00 to 1:10

*0:00–0:20: Show the settings window.*

Good morning. We are presenting Life Lab, our Conway's Game of Life application. Our objective is to compare sequential Python execution on the CPU with parallel CUDA execution on the GPU.

The application separates simulation logic, performance measurement, and the interface into different modules.

*0:20–0:50: Point to the relevant settings.*

This first window configures the experiment. We can select the grid size, iteration count, and initial pattern. Density controls the starting population, while the same seed gives both implementations identical input for a fair comparison.

We can also choose edge behavior, run mode, cell size, and CUDA block size. Live view shows the simulation, while Benchmark mode skips drawing during computation.

*0:50–1:10: Change the pattern to Glider, then back to Random. Click Run at the handoff.*

Changing the pattern or density updates the preview automatically.

For this demonstration, we selected a sixty-four by sixty-four grid and one hundred generations. Clicking Run opens the comparison window. My partner will now demonstrate the simulation.

## Person 2 — simulation window — 1:10 to 2:30

*1:10–1:35: Click Start simulation and point to the two canvases.*

Clicking Start begins the simulation. The left canvas shows normal Python, and the right canvas shows CUDA.

Python updates cells sequentially using nested loops. CUDA assigns one thread to each cell, allowing many cells to update in parallel. Both implementations apply the same rules.

*1:35–2:00: Show the graph, switch ms to s, and scroll to the statistics table.*

The graph plots iterations per unit time against accumulated compute time. We can switch between milliseconds and seconds.

The table compares execution time, throughput, memory, and GPU transfer costs. Speedup is CPU time divided by GPU time. The final-grid match checks whether both implementations produced identical results.

*2:00–2:30: Point to the controls. Do not reset before showing completed results.*

For performance, we reuse two buffers and keep simulation data on the GPU. Warmup is excluded from timing, and Benchmark mode avoids drawing overhead.

Pause and Resume control playback, Stop ends a run, and Reset restores the initial grid. We can also export CSV data and save the graphs and statistics.

Our benchmarks showed CUDA completing the tested workloads faster than sequential Python. Thank you.

Only use the final performance claim if your actual benchmark supports it. On a CPU-only machine, explain that real GPU measurements require a compatible NVIDIA machine. Live animation is paced and should not be used as a visual speed race.
