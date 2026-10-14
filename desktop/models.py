"""Validated desktop settings and portable, explicitly defined measurements."""
from dataclasses import asdict, dataclass, field
import math
import numpy as np
from life.core import Config

PATTERNS = {"Random": "random", "Glider": "glider", "Glider Gun": "gosper-gun",
            "R-pentomino": "r-pentomino", "Blinker": "blinker"}
CPU_LABELS = {"python": "Normal Python · nested loops", "numpy": "NumPy · vectorized 2D arrays"}


@dataclass(frozen=True)
class Settings:
    size: int = 64
    iterations: int = 100
    pattern: str = "random"
    density: float = .25
    seed: int = 42
    boundary: str = "wrap"
    mode: str = "both"
    display: str = "live"
    cell_size: int = 5
    block: int = 16
    cpu_backend: str = "python"

    def __post_init__(self):
        if self.size not in (64, 128, 256, 512, 1024):
            raise ValueError("Choose a grid size from 64 to 1024.")
        if type(self.iterations) is not int or not 1 <= self.iterations <= 100_000:
            raise ValueError("Iterations must be an integer from 1 to 100,000.")
        if self.pattern not in PATTERNS.values():
            raise ValueError("Choose an initial pattern.")
        if not math.isfinite(self.density) or not .1 <= self.density <= .5:
            raise ValueError("Random density must be between 10% and 50%.")
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("Seed must be an integer from 0 to 4,294,967,295.")
        if self.boundary not in ("wrap", "dead") or self.mode not in ("both", "cpu", "cuda"):
            raise ValueError("Invalid boundary or run mode.")
        if self.display not in ("live", "benchmark"):
            raise ValueError("Choose Live view or Benchmark.")
        if type(self.cell_size) is not int or not 1 <= self.cell_size <= 12:
            raise ValueError("Cell size must be 1 to 12 pixels.")
        if self.block not in (8, 16, 32) or self.cpu_backend not in CPU_LABELS:
            raise ValueError("Invalid CUDA block size or CPU implementation.")

    def grid_config(self):
        # Grid construction is independent of the desktop run's iteration limit.
        return Config(size=self.size, seed=self.seed, density=self.density,
                      boundary=self.boundary, pattern=self.pattern,
                      block_x=self.block, block_y=self.block)

    def to_dict(self):
        return asdict(self)


@dataclass
class Measurement:
    backend: str
    grid_bytes: int
    buffer_bytes: int = 0
    setup_ms: float = 0.0
    output_ms: float = 0.0
    h2d_ms: float = 0.0
    d2h_ms: float = 0.0
    preview_transfer_ms: float = 0.0
    durations_ms: list = field(default_factory=list)
    kernel_durations_ms: list = field(default_factory=list)
    final_alive: int | None = None
    final_sha256: str | None = None

    def summary(self, include_samples=False):
        n = len(self.durations_ms)
        compute = sum(self.durations_ms)
        total = self.setup_ms + compute + self.output_ms
        out = dict(backend=self.backend, generations=n, total_ms=total,
                   compute_ms=compute, setup_ms=self.setup_ms, output_ms=self.output_ms,
                   average_ms=total / n if n else None,
                   min_ms=min(self.durations_ms) if n else None,
                   max_ms=max(self.durations_ms) if n else None,
                   cells_per_second=self.grid_bytes * n * 1000 / total if n and total else None,
                   h2d_ms=self.h2d_ms, d2h_ms=self.d2h_ms,
                   transfer_ms=self.h2d_ms + self.d2h_ms,
                   preview_transfer_ms=self.preview_transfer_ms,
                   kernel_ms=sum(self.kernel_durations_ms) if self.backend == "cuda" else None,
                   grid_bytes=self.grid_bytes, buffer_bytes=self.buffer_bytes,
                   final_alive=self.final_alive, final_sha256=self.final_sha256)
        if n:
            elapsed = np.cumsum(self.durations_ms)
            indices = np.unique(np.linspace(0, n - 1, min(n, 1200), dtype=int))
            out["series"] = [[int(i + 1), float(elapsed[i]), 1 / self.durations_ms[i]
                              if self.durations_ms[i] > 0 else 0] for i in indices]
        else:
            out["series"] = []
        if include_samples:
            out["generation_ms"] = list(self.durations_ms)
            out["kernel_generation_ms"] = list(self.kernel_durations_ms)
        return out


METHODOLOGY = (
    "One run; identical uint8 input and boundary for both engines. Warmup/JIT excluded. "
    "Total = allocation/input setup + active generation processing + final output copy; "
    "pauses, live pacing, graph rendering and preview copies excluded. Average = total/completed generations. "
    "Min/max describe individual generation compute times, excluding setup/output. Throughput uses total. "
    "Live GPU generations synchronize individually. Benchmark GPU launches batches of up to 32; "
    "per-generation wall times are host submission time plus an equal share of batch synchronization. "
    "CUDA events bracket each kernel; event intervals can include dispatch/idle gaps, not profiler-only active time. "
    "H2D/D2H are synchronous host-observed copy times, already included in total. "
    "Preview transfers are reported separately. Buffer memory counts engine array payloads, not process/driver memory. "
    "Graph: per-generation rate versus each engine's cumulative active compute time; decimated above 1200 samples. "
    "Raw generation samples are retained in exports. Equality and speedup require both complete runs. "
    "The interactive per-generation instrumentation adds overhead; use repeated CLI experiments for report-grade medians."
)
