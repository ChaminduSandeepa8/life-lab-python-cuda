"""Shared inputs, patterns and the readable, scalar 2D-array reference."""
from dataclasses import dataclass, asdict
import numpy as np

PATTERNS = {
    "glider": [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
    "blinker": [(0, 0), (0, 1), (0, 2)],
    "block": [(0, 0), (0, 1), (1, 0), (1, 1)],
    "r-pentomino": [(0, 1), (0, 2), (1, 0), (1, 1), (2, 1)],
    "gosper-gun": [(5,1),(5,2),(6,1),(6,2),(3,13),(3,14),(4,12),
        (4,16),(5,11),(5,17),(6,11),(6,15),(6,17),(6,18),(7,11),
        (7,17),(8,12),(8,16),(9,13),(9,14),(1,25),(2,23),(2,25),
        (3,21),(3,22),(4,21),(4,22),(5,21),(5,22),(6,23),(6,25),
        (7,25),(3,35),(3,36),(4,35),(4,36)],
}

@dataclass(frozen=True)
class Config:
    size: int = 256
    iterations: int = 100
    repeats: int = 5
    seed: int = 42
    density: float = 0.25
    boundary: str = "wrap"
    pattern: str = "random"
    block_x: int = 32
    block_y: int = 8

    def __post_init__(self):
        for name, lo, hi in (("size", 8, 2048), ("iterations", 1, 1000),
                             ("repeats", 1, 10), ("seed", 0, 2**32-1)):
            value = getattr(self, name)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f"{name} must be an integer from {lo} to {hi}")
        if not isinstance(self.density, (int, float)) or not 0 <= self.density <= 1:
            raise ValueError("density must be between 0 and 1")
        if self.boundary not in ("wrap", "dead"):
            raise ValueError("boundary must be wrap or dead")
        if self.pattern not in ("random", "empty", *PATTERNS):
            raise ValueError("Unknown pattern")
        if (self.block_x, self.block_y) not in ((8,8), (16,16), (32,8), (32,16), (32,32)):
            raise ValueError("Unsupported CUDA block dimensions")

    def to_dict(self):
        return asdict(self)

def validate_grid(grid):
    a = np.asarray(grid)
    if a.ndim != 2 or min(a.shape) < 1 or max(a.shape) > 2048:
        raise ValueError("Expected a nonempty 2D grid, maximum dimension 2048")
    if not np.all((a == 0) | (a == 1)):
        raise ValueError("Cells must be 0 or 1")
    return np.array(a, dtype=np.uint8, order="C", copy=True)

def make_grid(config):
    if config.pattern == "random":
        return (np.random.default_rng(config.seed).random((config.size, config.size))
                < config.density).astype(np.uint8)
    a = np.zeros((config.size, config.size), dtype=np.uint8)
    if config.pattern == "empty":
        return a
    coords = np.asarray(PATTERNS[config.pattern])
    coords = coords - coords.min(axis=0)
    shape = coords.max(axis=0) + 1
    if max(shape) > config.size:
        raise ValueError(f"{config.pattern} needs a grid at least {max(shape)} cells wide")
    coords += (config.size - shape) // 2
    a[coords[:, 0], coords[:, 1]] = 1
    return a

def reference_step(grid, boundary="wrap"):
    """Explicit nested loops: independent correctness oracle, not timed baseline."""
    a = validate_grid(grid)
    if boundary not in ("wrap", "dead"):
        raise ValueError("Invalid boundary")
    h, w = a.shape
    out = np.zeros_like(a)
    for row in range(h):
        for col in range(w):
            count = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    r, c = row + dy, col + dx
                    if boundary == "wrap":
                        count += int(a[r % h, c % w])
                    elif 0 <= r < h and 0 <= c < w:
                        count += int(a[r, c])
            out[row, col] = count == 3 or (a[row, col] == 1 and count == 2)
    return out
