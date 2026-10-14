"""Vectorized 2D-array CPU engine with reusable halo and work arrays."""
from time import perf_counter
import numpy as np
from .core import validate_grid

class CPUEngine:
    def __init__(self, initial, boundary="wrap"):
        if boundary not in ("wrap", "dead"):
            raise ValueError("Invalid boundary")
        self.current = validate_grid(initial)
        self.next = np.empty_like(self.current)
        h, w = self.current.shape
        self.halo = np.zeros((h+2, w+2), dtype=np.uint8)
        self.count = np.empty_like(self.current)
        self.born = np.empty(self.current.shape, dtype=bool)
        self.survive = np.empty_like(self.born)
        self.boundary = boundary

    def step(self):
        a, p = self.current, self.halo
        p[1:-1, 1:-1] = a
        if self.boundary == "wrap":
            p[0, 1:-1], p[-1, 1:-1] = a[-1], a[0]
            p[1:-1, 0], p[1:-1, -1] = a[:, -1], a[:, 0]
            p[0, 0], p[0, -1] = a[-1, -1], a[-1, 0]
            p[-1, 0], p[-1, -1] = a[0, -1], a[0, 0]
        h, w = a.shape
        self.count.fill(0)
        for dy in range(3):
            for dx in range(3):
                if dy != 1 or dx != 1:
                    np.add(self.count, p[dy:dy+h, dx:dx+w], out=self.count)
        np.equal(self.count, 3, out=self.born)
        np.equal(self.count, 2, out=self.survive)
        np.logical_and(self.survive, a, out=self.survive)
        np.logical_or(self.born, self.survive, out=self.next)
        self.current, self.next = self.next, self.current
        return self.current

    def advance(self, iterations):
        if type(iterations) is not int or iterations < 0:
            raise ValueError("iterations must be a nonnegative integer")
        for _ in range(iterations):
            self.step()
        return self.current

def measure_cpu(initial, iterations, boundary):
    total_start = perf_counter()
    engine = CPUEngine(initial, boundary)
    start = perf_counter()
    final = engine.advance(iterations)
    compute_ms = (perf_counter() - start) * 1000
    final = final.copy()
    total_ms = (perf_counter() - total_start) * 1000
    return final, {"compute_ms": compute_ms, "total_ms": total_ms}
