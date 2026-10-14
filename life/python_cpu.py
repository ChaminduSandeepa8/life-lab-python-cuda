"""Sequential Python loops over two uint8 2D arrays; no JIT or vectorized update."""
import numpy as np
from .core import validate_grid


class PythonEngine:
    def __init__(self, initial, boundary="wrap"):
        if boundary not in ("wrap", "dead"):
            raise ValueError("Invalid boundary")
        self.current = validate_grid(initial)
        self.next = np.empty_like(self.current)
        self.wrap = boundary == "wrap"

    def step(self):
        a, out = self.current, self.next
        height, width = a.shape
        # Read only from the old generation. Swap buffers after the full update.
        for row in range(height):
            for col in range(width):
                count = 0
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        y, x = row + dy, col + dx
                        if self.wrap:
                            count += int(a[y % height, x % width])
                        elif 0 <= y < height and 0 <= x < width:
                            count += int(a[y, x])
                out[row, col] = count == 3 or (a[row, col] == 1 and count == 2)
        self.current, self.next = self.next, self.current
        return self.current

    def advance(self, iterations):
        if type(iterations) is not int or iterations < 0:
            raise ValueError("iterations must be a nonnegative integer")
        for _ in range(iterations):
            self.step()
        return self.current
