"""True CUDA implementation. Simulator is accepted only for correctness tests."""
from time import perf_counter
import numpy as np
from .core import validate_grid

try:
    from numba import cuda, config as numba_config
    IMPORT_ERROR = None
except Exception as exc:
    cuda = None
    numba_config = None
    IMPORT_ERROR = str(exc).splitlines()[0][:300]

if cuda is not None:
    @cuda.jit
    def life_kernel(current, following, wrap):
        # x follows contiguous columns, y follows rows (C-order arrays).
        x, y = cuda.grid(2)
        height, width = current.shape
        if x >= width or y >= height:
            return
        neighbors = 0
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                yy, xx = y + dy, x + dx
                if wrap:
                    yy = (yy + height) % height
                    xx = (xx + width) % width
                    neighbors += int(current[yy, xx])
                elif 0 <= yy < height and 0 <= xx < width:
                    neighbors += int(current[yy, xx])
        following[y, x] = 1 if (neighbors == 3 or
                                (current[y, x] == 1 and neighbors == 2)) else 0

def cuda_status():
    if cuda is None:
        return {"available": False, "simulator": False, "reason":
                "CUDA runtime unavailable. Install requirements-cuda.txt on an NVIDIA machine. "
                + (IMPORT_ERROR or "")}
    if numba_config.ENABLE_CUDASIM:
        return {"available": False, "simulator": True, "reason":
                "CUDA simulator enabled: correctness only; GPU timings are disabled."}
    try:
        if not cuda.is_available():
            return {"available": False, "simulator": False, "reason":
                    "No usable NVIDIA CUDA device/driver. CPU mode is available."}
        dev = cuda.get_current_device()
        name = dev.name.decode() if isinstance(dev.name, bytes) else str(dev.name)
        return {"available": True, "simulator": False, "name": name,
                "compute_capability": list(dev.compute_capability),
                "reason": "NVIDIA CUDA device detected"}
    except Exception as exc:
        return {"available": False, "simulator": False, "reason": str(exc).splitlines()[0][:300]}

class CUDAEngine:
    def __init__(self, initial, boundary="wrap", block=(32,8), *, allow_simulator=False):
        status = cuda_status()
        if not status["available"] and not (allow_simulator and status["simulator"]):
            raise RuntimeError(status["reason"])
        if boundary not in ("wrap", "dead"):
            raise ValueError("Invalid boundary")
        if block not in ((8,8), (16,16), (32,8), (32,16), (32,32)):
            raise ValueError("Unsupported block size")
        initial = validate_grid(initial)
        self.current = cuda.device_array_like(initial)
        transfer_start = perf_counter()
        self.current.copy_to_device(initial)
        self.input_transfer_ms = (perf_counter() - transfer_start) * 1000
        self.next = cuda.device_array_like(initial)
        self.wrap = boundary == "wrap"
        self.block = block
        h, w = initial.shape
        self.blocks = ((w + block[0] - 1)//block[0], (h + block[1] - 1)//block[1])

    def advance(self, iterations):
        if type(iterations) is not int or iterations < 0:
            raise ValueError("iterations must be a nonnegative integer")
        for _ in range(iterations):
            life_kernel[self.blocks, self.block](self.current, self.next, self.wrap)
            self.current, self.next = self.next, self.current

    def result(self, out=None):
        return self.current.copy_to_host(out)

def warmup_cuda(initial, boundary, block):
    engine = CUDAEngine(initial, boundary, block)
    engine.advance(1)
    cuda.synchronize()

def measure_cuda(initial, iterations, boundary, block):
    """Warm context/JIT first. Total includes allocations, H2D, kernels and D2H."""
    if not cuda_status()["available"]:
        raise RuntimeError("Real CUDA hardware required for performance measurement")
    start_event, stop_event = cuda.event(timing=True), cuda.event(timing=True)
    cuda.synchronize()
    total_start = perf_counter()
    engine = CUDAEngine(initial, boundary, block)
    start_event.record()
    engine.advance(iterations)
    stop_event.record()
    stop_event.synchronize()
    final = engine.result()
    total_ms = (perf_counter() - total_start) * 1000
    # The event interval covers queued launches, including any GPU idle gaps.
    compute_ms = cuda.event_elapsed_time(start_event, stop_event)
    return final, {"compute_ms": float(compute_ms), "total_ms": total_ms}
