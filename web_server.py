"""Optional browser dashboard: python app.py --web, then http://127.0.0.1:8000."""
import argparse
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import threading
from urllib.parse import urlparse
import uuid
import numpy as np
from life.core import Config, make_grid, validate_grid
from life.cpu import CPUEngine
from life.gpu import cuda_status
from life.benchmark import suite, write_csv, save_report

ROOT = Path(__file__).resolve().parent
LOCK = threading.Lock()
JOB = {"id": None, "state": "idle"}

def run_job(job_id, configs, mode):
    def progress(index, total, size):
        with LOCK:
            JOB.update(progress=index / total, message=f"Measuring {size} × {size} · {index+1}/{total}")
    try:
        report = suite(configs, mode, progress)
        save_report(report, ROOT / "results" / "dashboard")
        with LOCK:
            JOB.update(state="done", progress=1, report=report, message="Measurements complete")
    except Exception as exc:
        with LOCK:
            JOB.update(state="error", message=str(exc))

class Handler(BaseHTTPRequestHandler):
    def send(self, payload, status=200, content_type="application/json"):
        body = json.dumps(payload).encode() if content_type == "application/json" else payload
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/status":
            return self.send({"cuda": cuda_status(), "cpu": "NumPy 2D arrays"})
        if path == "/api/job":
            with LOCK:
                snapshot = dict(JOB)
            return self.send(snapshot)
        if path in ("/api/export.json", "/api/export.csv"):
            with LOCK:
                report = JOB.get("report")
            if not report:
                return self.send({"error": "Run a benchmark first"}, 404)
            if path.endswith("json"):
                return self.send(report)
            stream = io.StringIO()
            write_csv(report, stream)
            return self.send(stream.getvalue().encode(), content_type="text/csv")
        files = {"/": ("index.html", "text/html; charset=utf-8"),
                 "/style.css": ("style.css", "text/css"),
                 "/app.js": ("app.js", "text/javascript")}
        if path not in files:
            return self.send({"error": "Not found"}, 404)
        name, mime = files[path]
        self.send((ROOT / "web" / name).read_bytes(), content_type=mime)

    def do_POST(self):
        # No cross-origin writes to this loopback application.
        origin = self.headers.get("Origin")
        if origin and origin != "http://" + self.headers.get("Host", ""):
            return self.send({"error": "Cross-origin requests are not accepted"}, 403)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 2_000_000:
                raise ValueError("Invalid request size")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object")
            path = urlparse(self.path).path
            if path == "/api/grid":
                config = Config(**data)
                if config.size > 256:
                    raise ValueError("Live preview is limited to 256; benchmark supports 2048")
                grid = make_grid(config)
                return self.send({"grid": grid.tolist(), "alive": int(grid.sum())})
            if path == "/api/step":
                a = validate_grid(data["grid"])
                if max(a.shape) > 256:
                    raise ValueError("Live preview maximum is 256")
                steps = data.get("steps", 1)
                if type(steps) is not int or not 1 <= steps <= 20:
                    raise ValueError("steps must be 1 to 20")
                grid = CPUEngine(a, data.get("boundary", "wrap")).advance(steps)
                return self.send({"grid": grid.tolist(), "alive": int(grid.sum())})
            if path == "/api/benchmark":
                config = Config(**data.get("config", {}))
                sizes = data.get("sizes", [config.size])
                if not isinstance(sizes, list) or not 1 <= len(sizes) <= 5:
                    raise ValueError("Choose one to five grid sizes")
                configs = [replace(config, size=size) for size in sizes]
                mode = data.get("mode", "both")
                if mode not in ("cpu", "both", "cuda"):
                    raise ValueError("Invalid mode")
                if sum(c.size*c.size*c.iterations*c.repeats for c in configs) > 3_000_000_000:
                    raise ValueError("Workload too large for dashboard; reduce repeats/iterations or use CLI")
                with LOCK:
                    if JOB["state"] == "running":
                        return self.send({"error": "A benchmark is already running"}, 409)
                    job_id = uuid.uuid4().hex
                    JOB.clear()
                    JOB.update(id=job_id, state="running", progress=0, message="Warming up engines…")
                threading.Thread(target=run_job, args=(job_id, configs, mode), daemon=True).start()
                return self.send({"id": job_id}, 202)
            return self.send({"error": "Not found"}, 404)
        except (ValueError, TypeError, KeyError) as exc:
            self.send({"error": str(exc)}, 400)
        except Exception as exc:
            self.send({"error": str(exc)}, 500)

    def log_message(self, fmt, *args):
        pass

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Life Lab → http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nLife Lab stopped.")
    finally:
        server.server_close()

if __name__ == "__main__":
    main()
