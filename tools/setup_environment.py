"""Create a project-local environment on Windows, Linux or macOS."""
import argparse
from pathlib import Path
import platform
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true", help="Also install the optional CUDA 12 packages")
    parser.add_argument("--web", action="store_true", help="Install only the browser/CLI dependencies (no desktop)")
    args = parser.parse_args(argv)
    if not (3, 10) <= sys.version_info[:2] <= (3, 13):
        parser.error("Use Python 3.10-3.13; Python 3.12 is recommended for a new setup.")
    if args.cuda and platform.system() not in ("Linux", "Windows"):
        parser.error("This CUDA implementation needs Windows/Linux and an NVIDIA GPU. Omit --cuda on macOS.")
    folder = ROOT / ".venv"
    python = folder / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    try:
        if not python.exists():
            if folder.exists():
                parser.error("An incomplete .venv already exists. Rename it before trying setup again.")
            print(f"Creating {folder} with Python {platform.python_version()}", flush=True)
            venv.EnvBuilder(with_pip=True).create(folder)
        else:
            print("Reusing .venv; existing packages will be resolved against the requirements.", flush=True)
        subprocess.run([str(python), "-c", "import sys; raise SystemExit(0 if (3,10)<=sys.version_info[:2]<=(3,13) else 1)"], check=True)
        subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"], check=True, cwd=ROOT)
        requirements = "requirements.txt" if args.web else "requirements-desktop.txt"
        command = [str(python), "-m", "pip", "install", "-r", requirements]
        if args.cuda:
            command += ["-r", "requirements-cuda.txt"]
        subprocess.run(command, check=True, cwd=ROOT)
        check = [str(python), "app.py", "--check"]
        if args.web:
            check.append("--web")
        if args.cuda:
            check.append("--require-cuda")
        subprocess.run(check, check=True, cwd=ROOT)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Setup did not complete: {exc}\nSee docs/INSTALL.md and rerun after fixing the reported issue.", file=sys.stderr)
        return 1
    print(f'\nReady. Run: "{python}" app.py' + (" --web" if args.web else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
