"""Package only Git-tracked project files, excluding private local artifacts."""
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    listing = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True)
    paths = [Path(p.decode("utf-8")) for p in listing.stdout.split(b"\0") if p]
    if not paths:
        raise SystemExit("No tracked source files. Use a Git clone or download GitHub's source ZIP.")
    target = ROOT / "output" / "life-lab-project.zip"
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in paths:
            path = ROOT / relative
            if path.is_file():
                archive.write(path, Path("life-lab-python-cuda") / relative)
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        print(f"{target}: {len(archive.namelist())} files, {target.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
