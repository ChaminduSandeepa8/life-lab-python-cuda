"""Life Lab entry point: native Python windows by default; --web for the browser UI."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web", action="store_true", help="Launch the original localhost dashboard instead")
    parser.add_argument("--port", type=int, default=8000, help="Port for --web (default: 8000)")
    parser.add_argument("--check", action="store_true", help="Check dependencies and CPU/CUDA execution, then exit")
    parser.add_argument("--require-cuda", action="store_true", help="Fail --check unless a real CUDA kernel works")
    args = parser.parse_args()
    if args.require_cuda and not args.check:
        parser.error("--require-cuda must be used with --check")
    if args.check:
        from tools.check_environment import check
        return check(web=args.web, require_cuda=args.require_cuda)
    if args.web:
        from web_server import main as web_main
        return web_main(["--port", str(args.port)])
    try:
        from desktop.application import main as desktop_main
    except ImportError as exc:
        parser.exit(1, f"Desktop dependency unavailable: {exc}\n"
                    "Install: python -m pip install -r requirements-desktop.txt\n"
                    "Tkinter is also required. See docs/INSTALL.md for your operating system.\n")
    import tkinter as tk
    try:
        desktop_main()
    except tk.TclError as exc:
        parser.exit(1, f"Could not open the desktop: {exc}\n"
                    "Run in a graphical desktop session with working Tcl/Tk.\n"
                    "For a server or remote terminal, use python app.py --web instead.\n"
                    "See docs/INSTALL.md.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
