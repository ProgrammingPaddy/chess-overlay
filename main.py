"""Chess Overlay — launches the control panel (which owns the overlay).

The options menu is now the app hub: monitor selection, board calibration,
engine settings, and position analysis all live there. Run:

    python main.py
"""
from __future__ import annotations

import sys
import threading
import time
import traceback
from pathlib import Path

from PySide6 import QtWidgets

CRASH_LOG = Path(__file__).resolve().parent / "debug" / "crash.log"


def _write_crash(where: str, exc_type, exc, tb) -> None:
    try:
        CRASH_LOG.parent.mkdir(exist_ok=True)
        with open(CRASH_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')}  [{where}] ===\n")
            traceback.print_exception(exc_type, exc, tb, file=f)
    except Exception:
        pass
    traceback.print_exception(exc_type, exc, tb)   # also to stderr


def _install_crash_logging() -> None:
    """Log every unhandled exception (main thread, worker threads, and — on
    PySide6 — exceptions escaping Qt slots/virtuals) to debug/crash.log instead of
    dying silently. PySide6 routes slot exceptions through sys.excepthook, so the
    app keeps running after a logged error rather than aborting."""
    def excepthook(exc_type, exc, tb):
        _write_crash("uncaught", exc_type, exc, tb)
    sys.excepthook = excepthook

    def thread_hook(args):
        _write_crash(f"thread:{args.thread.name}", args.exc_type,
                     args.exc_value, args.exc_traceback)
    threading.excepthook = thread_hook


def _enable_dpi_awareness() -> None:
    """Mark the process per-monitor DPI aware before Qt starts, so screen
    capture reports true physical pixels."""
    import ctypes
    for attempt in (
        lambda: ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)),
        lambda: ctypes.windll.shcore.SetProcessDpiAwareness(2),
        lambda: ctypes.windll.user32.SetProcessDPIAware(),
    ):
        try:
            attempt()
            return
        except Exception:
            continue


def main() -> int:
    _install_crash_logging()
    _enable_dpi_awareness()
    app = QtWidgets.QApplication(sys.argv)
    from src.menu import MenuWindow      # imported after QApplication exists
    menu = MenuWindow(app)
    menu.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
