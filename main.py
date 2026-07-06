"""Chess Overlay — launches the control panel (which owns the overlay).

The options menu is now the app hub: monitor selection, board calibration,
engine settings, and position analysis all live there. Run:

    python main.py
"""
from __future__ import annotations

import faulthandler
import gc
import sys
import threading
import time
import traceback
from pathlib import Path

from PySide6 import QtCore, QtWidgets

CRASH_LOG = Path(__file__).resolve().parent / "debug" / "crash.log"
_fault_fh = None        # kept alive for the process lifetime so faulthandler can write
_gc_timer = None        # kept alive so the GUI-thread collector keeps firing


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

    # A segfault (C-level — a native lib or a cross-thread data race) bypasses
    # sys.excepthook entirely; faulthandler dumps a Python+C traceback for ALL threads
    # to this file so such a crash isn't silent.
    global _fault_fh
    try:
        CRASH_LOG.parent.mkdir(exist_ok=True)
        _fault_fh = open(CRASH_LOG.parent / "faulthandler.log", "a", encoding="utf-8")
        faulthandler.enable(file=_fault_fh, all_threads=True)
    except Exception:
        try:
            faulthandler.enable(all_threads=True)   # fall back to stderr
        except Exception:
            pass


def _install_gc_guard() -> None:
    """Confine cyclic garbage collection to the GUI thread — the fix for the intermittent
    segfault in debug/faulthandler.log (always 'Garbage-collecting' inside python-chess's
    PV parser on the engine's background thread).

    python-chess drives the engine through an asyncio loop in a BACKGROUND thread.
    CPython's automatic cyclic collector runs on whichever thread crosses the allocation
    threshold — often that busy parser thread. When a sweep there reaches a cycle that
    touches a PySide6/Qt object, Qt's C++ deletion runs off the GUI thread and corrupts
    the heap → the crash. Disabling AUTOMATIC collection stops any background thread from
    ever running the collector (reference-count freeing is immediate and GIL-safe, so it
    is unaffected); a GUI-thread timer performs the ONLY cyclic sweeps, where Qt objects
    are safe to touch. Isolated engine churn never reproduced the crash; adding Qt did —
    this targets exactly that interaction. Guarded so it can never block startup."""
    global _gc_timer
    try:
        gc.disable()
        gc.collect()
        gc.freeze()                      # keep startup/import objects out of every sweep
        _gc_timer = QtCore.QTimer()
        _gc_timer.timeout.connect(gc.collect)
        _gc_timer.start(2000)            # sweep cycles ~every 2 s, on the GUI thread only
    except Exception:
        pass


def _teardown_gc_guard() -> None:
    """Undo the GC guard for a clean interpreter shutdown: stop the collector timer and
    re-enable automatic GC. Worker threads are already joined by MenuWindow.closeEvent, so
    the final teardown sweep no longer has a background thread to collide with."""
    global _gc_timer
    try:
        if _gc_timer is not None:
            _gc_timer.stop()
            _gc_timer = None
        gc.enable()
    except Exception:
        pass


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
    _install_gc_guard()                 # confine cyclic GC to the GUI thread (crash fix)
    from src.menu import MenuWindow      # imported after QApplication exists
    menu = MenuWindow(app)
    menu.show()
    try:
        return app.exec()
    finally:
        _teardown_gc_guard()            # stop the collector + re-enable GC for a clean exit


if __name__ == "__main__":
    raise SystemExit(main())
