"""Chess Overlay — launches the control panel (which owns the overlay).

The options menu is now the app hub: monitor selection, board calibration,
engine settings, and position analysis all live there. Run:

    python main.py
"""
from __future__ import annotations

import sys

from PySide6 import QtWidgets


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
    _enable_dpi_awareness()
    app = QtWidgets.QApplication(sys.argv)
    from src.menu import MenuWindow      # imported after QApplication exists
    menu = MenuWindow(app)
    menu.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
