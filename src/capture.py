"""Fast screen capture in PHYSICAL pixels (Windows).

Uses mss for portability. Because the process is per-monitor DPI aware (set at
startup), mss coordinates are physical pixels in the virtual-desktop space —
the same basis as the rectangles from ``displays.enumerate_monitors``.
"""
from __future__ import annotations

from pathlib import Path

import mss
import numpy as np


class ScreenCapture:
    def __init__(self):
        self._sct = mss.MSS()

    def grab(self, left: int, top: int, width: int, height: int) -> np.ndarray:
        """Capture a physical-pixel region -> BGR uint8 array, shape (H, W, 3)."""
        shot = self._sct.grab({"left": int(left), "top": int(top),
                               "width": int(width), "height": int(height)})
        img = np.asarray(shot)                     # BGRA, (H, W, 4)
        return np.ascontiguousarray(img[:, :, :3])  # -> BGR

    def close(self) -> None:
        try:
            self._sct.close()
        except Exception:
            pass


def save_image(path: str | Path, img_bgr: np.ndarray) -> None:
    """Write a BGR array to disk (PNG inferred from extension)."""
    import cv2
    cv2.imwrite(str(path), img_bgr)
