"""Background vision thread: capture + recognize at high FPS, off the GUI thread.

Runs its own screen-capture instance in its own thread and emits each (placement
board, debug) reading to the GUI via a queued signal, so recognition never blocks
or stutters the interface. The owner stops the worker before recalibrating the
model (the only time the model is written), so there's no read/write race.
"""
from __future__ import annotations

from typing import Callable

from PySide6 import QtCore

from src.capture import ScreenCapture


class VisionWorker(QtCore.QThread):
    frame = QtCore.Signal(object, object)   # (placement board, debug list)

    def __init__(self, vision, region_fn: Callable, orient_fn: Callable,
                 interval_ms: int = 50, parent=None):
        super().__init__(parent)
        self._vision = vision
        self._region_fn = region_fn        # () -> (left, top, w, h) | None
        self._orient_fn = orient_fn        # () -> white_bottom: bool
        self._interval = interval_ms
        self._stop = False

    def run(self) -> None:
        capture = ScreenCapture()
        try:
            while not self._stop:
                region = self._region_fn()
                if region is None or not self._vision.calibrated:
                    self.msleep(80)
                    continue
                try:
                    img = capture.grab(*region)
                    board, debug = self._vision.analyze(img, self._orient_fn())
                    self.frame.emit(board, debug)
                except Exception:
                    pass
                self.msleep(self._interval)
        finally:
            capture.close()

    def stop(self) -> None:
        self._stop = True
        self.wait(1500)
