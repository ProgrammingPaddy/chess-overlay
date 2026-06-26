"""Manual board calibration: drag a box over the board.

Returns the board in BOTH coordinate spaces:
  * a capture region in global PHYSICAL pixels (for screen capture), and
  * a ``BoardGeometry`` in global LOGICAL pixels (for the overlay).

The monitor's physical origin and scale come from ``MonitorFromWindow`` on the
selector window — reliable regardless of DPI or monitor layout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
from PySide6 import QtCore, QtGui, QtWidgets

from src.board_detect import find_board
from src.capture import ScreenCapture
from src.displays import MonitorInfo, monitor_from_hwnd
from src.overlay import BoardGeometry


@dataclass
class CalibrationResult:
    screen_index: int
    phys_left: int          # capture region, global physical px
    phys_top: int
    phys_side: int          # square board (width == height)
    geometry: BoardGeometry  # global logical px, for the overlay


class RegionSelector(QtWidgets.QWidget):
    """A translucent, interactive full-monitor window to drag-select a square."""

    def __init__(self, screen: QtGui.QScreen, rough: bool = False):
        super().__init__()
        self._screen = screen
        self._rough = rough
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setScreen(screen)
        self.setGeometry(screen.geometry())
        self.setCursor(QtCore.Qt.CrossCursor)
        self.monitor: MonitorInfo | None = None
        self._origin: QtCore.QPointF | None = None
        self._current: QtCore.QPointF | None = None
        self._result: QtCore.QRectF | None = None
        self._loop = QtCore.QEventLoop()

    def select(self) -> Optional[QtCore.QRectF]:
        """Show, identify the monitor, then block until the user finishes."""
        self.show()
        self.raise_()
        self.activateWindow()
        QtWidgets.QApplication.processEvents()
        self.monitor = monitor_from_hwnd(int(self.winId()))
        self._loop.exec()
        return self._result

    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        if e.button() == QtCore.Qt.LeftButton:
            self._origin = self._current = e.position()
            self.update()

    def mouseMoveEvent(self, e: QtGui.QMouseEvent) -> None:
        if self._origin is not None:
            self._current = e.position()
            self.update()

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent) -> None:
        if e.button() == QtCore.Qt.LeftButton and self._origin is not None:
            self._result = QtCore.QRectF(self._origin, e.position()).normalized()
            self._finish()

    def keyPressEvent(self, e: QtGui.QKeyEvent) -> None:
        if e.key() == QtCore.Qt.Key_Escape:
            self._result = None
            self._finish()

    def _finish(self) -> None:
        if self._loop.isRunning():
            self._loop.quit()
        self.close()

    def paintEvent(self, e: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 90))
        p.setPen(QtGui.QColor(255, 255, 255))
        font = p.font()
        font.setPixelSize(18)
        font.setBold(True)
        p.setFont(font)
        msg = ("Drag a ROUGH box around the board — it snaps to the exact board.  Esc to cancel."
               if self._rough else
               "Drag a box over the board (corner to corner).  Esc to cancel.")
        p.drawText(self.rect().adjusted(0, 24, 0, 0),
                   QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop, msg)
        if self._origin is not None and self._current is not None:
            rect = QtCore.QRectF(self._origin, self._current).normalized()
            p.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)
            p.fillRect(rect, QtCore.Qt.transparent)
            p.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)
            p.setPen(QtGui.QPen(QtGui.QColor(60, 200, 90), 2))
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawRect(rect)
            side = (rect.width() + rect.height()) / 2
            p.setPen(QtGui.QColor(230, 230, 230))
            p.drawText(rect.adjusted(2, -22, 0, 0),
                       QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop,
                       f"{int(side)}x{int(side)} px  (square ≈ {side / 8:.0f}px)")


def _build_result(screen, screen_index, monitor, lx, ly, side_logical,
                  white_bottom) -> CalibrationResult:
    """A CalibrationResult from a board rect given in this screen's logical px."""
    scale = monitor.scale if monitor else screen.devicePixelRatio()
    m_left = monitor.left if monitor else int(round(screen.geometry().x() * scale))
    m_top = monitor.top if monitor else int(round(screen.geometry().y() * scale))
    geometry = BoardGeometry(
        origin_x=screen.geometry().x() + lx,        # global logical
        origin_y=screen.geometry().y() + ly,
        square=side_logical / 8.0,
        white_bottom=white_bottom,
    )
    return CalibrationResult(
        screen_index=screen_index,
        phys_left=int(round(m_left + lx * scale)),   # global physical
        phys_top=int(round(m_top + ly * scale)),
        phys_side=int(round(side_logical * scale)),
        geometry=geometry,
    )


def _pick_screen(app, screen_index):
    screens = app.screens()
    i = max(0, min(screen_index, len(screens) - 1))
    return i, screens[i]


def calibrate(app: QtWidgets.QApplication, screen_index: int,
              white_bottom: bool) -> Optional[CalibrationResult]:
    """Manual calibration — drag a box over the board (the box IS the region)."""
    screen_index, screen = _pick_screen(app, screen_index)
    selector = RegionSelector(screen)
    rect = selector.select()
    if rect is None or rect.width() < 16 or rect.height() < 16:
        return None
    side = (rect.width() + rect.height()) / 2.0
    return _build_result(screen, screen_index, selector.monitor,
                         rect.x(), rect.y(), side, white_bottom)


def auto_calibrate(app: QtWidgets.QApplication, screen_index: int,
                   white_bottom: bool) -> Optional[CalibrationResult]:
    """Auto-align calibration — drag a ROUGH box; snap to the exact board.

    Returns the SAME result type as ``calibrate`` (so it feeds identical
    downstream state), or None if no board is confidently found (caller falls
    back to a message / manual mode). The piece recognition is untouched."""
    screen_index, screen = _pick_screen(app, screen_index)
    selector = RegionSelector(screen, rough=True)
    rect = selector.select()
    if rect is None or rect.width() < 48 or rect.height() < 48:
        return None
    selector.hide()
    for _ in range(6):                      # let the selector overlay leave the screen
        QtWidgets.QApplication.processEvents()
        QtCore.QThread.msleep(15)

    monitor = selector.monitor
    scale = monitor.scale if monitor else screen.devicePixelRatio()
    m_left = monitor.left if monitor else int(round(screen.geometry().x() * scale))
    m_top = monitor.top if monitor else int(round(screen.geometry().y() * scale))
    px, py = int(round(m_left + rect.x() * scale)), int(round(m_top + rect.y() * scale))
    pw, ph = int(round(rect.width() * scale)), int(round(rect.height() * scale))

    cap = ScreenCapture()
    try:
        img = cap.grab(px, py, pw, ph)
    except Exception:
        return None
    finally:
        cap.close()

    region = find_board(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if region is None:
        return None
    dx, dy, dw, dh = region                 # board within the grabbed region (physical px)
    side_phys = (dw + dh) / 2.0
    return _build_result(screen, screen_index, monitor,
                         rect.x() + dx / scale, rect.y() + dy / scale,
                         side_phys / scale, white_bottom)
