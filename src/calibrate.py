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

from PySide6 import QtCore, QtGui, QtWidgets

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

    def __init__(self, screen: QtGui.QScreen):
        super().__init__()
        self._screen = screen
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
        p.drawText(self.rect().adjusted(0, 24, 0, 0),
                   QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop,
                   "Drag a box over the board (corner to corner).  Esc to cancel.")
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


def calibrate(app: QtWidgets.QApplication, screen_index: int,
              white_bottom: bool) -> Optional[CalibrationResult]:
    screens = app.screens()
    screen_index = max(0, min(screen_index, len(screens) - 1))
    screen = screens[screen_index]

    selector = RegionSelector(screen)
    rect = selector.select()
    if rect is None or rect.width() < 16 or rect.height() < 16:
        return None

    side_logical = (rect.width() + rect.height()) / 2.0
    lx, ly = rect.x(), rect.y()

    mon = selector.monitor
    scale = mon.scale if mon else screen.devicePixelRatio()
    m_left = mon.left if mon else int(round(screen.geometry().x() * scale))
    m_top = mon.top if mon else int(round(screen.geometry().y() * scale))

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
