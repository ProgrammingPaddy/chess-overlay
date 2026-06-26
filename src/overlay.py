"""Transparent, click-through overlay (Windows), DPI-correct across monitors.

Design
------
* ONE overlay window per monitor (``_OverlayWindow``), each covering exactly its
  screen. A window therefore always lives in a single DPI context, so Qt scales
  it correctly and Windows never bitmap-stretches it across a monitor boundary.
* ``OverlayManager`` owns those windows and broadcasts geometry/annotations to
  all of them. Each window clips to its own screen, so a board on monitor 2 is
  simply drawn by monitor 2's window.
* Coordinate contract: every coordinate here is in GLOBAL, device-independent
  (Qt "logical") pixels. The capture/CV layer is the single place that converts
  physical capture pixels into this space (physical / devicePixelRatio + screen
  origin), so DPI math lives in exactly one spot and never leaks in here.

Click-through is enforced both at the Qt level (WA_TransparentForMouseEvents)
and the Win32 level (WS_EX_LAYERED | WS_EX_TRANSPARENT) for robustness.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import ctypes

import chess
from PySide6 import QtCore, QtGui, QtWidgets

import win32con
import win32gui

WDA_EXCLUDEFROMCAPTURE = 0x11   # window is visible to the user but not to screen capture

# Player moves in green, opponent (predicted) moves in red. Within each family
# the strongest move is the most opaque/visible; all three stay readable.
GREEN_SHADES = [
    QtGui.QColor(40, 200, 80, 255),
    QtGui.QColor(70, 205, 105, 190),
    QtGui.QColor(105, 215, 135, 140),
]
RED_SHADES = [
    QtGui.QColor(225, 55, 55, 255),
    QtGui.QColor(230, 95, 95, 190),
    QtGui.QColor(235, 135, 135, 140),
]


@dataclass
class BoardGeometry:
    """Board placement in GLOBAL, device-independent (logical) pixels."""
    origin_x: float        # left edge of the a-file
    origin_y: float        # top edge of the 8th rank
    square: float          # square size
    white_bottom: bool = True

    def square_center(self, sq: chess.Square) -> QtCore.QPointF:
        file = chess.square_file(sq)        # 0..7 => a..h
        rank = chess.square_rank(sq)        # 0..7 => 1..8
        if self.white_bottom:
            col, row = file, 7 - rank
        else:
            col, row = 7 - file, rank
        x = self.origin_x + (col + 0.5) * self.square
        y = self.origin_y + (row + 0.5) * self.square
        return QtCore.QPointF(x, y)


@dataclass
class Annotation:
    """One move to draw on the board."""
    move: chess.Move
    rank: int = 1
    label: str = ""        # optional text near the destination (e.g. the eval)
    opponent: bool = False  # True => draw red (predicted opponent move), else green


def build_annotations(suggestions, opp_move=None, show_opponent=True) -> list["Annotation"]:
    """Turn engine output into arrows under the fixed tempo model:

      * GREEN = the player's moves (always the analysed side, so eval labels are
        already from the player's point of view).
      * RED  = the opponent's single predicted best move, present ONLY when we
        looked ahead (it is the opponent's turn). One red arrow, never a fan of
        them — the move the green prep assumes.

    ``suggestions`` are ``MoveSuggestion``; ``opp_move`` is a ``chess.Move`` or
    ``None`` (None => it is the player's turn, no look-ahead, no red)."""
    anns: list[Annotation] = []
    if opp_move is not None and show_opponent:
        anns.append(Annotation(move=opp_move, rank=1, opponent=True))
    for s in suggestions:
        anns.append(Annotation(move=s.move, rank=s.rank, label=s.eval_text(), opponent=False))
    return anns


def visible_annotations(board, annotations):
    """Keep only arrows whose from-square holds a piece on ``board``.

    The engine's moves are legal for the position it analysed, but the live board
    may have moved on. Hiding arrows whose source square is now empty stops a
    moved piece's stale arrow from lingering over an empty square."""
    if board is None:
        return list(annotations)
    return [a for a in annotations if board.piece_at(a.move.from_square) is not None]


class _OverlayWindow(QtWidgets.QWidget):
    """A single transparent click-through window pinned to one monitor."""

    def __init__(self, screen: QtGui.QScreen):
        super().__init__()
        self._screen = screen
        self._geometry: BoardGeometry | None = None
        self._annotations: list[Annotation] = []
        self._visible = True

        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.Tool                # keep out of taskbar / alt-tab
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.setScreen(screen)
        self.setGeometry(screen.geometry())   # cover exactly this monitor

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        # Pin to the right screen and make the native window click-through.
        handle = self.windowHandle()
        if handle is not None:
            handle.setScreen(self._screen)
        self.setGeometry(self._screen.geometry())
        hwnd = int(self.winId())
        ex = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        ex |= (win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT
               | win32con.WS_EX_TOOLWINDOW | win32con.WS_EX_NOACTIVATE)
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, ex)
        # Keep our own arrows out of screen captures so vision never reads them.
        try:
            ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
        except Exception:
            pass

    # --------------------------------------------------------------- state
    def apply(self, geometry: BoardGeometry | None,
              annotations: list[Annotation], visible: bool) -> None:
        self._geometry = geometry
        self._annotations = annotations
        self._visible = visible
        self.update()

    # -------------------------------------------------------------- paint
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        if not (self._visible and self._geometry and self._annotations):
            return
        # Window-local (0,0) == this screen's global top-left, so shift global
        # logical coordinates into local space and let Qt handle this monitor's
        # devicePixelRatio.
        origin = self._screen.geometry().topLeft()
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.translate(-origin.x(), -origin.y())
        # Draw lower-ranked / opponent moves first so the best player move is on top.
        order = sorted(self._annotations, key=lambda a: (not a.opponent, -a.rank))
        for ann in order:
            shades = RED_SHADES if ann.opponent else GREEN_SHADES
            color = shades[(ann.rank - 1) % len(shades)]
            self._draw_move(painter, ann, color)

    def _draw_move(self, painter: QtGui.QPainter, ann: Annotation,
                   color: QtGui.QColor) -> None:
        g = self._geometry
        src = g.square_center(ann.move.from_square)
        dst = g.square_center(ann.move.to_square)
        radius = g.square * 0.42

        painter.setPen(QtGui.QPen(color, max(3.0, g.square * 0.06)))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawEllipse(src, radius, radius)      # circle the piece
        painter.drawEllipse(dst, radius, radius)      # circle the destination

        self._draw_arrow(painter, src, dst, color, g.square)
        if ann.label:
            self._draw_label(painter, dst, ann.label, color, g.square)

    @staticmethod
    def _draw_arrow(painter: QtGui.QPainter, start: QtCore.QPointF,
                    end: QtCore.QPointF, color: QtGui.QColor, square: float) -> None:
        dx, dy = end.x() - start.x(), end.y() - start.y()
        dist = math.hypot(dx, dy)
        if dist < 1.0:
            return
        ux, uy = dx / dist, dy / dist
        pad = square * 0.45            # begin/end outside the circles
        sx, sy = start.x() + ux * pad, start.y() + uy * pad
        ex, ey = end.x() - ux * pad, end.y() - uy * pad

        shaft = QtGui.QPen(color, max(3.0, square * 0.10))
        shaft.setCapStyle(QtCore.Qt.RoundCap)
        painter.setPen(shaft)
        painter.drawLine(QtCore.QPointF(sx, sy), QtCore.QPointF(ex, ey))

        head = square * 0.28
        ang = math.atan2(ey - sy, ex - sx)
        left, right = ang + math.radians(150), ang - math.radians(150)
        path = QtGui.QPainterPath(QtCore.QPointF(ex, ey))
        path.lineTo(ex + head * math.cos(left), ey + head * math.sin(left))
        path.lineTo(ex + head * math.cos(right), ey + head * math.sin(right))
        path.closeSubpath()
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(color)
        painter.drawPath(path)

    @staticmethod
    def _draw_label(painter: QtGui.QPainter, at: QtCore.QPointF, text: str,
                    color: QtGui.QColor, square: float) -> None:
        font = painter.font()
        font.setPixelSize(max(10, int(square * 0.22)))
        font.setBold(True)
        painter.setFont(font)
        box = QtCore.QRectF(at.x() - square / 2, at.y() - square / 2,
                            square, square * 0.3)
        painter.fillRect(box, QtGui.QColor(color.red(), color.green(),
                                           color.blue(), 220))
        painter.setPen(QtGui.QPen(QtGui.QColor(20, 20, 20)))
        painter.drawText(box, QtCore.Qt.AlignCenter, text)


class OverlayManager:
    """Owns one ``_OverlayWindow`` per monitor and broadcasts state to all."""

    def __init__(self, app: QtWidgets.QApplication):
        self._geometry: BoardGeometry | None = None
        self._annotations: list[Annotation] = []
        self._visible = True
        self._windows = [_OverlayWindow(s) for s in app.screens()]

    def _refresh(self) -> None:
        for w in self._windows:
            w.apply(self._geometry, self._annotations, self._visible)

    def set_board_geometry(self, geometry: BoardGeometry) -> None:
        self._geometry = geometry
        self._refresh()

    def set_annotations(self, annotations: list[Annotation]) -> None:
        self._annotations = annotations
        self._refresh()

    def clear(self) -> None:
        self.set_annotations([])

    def set_overlay_visible(self, visible: bool) -> None:
        """Master On/Off for the board UI (the desired toggle feature)."""
        self._visible = visible
        self._refresh()

    def show(self) -> None:
        for w in self._windows:
            w.show()
        self._refresh()
