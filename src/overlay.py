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

# Arrow style encodes the eval (player POV). A move that keeps/gives an advantage
# is GREEN, fading from faint near 0.00 to rich and opaque as it gets better; a
# move whose best outcome still leaves the player worse off is GREY. The
# opponent's predicted move is solid RED. The eval number is green when >= 0 and
# red when < 0.
GREEN = (40, 200, 80)
GREY = (150, 150, 150)
RED = (225, 60, 60)
FAINT_ALPHA = 70           # opacity of the weakest / a clustered move (still visible)
SPREAD_FULL_CP = 10.0      # eval spread (cp) among the shown moves at which contrast is full


def _advantage(ann: "Annotation") -> float:
    """ABSOLUTE advantage in pawns (+ White better, - Black better); mate clamps large."""
    if ann.mate is not None:
        return 99.0 if ann.mate > 0 else -99.0
    return (ann.score_cp / 100.0) if ann.score_cp is not None else 0.0


def _player_value(s) -> float:
    """Player-POV value (higher = better for the player) used to rank the moves."""
    if s.mate_in is not None:
        return 1e5 if s.mate_in > 0 else -1e5
    return float(s.score_cp) if s.score_cp is not None else 0.0


def _relative_strengths(suggestions) -> list[float]:
    """0..1 per move — how much it stands out from the others. A tight pack of
    near-equal moves stays near 0 (faint); a clear breakout reaches ~1 (bright)."""
    if not suggestions:
        return []
    if len(suggestions) == 1:
        return [1.0]
    vals = [_player_value(s) for s in suggestions]
    lo, hi = min(vals), max(vals)
    spread = hi - lo
    if spread < 1e-6:
        return [0.0] * len(suggestions)
    weight = min(1.0, spread / SPREAD_FULL_CP)
    return [((v - lo) / spread) * weight for v in vals]


def _arrow_color(ann: "Annotation") -> QtGui.QColor:
    if ann.opponent:
        return QtGui.QColor(*RED, 235)
    alpha = int(round(FAINT_ALPHA + (255 - FAINT_ALPHA) * max(0.0, min(1.0, ann.strength))))
    return QtGui.QColor(*(GREEN if _advantage(ann) >= 0 else GREY), alpha)


def _eval_text_color(ann: "Annotation") -> QtGui.QColor:
    return QtGui.QColor(255, 95, 95) if _advantage(ann) < 0 else QtGui.QColor(120, 240, 150)


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
    opponent: bool = False  # True => draw red (predicted opponent move)
    score_cp: int | None = None    # ABSOLUTE centipawns (+ White, - Black) -> colour
    mate: int | None = None        # ABSOLUTE mate distance (+ White mates, - Black mates)
    strength: float = 1.0          # 0..1 relative standout among shown moves -> opacity


def build_annotations(suggestions, opp_move=None, show_opponent=True,
                      white_to_move=True) -> list["Annotation"]:
    """Turn engine output into arrows under the fixed tempo model:

      * the player's moves — GREEN when the resulting position favours White, GREY
        when it favours Black; opacity is RELATIVE to the other shown moves; the
        eval number is ABSOLUTE (+ White, - Black);
      * RED — the opponent's single predicted best move (only when looking ahead).

    ``suggestions`` are ``MoveSuggestion`` with player-POV scores; ``white_to_move``
    is the analysed board's side to move, used to flip those scores to absolute."""
    anns: list[Annotation] = []
    if opp_move is not None and show_opponent:
        anns.append(Annotation(move=opp_move, rank=1, opponent=True))
    flip = not white_to_move
    for s, strength in zip(suggestions, _relative_strengths(suggestions)):
        abs_cp = (-s.score_cp if (flip and s.score_cp is not None) else s.score_cp)
        abs_mate = (-s.mate_in if (flip and s.mate_in is not None) else s.mate_in)
        anns.append(Annotation(move=s.move, rank=s.rank, opponent=False,
                               label=s.eval_text_pov(flip), score_cp=abs_cp,
                               mate=abs_mate, strength=strength))
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
            self._draw_move(painter, ann)

    def _draw_move(self, painter: QtGui.QPainter, ann: Annotation) -> None:
        g = self._geometry
        color = _arrow_color(ann)
        src = g.square_center(ann.move.from_square)
        dst = g.square_center(ann.move.to_square)
        radius = g.square * 0.42

        painter.setPen(QtGui.QPen(color, max(3.0, g.square * 0.06)))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawEllipse(src, radius, radius)      # circle the piece
        painter.drawEllipse(dst, radius, radius)      # circle the destination

        self._draw_arrow(painter, src, dst, color, g.square)
        if ann.label:
            self._draw_label(painter, dst, ann.label, _eval_text_color(ann), g.square)

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
                    text_color: QtGui.QColor, square: float) -> None:
        font = painter.font()
        font.setPixelSize(max(10, int(square * 0.22)))
        font.setBold(True)
        painter.setFont(font)
        box = QtCore.QRectF(at.x() - square / 2, at.y() - square / 2,
                            square, square * 0.32)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(15, 15, 15, 205))   # dark chip so the number reads anywhere
        painter.drawRoundedRect(box, 4, 4)
        painter.setPen(QtGui.QPen(text_color))
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
