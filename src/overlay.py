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
DARK_RED = (120, 0, 12)    # opponent's dominant move (>=1 pawn / mate) — red analog of gold
GOLD = (255, 195, 30)      # the player's standout (>=1 pawn clear) or forced-mate move
FAINT_ALPHA = 45           # opacity of the weakest / a clustered move (still visible)
SPREAD_FULL_CP = 7.0       # eval spread (cp) among the shown moves at which contrast is full
GOLD_LEAD_CP = 100.0       # a non-mate move this many cp clear of the rest goes gold


def _advantage(ann: "Annotation") -> float:
    """ABSOLUTE advantage in pawns (+ White better, - Black better); mate clamps large."""
    if ann.mate is not None:
        return 99.0 if ann.mate > 0 else -99.0
    return (ann.score_cp / 100.0) if ann.score_cp is not None else 0.0


def _player_advantage(ann: "Annotation") -> float:
    """Advantage from the PLAYER's POV (+ good for the player). The eval NUMBER is
    shown ABSOLUTE (negative = Black winning), but the green/grey arrow and the
    number's colour must track whether the move is good FOR THE PLAYER — otherwise
    a winning Black player sees their best moves greyed out and their evals in red,
    because in absolute terms the advantage is negative. Side on bottom is the
    player, so this is exactly the absolute value flipped for a Black player."""
    a = _advantage(ann)
    return a if ann.player_is_white else -a


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


def _gold_strengths(suggestions) -> dict:
    """Indices -> gold opacity (0..1). A move is GOLD when it is a forced mate FOR
    the player (fastest mate full, longer mates fainter) or — absent a mate — a
    single move at least ~1 pawn clear of every other shown move."""
    mates = [(i, s.mate_in) for i, s in enumerate(suggestions)
             if s.mate_in is not None and s.mate_in > 0]
    if mates:
        fastest = min(m for _, m in mates)
        return {i: fastest / m for i, m in mates}
    vals = [_player_value(s) for s in suggestions]
    if len(vals) >= 2:
        order = sorted(range(len(vals)), key=lambda i: -vals[i])
        if vals[order[0]] - vals[order[1]] >= GOLD_LEAD_CP:
            return {order[0]: 1.0}
    return {}


def _arrow_color(ann: "Annotation") -> QtGui.QColor:
    alpha = int(round(FAINT_ALPHA + (255 - FAINT_ALPHA) * max(0.0, min(1.0, ann.strength))))
    if ann.opponent:                          # opponent's likely moves, in red
        if ann.gold:                          # an overwhelmingly strong reply -> very dark red
            return QtGui.QColor(*DARK_RED, max(alpha, 230))
        return QtGui.QColor(*RED, alpha)
    if ann.gold:                              # the player's standout / mate -> gold
        return QtGui.QColor(*GOLD, alpha)
    return QtGui.QColor(*(GREEN if _player_advantage(ann) >= 0 else GREY), alpha)


def _eval_text_color(ann: "Annotation") -> QtGui.QColor:
    if ann.opponent:
        return QtGui.QColor(255, 70, 70) if ann.gold else QtGui.QColor(255, 140, 140)
    if ann.gold:
        return QtGui.QColor(255, 220, 90)
    return QtGui.QColor(255, 95, 95) if _player_advantage(ann) < 0 else QtGui.QColor(120, 240, 150)


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
    score_cp: int | None = None    # ABSOLUTE centipawns (+ White, - Black) -> the number
    mate: int | None = None        # ABSOLUTE mate distance (+ White mates, - Black mates)
    strength: float = 1.0          # 0..1 relative standout among shown moves -> opacity
    gold: bool = False             # standout (>=1 pawn clear) or forced-mate move
    player_is_white: bool = True   # which colour the player is -> green/grey is player-POV


def _styled_set_policy(suggestions, opponent: bool, flip: bool, gold: bool,
                       player_is_white: bool) -> list["Annotation"]:
    """Policy engines (Maia 2): opacity = human-move likelihood relative to the
    most-likely move, label = that probability. The green/grey/red colour still
    comes from the position win probability (already mapped into ``score_cp``)."""
    pols = [float(s.policy or 0.0) for s in suggestions]
    mx = max(pols) if pols else 0.0
    golds = {pols.index(mx): 1.0} if (gold and mx >= 0.5) else {}   # a clearly dominant human move
    out = []
    for i, s in enumerate(suggestions):
        abs_cp = (-s.score_cp if (flip and s.score_cp is not None) else s.score_cp)
        dom = i in golds
        strength = (pols[i] / mx) if mx > 0 else 0.0
        out.append(Annotation(move=s.move, rank=s.rank, opponent=opponent,
                              label=f"{pols[i] * 100:.0f}%", score_cp=abs_cp, mate=None,
                              strength=(1.0 if dom else strength), gold=dom,
                              player_is_white=player_is_white))
    return out


def _styled_set(suggestions, opponent: bool, flip: bool, gold: bool,
                player_is_white: bool, policy_mode: bool = False) -> list["Annotation"]:
    """Annotations for one side: relative opacity within the set + a dominant flag,
    eval shown ABSOLUTE (+ White, - Black). ``flip`` negates the side-to-move score;
    ``player_is_white`` lets the green/grey colour track the PLAYER's POV.
    ``policy_mode`` switches to human-likelihood opacity + probability labels."""
    if policy_mode:
        return _styled_set_policy(suggestions, opponent, flip, gold, player_is_white)
    strengths = _relative_strengths(suggestions)
    golds = _gold_strengths(suggestions) if gold else {}
    out = []
    for i, s in enumerate(suggestions):
        abs_cp = (-s.score_cp if (flip and s.score_cp is not None) else s.score_cp)
        abs_mate = (-s.mate_in if (flip and s.mate_in is not None) else s.mate_in)
        dom = i in golds
        out.append(Annotation(move=s.move, rank=s.rank, opponent=opponent,
                              label=s.eval_text_pov(flip), score_cp=abs_cp, mate=abs_mate,
                              strength=(golds[i] if dom else strengths[i]), gold=dom,
                              player_is_white=player_is_white))
    return out


def build_annotations(suggestions, opp_suggestions=None, show_opponent=True,
                      white_to_move=True, gold_enabled=True,
                      policy_mode=False) -> list["Annotation"]:
    """Turn engine output into arrows under the fixed tempo model:

      * the player's moves — GOLD when a forced mate or clearly best (>= ~1 pawn),
        else GREEN (resulting position favours White) / GREY (favours Black);
      * the opponent's likely moves (look-ahead only) — the SAME rules in RED, with
        an overwhelmingly strong reply in very dark red.

    Opacity is RELATIVE within each side; eval numbers are ABSOLUTE (+ White,
    - Black). ``white_to_move`` is the player-position's side to move; since the
    analysed position is always one where it is the PLAYER's turn (we push the
    opponent's move first when looking ahead), ``white_to_move`` IS 'the player is
    White' — so it both flips the scores and selects the player's POV colour. The
    opponent moved one ply earlier, so their scores flip the opposite way."""
    player_is_white = white_to_move
    anns: list[Annotation] = []
    if opp_suggestions and show_opponent:
        anns += _styled_set(opp_suggestions, opponent=True, flip=white_to_move,
                            gold=gold_enabled, player_is_white=player_is_white,
                            policy_mode=policy_mode)
    anns += _styled_set(suggestions, opponent=False, flip=not white_to_move,
                        gold=gold_enabled, player_is_white=player_is_white,
                        policy_mode=policy_mode)
    return anns


# Solution-line fade: the move to play now is full strength; each step further down
# the forced line is fainter, floored so even deep moves stay visible.
LINE_FADE_STEP = 0.17
LINE_MIN_STRENGTH = 0.30


def build_puzzle_line(top, board, hero_is_white: bool, show_opponent: bool = True,
                      max_plies: int = 8, move_numbers: bool = False) -> list["Annotation"]:
    """Fading arrows for a WHOLE forced solution, drawn at once.

    ``top`` is the best line; its ``pv`` is walked from ``board`` (whose turn is the
    side to move). The hero (winning) side's moves are GREEN — the immediate one
    GOLD — and the opponent's forced replies RED; each step is fainter the deeper in
    the line it sits, so the move to play now is the most opaque. Opponent arrows are
    dropped when ``show_opponent`` is False ('winning side only').

    Labels: with ``move_numbers`` off, only the current move carries an eval, so the deeper
    arrows stay uncluttered. With it on, EACH shown arrow is numbered by its position in the
    displayed line (1 = the move to play now, 2 = next, …); moves that land on the same
    square share one combined label ('1,3') so the numbers never overprint."""
    anns: list[Annotation] = []
    if not top or not getattr(top, "pv", None):
        return anns
    flip = board.turn != chess.WHITE          # side-to-move POV -> ABSOLUTE (+White, -Black)
    abs_cp = -top.score_cp if (flip and top.score_cp is not None) else top.score_cp
    abs_mate = -top.mate_in if (flip and top.mate_in is not None) else top.mate_in
    walk = board.copy()
    shown = 0
    for k, move in enumerate(top.pv[:max_plies]):
        is_hero = ((walk.turn == chess.WHITE) == hero_is_white)
        if is_hero or show_opponent:
            shown += 1
            strength = max(LINE_MIN_STRENGTH, 1.0 - k * LINE_FADE_STEP)
            label = str(shown) if move_numbers else (top.eval_text_pov(flip) if k == 0 else "")
            anns.append(Annotation(
                move=move, rank=k + 1, opponent=not is_hero,
                label=label, score_cp=abs_cp, mate=abs_mate, strength=strength,
                gold=(is_hero and k == 0), player_is_white=hero_is_white))
        try:
            walk.push(move)
        except Exception:
            break
    if move_numbers:
        _combine_same_square_labels(anns)
    return anns


def _combine_same_square_labels(anns: list["Annotation"]) -> None:
    """Move-number labels sit on the destination square; when several moves land on the
    SAME square, merge their numbers onto one arrow ('1,3') and blank the rest so the
    labels don't overprint (kept on the earliest so its colour/side reads sensibly)."""
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for a in anns:
        if a.label:
            groups[a.move.to_square].append(a)
    for group in groups.values():
        if len(group) > 1:
            group[0].label = ",".join(a.label for a in group)
            for a in group[1:]:
                a.label = ""


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
        self._show_border = False
        self._orient: tuple | None = None      # (show, agree|None, confidence)

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
    def apply(self, geometry: BoardGeometry | None, annotations: list[Annotation],
              visible: bool, show_border: bool = False, orient: tuple | None = None) -> None:
        self._geometry = geometry
        self._annotations = annotations
        self._visible = visible
        self._show_border = show_border
        self._orient = orient
        self.update()

    # -------------------------------------------------------------- paint
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        if not (self._visible and self._geometry):
            return
        # Window-local (0,0) == this screen's global top-left, so shift global
        # logical coordinates into local space and let Qt handle this monitor's
        # devicePixelRatio.
        origin = self._screen.geometry().topLeft()
        painter = QtGui.QPainter(self)
        try:
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            painter.translate(-origin.x(), -origin.y())
            if self._show_border:
                self._draw_border(painter, self._geometry)
            if self._orient and self._orient[0]:
                self._draw_orientation(painter, self._geometry, self._orient)
            # Lower-ranked / opponent moves first so the best player move is on top.
            for ann in sorted(self._annotations, key=lambda a: (not a.opponent, -a.rank)):
                self._draw_move(painter, ann)
        except Exception:               # a paint must never crash the app
            import traceback
            traceback.print_exc()
        finally:
            painter.end()

    @staticmethod
    def _draw_border(painter: QtGui.QPainter, g: BoardGeometry) -> None:
        """The calibrated board outline + grid, so the user can see what was found."""
        side = g.square * 8
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 220, 255, 235), max(2.0, g.square * 0.05)))
        painter.drawRect(QtCore.QRectF(g.origin_x, g.origin_y, side, side))
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 220, 255, 90), 1.0))
        for k in range(1, 8):
            x, y = g.origin_x + k * g.square, g.origin_y + k * g.square
            painter.drawLine(QtCore.QPointF(x, g.origin_y), QtCore.QPointF(x, g.origin_y + side))
            painter.drawLine(QtCore.QPointF(g.origin_x, y), QtCore.QPointF(g.origin_x + side, y))

    @staticmethod
    def _draw_orientation(painter: QtGui.QPainter, g: BoardGeometry, orient: tuple) -> None:
        """Side-of-board indicator of which way the board faces: a 'W' disc at White's
        home end, a 'B' disc at Black's, and an arrow pointing the way White advances.
        Cyan when the CV agrees with this orientation, amber when it thinks the board
        is flipped (a hint to flip), dim grey before the first confident read."""
        _, agree, conf = orient
        side = g.square * 8.0
        r = max(7.0, g.square * 0.24)
        cx = g.origin_x - g.square * 0.62              # left margin, or right if no room
        if cx - r < 2.0:
            cx = g.origin_x + side + g.square * 0.62
        top_y = g.origin_y + r * 1.2
        bot_y = g.origin_y + side - r * 1.2
        w_y, b_y = (bot_y, top_y) if g.white_bottom else (top_y, bot_y)   # White's home end

        if agree is None:
            col = QtGui.QColor(140, 150, 160, 170)
        elif agree:
            col = QtGui.QColor(0, 200, 230, int(150 + 105 * max(0.0, min(1.0, conf))))
        else:
            col = QtGui.QColor(255, 165, 40, int(150 + 105 * max(0.0, min(1.0, conf))))

        ddir = 1 if b_y > w_y else -1                  # +1: Black below (arrow down)
        sx, sy, ex, ey = cx, w_y + ddir * r, cx, b_y - ddir * r
        pen = QtGui.QPen(col, max(2.5, g.square * 0.06))
        pen.setCapStyle(QtCore.Qt.RoundCap)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawLine(QtCore.QPointF(sx, sy), QtCore.QPointF(ex, ey))
        ah = g.square * 0.20
        head = QtGui.QPainterPath(QtCore.QPointF(ex, ey))
        head.lineTo(ex - ah * 0.6, ey - ddir * ah)
        head.lineTo(ex + ah * 0.6, ey - ddir * ah)
        head.closeSubpath()
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(col)
        painter.drawPath(head)

        font = painter.font()
        font.setPixelSize(max(9, int(r * 1.05)))
        font.setBold(True)
        for cy, fill, txt, letter in ((w_y, (245, 245, 245), (20, 20, 20), "W"),
                                      (b_y, (25, 25, 25), (240, 240, 240), "B")):
            painter.setBrush(QtGui.QColor(*fill))
            painter.setPen(QtGui.QPen(col, max(2.0, g.square * 0.04)))
            painter.drawEllipse(QtCore.QPointF(cx, cy), r, r)
            painter.setFont(font)
            painter.setPen(QtGui.QPen(QtGui.QColor(*txt)))
            painter.drawText(QtCore.QRectF(cx - r, cy - r, 2 * r, 2 * r),
                             QtCore.Qt.AlignCenter, letter)

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
        self._show_border = False
        self._orient: tuple | None = None
        self._windows = [_OverlayWindow(s) for s in app.screens()]

    def _refresh(self) -> None:
        for w in self._windows:
            w.apply(self._geometry, self._annotations, self._visible, self._show_border,
                    self._orient)

    def set_show_border(self, show: bool) -> None:
        """Toggle the calibrated board outline (verification aid for both modes)."""
        self._show_border = show
        self._refresh()

    def set_orientation(self, show: bool, agree: bool | None, confidence: float) -> None:
        """Update the board-direction indicator. ``agree`` is whether the CV believes
        the active orientation is correct (None = no confident read yet)."""
        self._orient = (show, agree, confidence)
        self._refresh()

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
