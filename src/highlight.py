"""Optional, gated last-move HIGHLIGHT reader — the reliable 'whose move' signal.

Most chess sites tint the two squares of the last move. Reading that pair pins the
side to move with near-certainty (the piece that just moved belongs to the side that
is NOT to move) — exactly what a single static frame otherwise cannot tell, and the
cause of the residual ~10% side-pick misses.

Design (deliberately non-invasive):
  * Works on the raw COLOUR board image, entirely SEPARATE from piece recognition
    (which normalises the tint away on purpose), so enabling it never changes the CV.
  * THEME-INDEPENDENT / self-calibrating: it learns the board's own light/dark
    background colours per frame and finds the two squares whose background deviates
    most, in a CONSISTENT direction (the same highlight hue) — no hard-coded colour.
  * CONSERVATIVE: if it can't see a clear highlighted pair it returns None and the
    caller falls back to the engine's side pick. It only ever OVERRIDES when confident,
    and parity from a played move always wins — so a wrong read can never persist.
  * CHEAP: 64 border-median colours + a sort. ~1 ms, and cached with the recognition
    on a static board, so it does not slow processing.
"""
from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np

from src.vision import _is_light, cell_to_square, split_cells

# Tuning in BGR 0-255 space. Conservative — abstain rather than mis-pin the side.
_MIN_TINT = 6.0        # a highlighted square's background must deviate at least this much
_GAP_RATIO = 1.5       # the 2nd-strongest deviation must exceed the 3rd by this factor
_MIN_COSINE = 0.5      # the two highlighted tints must point the same way (same colour)
_BORDER = 0.20         # fraction of each cell edge used as the piece-free background ring


@dataclass
class Highlight:
    from_square: chess.Square
    to_square: chess.Square
    side_to_move: bool | None      # None if the moved piece couldn't be identified
    confidence: float


def _bg_color(cell: np.ndarray) -> np.ndarray:
    """A cell's background colour from its border ring (the centre holds the piece).
    Median makes it robust to piece edges, shadows and a corner coordinate glyph."""
    if cell.ndim != 3 or cell.size == 0:
        return np.zeros(3, np.float32)
    h, w = cell.shape[:2]
    by, bx = max(1, int(h * _BORDER)), max(1, int(w * _BORDER))
    ring = np.concatenate([
        cell[:by].reshape(-1, 3), cell[h - by:].reshape(-1, 3),
        cell[by:h - by, :bx].reshape(-1, 3), cell[by:h - by, w - bx:].reshape(-1, 3),
    ], axis=0).astype(np.float32)
    return np.median(ring, axis=0)


def detect_last_move(board_img, white_bottom: bool,
                     recognized: chess.Board | None = None) -> "Highlight | None":
    """Find the last-move highlighted square pair (and the resulting side to move), or
    None if a clear highlight isn't present. ``recognized`` (the CV placement) is used
    only to tell the 'from' (empty) square from the 'to' (the moved piece)."""
    if board_img is None or getattr(board_img, "ndim", 0) != 3:
        return None
    cells = split_cells(board_img)
    if len(cells) < 64:
        return None
    bg = {rc: _bg_color(cell) for rc, cell in cells.items()}
    # Per-colour-class median background — robust to the ~2 highlighted squares.
    by_class = {True: [], False: []}
    for rc in cells:
        by_class[_is_light(cell_to_square(*rc, white_bottom))].append(rc)
    median = {light: (np.median(np.array([bg[rc] for rc in rcs], np.float32), axis=0)
                      if rcs else np.zeros(3, np.float32))
              for light, rcs in by_class.items()}
    # Residual of each cell from its class median; the highlighted pair deviates most.
    resid, mag = {}, {}
    for rc in cells:
        r = bg[rc] - median[_is_light(cell_to_square(*rc, white_bottom))]
        resid[rc], mag[rc] = r, float(np.linalg.norm(r))
    order = sorted(cells, key=lambda rc: mag[rc], reverse=True)
    a, b, c = order[0], order[1], order[2]
    # Confidence gates: real tint, a clear top-two, and a shared hue (same highlight).
    if mag[b] < _MIN_TINT or mag[b] < _GAP_RATIO * mag[c]:
        return None
    denom = (np.linalg.norm(resid[a]) * np.linalg.norm(resid[b])) or 1.0
    cosine = float(np.dot(resid[a], resid[b]) / denom)
    if cosine < _MIN_COSINE:
        return None
    sq_a, sq_b = cell_to_square(*a, white_bottom), cell_to_square(*b, white_bottom)
    from_sq, to_sq, side = sq_a, sq_b, None
    if recognized is not None:
        pa, pb = recognized.piece_at(sq_a), recognized.piece_at(sq_b)
        if (pa is None) != (pb is None):        # exactly one occupied = a clean from -> to
            (from_sq, to_sq, mover) = (sq_a, sq_b, pb) if pa is None else (sq_b, sq_a, pa)
            side = not mover.color              # side to move = opposite of who just moved
    conf = min(1.0, mag[b] / (mag[c] + 1e-6) - 1.0) * max(0.0, cosine)
    return Highlight(from_sq, to_sq, side, round(conf, 3))


def _square_to_cell(sq: chess.Square, white_bottom: bool) -> tuple[int, int]:
    """Inverse of ``cell_to_square``: board square -> (row, col) grid cell."""
    f, r = chess.square_file(sq), chess.square_rank(sq)
    return (7 - r, f) if white_bottom else (r, 7 - f)


def dump_highlight(out_dir, board_img, white_bottom: bool, hl: "Highlight | None") -> None:
    """Write debug/highlight.png visualising the last-move read (or the abstention) so a
    theme where it 'doesn't work' can be diagnosed: green = detected 'from', gold = 'to',
    with the side + confidence. Best-effort; never raises into the caller."""
    try:
        import cv2
        from pathlib import Path
        out = Path(out_dir)
        out.mkdir(exist_ok=True)
        vis = board_img.copy()
        h, w = vis.shape[:2]
        sy, sx = h / 8.0, w / 8.0
        if hl is not None:
            for sq, color, tag in ((hl.from_square, (60, 220, 60), "from"),
                                   (hl.to_square, (40, 200, 255), "to")):
                r, c = _square_to_cell(sq, white_bottom)
                x, y = int(c * sx), int(r * sy)
                cv2.rectangle(vis, (x + 2, y + 2), (int(x + sx) - 2, int(y + sy) - 2), color, 3)
                cv2.putText(vis, tag, (x + 4, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            side = "?" if hl.side_to_move is None else ("White" if hl.side_to_move else "Black")
            label = f"side={side} conf={hl.confidence:.2f}"
        else:
            label = "no clear highlight (abstained)"
        cv2.rectangle(vis, (0, 0), (w, 22), (0, 0, 0), -1)
        cv2.putText(vis, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(str(out / "highlight.png"), vis)
    except Exception:
        pass
