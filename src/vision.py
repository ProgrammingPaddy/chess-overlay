"""Board -> placement for a DIGITAL board (lichess / chess.com).

CV layer only: image in, a ``chess.Board`` placement out, per frame, for ANY
position. Turn/legality/temporal smoothing live downstream.

Two ideas make it reliable on a real (textured, slightly mis-cropped) board:

1. Empty-reference subtraction. At calibration on the start position we learn the
   actual empty LIGHT and empty DARK square (grain and all). A square is matched as
   ``empty`` or as one of the 12 pieces composited onto that empty reference, so
   the board texture is modelled rather than guessed, and a piece reads correctly
   even on a square colour it was not calibrated on.

2. Offset-tolerant matching. The captured grid is rarely pixel-perfect (region a
   few px off, non-square capture, a piece not perfectly centred). Each square is
   compared to the templates by SLIDING them over a slightly enlarged patch
   (cv2.matchTemplate) and taking the best alignment — so a few pixels of
   misalignment no longer flip the colour or the piece. Match scores are mean-
   normalized, which also absorbs a last-move highlight's uniform tint.

Orientation is taken from the caller ('White on bottom') and never auto-flipped.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import chess
import numpy as np

N = 40                 # template size
PAD = 6                # search half-window (template px); tolerates ~±10 source px
M = N + 2 * PAD        # enlarged patch the template slides within
MARGIN = 0.12          # central fraction for the occupancy energy check


def _sizes(img):
    h, w = img.shape[:2]
    return w / 8.0, h / 8.0           # (sx, sy) — separate axes (non-square safe)


def cell_to_square(r: int, c: int, white_bottom: bool) -> chess.Square:
    return chess.square(c, 7 - r) if white_bottom else chess.square(7 - c, r)


def _is_light(sq: chess.Square) -> bool:
    return (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 1


def split_cells(board_img: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    sx, sy = _sizes(board_img)
    return {(r, c): board_img[int(round(r * sy)):int(round((r + 1) * sy)),
                              int(round(c * sx)):int(round((c + 1) * sx))]
            for r in range(8) for c in range(8)}


def _gray(cell: np.ndarray, size: int) -> np.ndarray:
    import cv2
    g = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if cell.ndim == 3 else cell
    return cv2.resize(g, (size, size)).astype(np.float32)


def _center(a: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    m = int(n * MARGIN)
    return a[m:n - m, m:n - m]


def _gap_threshold(vals, floor):
    if not vals:
        return floor
    vals = sorted(vals)
    if len(vals) == 1:
        return max(floor, vals[0] * 0.5)
    gap, i = max((vals[i + 1] - vals[i], i) for i in range(len(vals) - 1))
    return max(floor, (vals[i] + vals[i + 1]) / 2)


class VisionModel:
    def __init__(self):
        self.empty: dict[bool, np.ndarray] = {}
        self.sprites: dict[str, np.ndarray] = {}
        self.occ_thresh = 6.0
        self.white_bottom = True
        self.calibrated = False

    def calibrate(self, board_img: np.ndarray, white_bottom: bool = True) -> str:
        start = chess.Board()
        cells = {rc: _gray(cell, N) for rc, cell in split_cells(board_img).items()}

        # Orientation is READ FROM THE PIXELS, not the setting: the white army is
        # visibly brighter, so whichever end (top rows 0-1 vs bottom rows 6-7) is
        # brighter is White's. This is what stops a stale 'White on bottom' value
        # from 180-degree-rotating the board — which would invert every colour and
        # swap kings<->queens. The passed value is only a fallback when the two
        # ends are indistinguishable (e.g. the board isn't a start position).
        top = float(np.mean([_center(cells[(r, c)]).mean() for r in (0, 1) for c in range(8)]))
        bot = float(np.mean([_center(cells[(r, c)]).mean() for r in (6, 7) for c in range(8)]))
        self.white_bottom = bool(white_bottom) if abs(bot - top) < 2.0 else bool(bot > top)

        empties = {True: [], False: []}
        for rc, g in cells.items():
            sq = cell_to_square(*rc, self.white_bottom)
            if start.piece_at(sq) is None:
                empties[_is_light(sq)].append(g)
        self.empty = {light: (np.mean(gs, axis=0) if gs else np.full((N, N), 128.0, np.float32))
                      for light, gs in empties.items()}

        sprite_acc = defaultdict(list)
        empty_sig, piece_sig = [], []
        for rc, g in cells.items():
            sq = cell_to_square(*rc, self.white_bottom)
            resid = g - self.empty[_is_light(sq)]
            sig = float(_center(resid).std())
            pc = start.piece_at(sq)
            if pc is None:
                empty_sig.append(sig)
            else:
                sprite_acc[pc.symbol()].append(resid)
                piece_sig.append(sig)
        self.sprites = {s: np.mean(rs, axis=0) for s, rs in sprite_acc.items()}
        self.occ_thresh = _gap_threshold(empty_sig + piece_sig, floor=3.0)
        self.calibrated = True
        return f"{len(piece_sig)} pieces seen (a fresh start has 32)" if len(piece_sig) != 32 else ""

    def analyze(self, board_img: np.ndarray,
                white_bottom: bool | None = None) -> tuple[chess.Board, list[dict]]:
        if not self.calibrated:
            raise RuntimeError("Vision not calibrated.")
        import cv2
        wb = self.white_bottom if white_bottom is None else white_bottom
        sx, sy = _sizes(board_img)
        gray = (cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY) if board_img.ndim == 3 else board_img)
        gray = gray.astype(np.float32)
        pad = max(1, int(round(min(sx, sy) * PAD / N)))
        padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

        board = chess.Board.empty()
        debug: list[dict] = []
        for r in range(8):
            for c in range(8):
                sq = cell_to_square(r, c, wb)
                light = _is_light(sq)
                # cheap occupancy gate: residual energy at the nominal cell
                cell = cv2.resize(gray[int(round(r * sy)):int(round((r + 1) * sy)),
                                       int(round(c * sx)):int(round((c + 1) * sx))], (N, N))
                sig = float(_center(cell - self.empty[light]).std())
                sym, best = None, 0.0
                if sig > self.occ_thresh and self.sprites:
                    # offset-tolerant piece id: slide each glyph delta over an
                    # enlarged patch and take the best-aligned correlation.
                    ya, yb = int(round(r * sy)), int(round((r + 1) * sy)) + 2 * pad
                    xa, xb = int(round(c * sx)), int(round((c + 1) * sx)) + 2 * pad
                    patch = cv2.resize(padded[ya:yb, xa:xb], (M, M))
                    scores = {s: float(cv2.matchTemplate(patch, sp, cv2.TM_CCOEFF_NORMED).max())
                              for s, sp in self.sprites.items()}
                    sym = max(scores, key=scores.get)
                    best = scores[sym]
                    board.set_piece_at(sq, chess.Piece.from_symbol(sym))
                occupied = sym is not None
                conf = (max(0.0, min(1.0, best)) if occupied
                        else min(1.0, abs(sig - self.occ_thresh) / (self.occ_thresh + 1e-6)))
                debug.append({"square": chess.square_name(sq), "rc": [r, c], "occupied": occupied,
                              "sig": round(sig, 2), "conf": round(conf, 3),
                              "piece": sym or "", "score": round(best, 3)})
        return board, debug

    def recognize(self, board_img: np.ndarray,
                  white_bottom: bool | None = None) -> chess.Board:
        return self.analyze(board_img, white_bottom)[0]


def certainty(debug: list[dict]) -> float:
    if not debug:
        return 0.0
    return sum(d.get("conf", 0.0) for d in debug) / len(debug)


def weakest_squares(debug: list[dict], n: int = 3) -> list[str]:
    return [d["square"] for d in sorted(debug, key=lambda d: d.get("conf", 0.0))[:n]]


# --------------------------------------------------------------------- debug
def _norm_u8(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)


def dump_recognition(out_dir: Path, board_img: np.ndarray, debug: list[dict]) -> None:
    import cv2
    out_dir.mkdir(exist_ok=True)
    vis = board_img.copy()
    h, w = vis.shape[:2]
    sx, sy = w / 8.0, h / 8.0
    for d in debug:
        r, c = d["rc"]
        x, y = int(c * sx), int(r * sy)
        cv2.rectangle(vis, (x, y), (int(x + sx), int(y + sy)), (0, 200, 0), 1)
        label = d["piece"] or ("?" if d["occupied"] else ".")
        color = (0, 0, 255) if d["occupied"] else (170, 170, 170)
        cv2.putText(vis, label, (x + 4, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imwrite(str(out_dir / "recognition_overlay.png"), vis)
    (out_dir / "recognition.json").write_text(json.dumps(debug, indent=1), encoding="utf-8")


def dump_calibration(out_dir: Path, board_img: np.ndarray, model: VisionModel) -> None:
    import cv2
    out_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(out_dir / "vision_calibration.png"), board_img)
    tiles = [(sym, _norm_u8(t)) for sym, t in sorted(model.sprites.items())]
    if tiles:
        tsz = tiles[0][1].shape[0]
        montage = np.full((tsz + 18, len(tiles) * (tsz + 4), 3), 40, np.uint8)
        for i, (name, img) in enumerate(tiles):
            x = i * (tsz + 4)
            montage[18:18 + tsz, x:x + tsz] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.putText(montage, name, (x + 2, 13), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1)
        cv2.imwrite(str(out_dir / "vision_templates.png"), montage)
        (out_dir / "vision_meta.json").write_text(json.dumps(
            {"occ_thresh": round(model.occ_thresh, 2), "white_bottom": model.white_bottom,
             "sprites": sorted(model.sprites.keys())}, indent=1), encoding="utf-8")
