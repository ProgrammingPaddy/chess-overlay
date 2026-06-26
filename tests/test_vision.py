"""Vision tests — driven entirely by the user's REAL board pixels.

The only fixture is ``tests/whiteplayerstartingposition.png`` (an actual capture of
the user's wood-themed board, white on bottom). From it we extract the real square
crops and COMPOSE arbitrary positions with genuine pixels — real grain, real
glyphs, real anti-aliasing, and (alpha-composited) pieces on either square colour
plus simulated last-move highlights. This is strictly for *testing*: calibration in
the test, like in the app, learns only from a real start image.

What is locked in:
  * reads the real start exactly,
  * reads composed midgames / endgames / random placements exactly (incl. a piece
    on a square colour it was not calibrated on),
  * a last-move highlight tint never invents or hides a piece,
  * orientation is whatever the caller passes (never silently flipped),
  * empty board reads empty; uncalibrated raises; junk images don't crash.
Run: python tests/test_vision.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
import cv2
import numpy as np
from src.vision import VisionModel, certainty

REAL = cv2.imread(str(Path(__file__).resolve().parent / "whiteplayerstartingposition.png"))
PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}   {detail}")


# ---------------------------------------------------- real-pixel composer
_H, _W = REAL.shape[:2]
_S = min(_H, _W) // 8


def _is_light(sq):
    return (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 1


def _cell(r, c):
    return REAL[r * _S:(r + 1) * _S, c * _S:(c + 1) * _S].copy()


_START = chess.Board()
_EMPTY, _CROPS = {}, {}
for _r in range(8):
    for _c in range(8):
        _sq = chess.square(_c, 7 - _r)
        _pc = _START.piece_at(_sq)
        if _pc is None:
            _EMPTY.setdefault(_is_light(_sq), _cell(_r, _c))
        else:
            _CROPS.setdefault(_pc.symbol(), {})[_is_light(_sq)] = (_cell(_r, _c), _is_light(_sq))

_TINT = {True: np.array([107, 210, 206], np.float32), False: np.array([59, 162, 170], np.float32)}


def _alpha(crop, src_light):
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    e = cv2.cvtColor(_EMPTY[src_light], cv2.COLOR_BGR2GRAY).astype(np.float32)
    return np.clip(np.abs(g - e) / 28.0, 0, 1)[..., None]


def compose(piece_map, white_bottom=True, highlights=()):
    """Build a board image for ``piece_map`` from the real square crops."""
    pm = piece_map.piece_map() if isinstance(piece_map, chess.Board) else dict(piece_map)
    out = np.zeros((_S * 8, _S * 8, 3), np.uint8)
    for r in range(8):
        for c in range(8):
            sq = chess.square(c, 7 - r) if white_bottom else chess.square(7 - c, r)
            light = _is_light(sq)
            bg = _EMPTY[light].astype(np.float32).copy()
            if sq in highlights:
                bg = bg * 0.59 + _TINT[light] * 0.41          # lichess last-move tint
            if sq in pm:
                crop, src = _CROPS[pm[sq].symbol()].get(light) or next(iter(_CROPS[pm[sq].symbol()].values()))
                a = _alpha(crop, src)
                bg = bg * (1 - a) + crop.astype(np.float32) * a
            out[r * _S:(r + 1) * _S, c * _S:(c + 1) * _S] = np.clip(bg, 0, 255).astype(np.uint8)
    return out


def wrong(got, piece_map):
    pm = piece_map.piece_map() if isinstance(piece_map, chess.Board) else dict(piece_map)
    return [chess.square_name(s) for s in chess.SQUARES
            if (got.piece_at(s).symbol() if got.piece_at(s) else ".")
            != (pm[s].symbol() if s in pm else ".")]


def _game(*sans):
    b = chess.Board()
    for m in sans:
        b.push_san(m)
    return b


def _calibrated():
    m = VisionModel()
    m.calibrate(REAL, white_bottom=True)
    return m


# ----------------------------------------------------------------- tests
def test_reads_the_real_start_exactly():
    m = VisionModel()
    warn = m.calibrate(REAL, white_bottom=True)
    got = m.recognize(REAL)
    check("real start: clean calibration (no warning)", warn == "", warn)
    check("real start: exact placement", not wrong(got, chess.Board()), f"wrong={wrong(got, chess.Board())}")


def test_composed_positions_exact():
    m = _calibrated()
    positions = {
        "ruy": _game("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"),
        "sicilian": _game("e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6"),
        "qgd": _game("d4", "d5", "c4", "e6", "Nc3", "Nf6", "Bg5", "Be7", "e3", "O-O"),
        "endgame": chess.Board("8/5pk1/6p1/8/3K4/4P3/5PP1/8 w - - 0 1"),
        # a piece on a square colour it was NOT calibrated on (black Q started on light d8):
        "queen-on-dark": chess.Board("rnb1kbnr/pppp1ppp/8/4p3/7q/5P2/PPPPP1PP/RNBQKBNR w - - 0 1"),
    }
    for name, ref in positions.items():
        got, _ = m.analyze(compose(ref))
        check(f"composed/{name}: exact", not wrong(got, ref), f"wrong={wrong(got, ref)[:8]}")


def test_last_move_highlight():
    m = _calibrated()
    pos = _game("e4", "e5", "Nf3", "Nc6")
    got = m.recognize(compose(pos, highlights={chess.F3, chess.C6, chess.E4, chess.E5}))
    check("last-move tint: no phantom, no swap", not wrong(got, pos), f"wrong={wrong(got, pos)}")


def test_both_orientations():
    for wb in (True, False):
        m = VisionModel()
        m.calibrate(compose(chess.Board(), white_bottom=wb), white_bottom=wb)
        pos = _game("e4", "c5", "Nf3", "d6", "d4", "cxd4")
        got, _ = m.analyze(compose(pos, white_bottom=wb), wb)
        check(f"orientation {'white' if wb else 'black'}-bottom: exact", not wrong(got, pos),
              f"wrong={wrong(got, pos)[:8]}")


def test_orientation_autodetected_from_pixels():
    # Orientation is read from the board (white army is brighter), so a wrong/stale
    # 'White on bottom' setting cannot 180-rotate it.
    m = VisionModel()
    m.calibrate(REAL, white_bottom=False)               # wrong hint for a white-bottom board
    check("white-bottom board auto-detected white_bottom=True", m.white_bottom is True,
          f"got {m.white_bottom}")
    rot = cv2.rotate(REAL, cv2.ROTATE_180)              # now black is on the bottom
    m2 = VisionModel()
    m2.calibrate(rot, white_bottom=True)                # wrong hint again
    check("black-bottom board auto-detected white_bottom=False", m2.white_bottom is False,
          f"got {m2.white_bottom}")


def test_stale_setting_does_not_rotate():
    # THE reported bug: white-on-bottom board with a stale white_bottom=False setting
    # produced an all-colours-inverted, kings<->queens-swapped board. The start is
    # symmetric and hides it, so check a position AFTER moves.
    m = VisionModel()
    m.calibrate(REAL, white_bottom=False)              # the stale wrong setting
    pos = _game("e4", "e5", "Nf3", "Nc6", "Bb5")       # asymmetric -> exposes any rotation
    got = m.recognize(compose(pos))                    # compose renders white-on-bottom (physical)
    check("stale white_bottom=False no longer rotates the board", not wrong(got, pos),
          f"wrong={wrong(got, pos)[:8]}")


def test_exhaustive_random_real_pixels():
    m = _calibrated()
    pieces = [chess.Piece.from_symbol(s) for s in "KQRBNPkqrbnp"]
    rng = random.Random(0)
    total = bad = squares = 0
    for _ in range(200):
        pm = {sq: rng.choice(pieces) for sq in chess.SQUARES if rng.random() < 0.45}
        wb = bool(rng.getrandbits(1))
        hl = set(rng.sample(list(chess.SQUARES), rng.randint(0, 4)))
        got, _ = m.analyze(compose(pm, white_bottom=wb, highlights=hl), wb)
        w = len(wrong(got, pm))
        total += w
        squares += 64
        bad += (1 if w else 0)
    acc = 100 * (1 - total / squares)
    check(f"random real-pixel placements: {acc:.4f}% per-square (>=99.9)", acc >= 99.9, f"{total}/{squares}")
    check(f"random real-pixel: {200 - bad}/200 boards perfect (>=99%)", (200 - bad) / 200 >= 0.99, f"{bad} bad")


def test_robust_to_misalignment():
    # The capture grid is rarely pixel-perfect (region a few px off, a piece not
    # dead-centre). Recognition must survive a few px of shift without flipping
    # pieces/colours — this is what caused the live errors and "colour swap".
    m = _calibrated()
    pos = _game("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5")
    base = compose(pos)
    worst = 0
    for dx in (-4, -2, 0, 2, 4):
        for dy in (-4, -2, 0, 2, 4):
            mat = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(base, mat, (base.shape[1], base.shape[0]),
                                     borderMode=cv2.BORDER_REPLICATE)
            worst = max(worst, len(wrong(m.recognize(shifted), pos)))
    check(f"recognition robust to +/-4px misalignment (worst={worst} errors)", worst == 0,
          f"worst {worst}")


def test_empty_board_reads_empty():
    m = _calibrated()
    got = m.recognize(compose({}))
    check("empty board reads fully empty", not got.piece_map(), f"phantoms {got.board_fen()}")


def test_uncalibrated_raises():
    try:
        VisionModel().analyze(REAL)
        check("uncalibrated analyze raises", False)
    except RuntimeError:
        check("uncalibrated analyze raises RuntimeError", True)


def test_degenerate_images_no_crash():
    m = _calibrated()
    for nm, img in {"black": np.zeros((400, 400, 3), np.uint8),
                    "noise": (np.random.rand(400, 400, 3) * 255).astype(np.uint8),
                    "tiny": np.zeros((30, 30, 3), np.uint8)}.items():
        try:
            _, dbg = m.analyze(img)
            check(f"degenerate {nm}: 64 cells, no crash", len(dbg) == 64)
        except Exception as exc:
            check(f"degenerate {nm}: no crash", False, repr(exc))


def test_certainty_signal():
    check("certainty([]) == 0", certainty([]) == 0.0)
    m = _calibrated()
    _, real = m.analyze(REAL)
    _, noise = m.analyze((np.random.rand(_S * 8, _S * 8, 3) * 255).astype(np.uint8))
    check("real board certainty high (>0.5)", certainty(real) > 0.5, f"{certainty(real):.2f}")
    check("real board more certain than noise", certainty(real) > certainty(noise))


if __name__ == "__main__":
    for fn in (test_reads_the_real_start_exactly, test_composed_positions_exact,
               test_last_move_highlight, test_both_orientations,
               test_orientation_autodetected_from_pixels, test_stale_setting_does_not_rotate,
               test_exhaustive_random_real_pixels, test_robust_to_misalignment,
               test_empty_board_reads_empty,
               test_uncalibrated_raises, test_degenerate_images_no_crash, test_certainty_signal):
        print(fn.__name__)
        fn()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
