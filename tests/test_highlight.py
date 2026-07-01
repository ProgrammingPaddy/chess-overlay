"""Move-highlight side detection (synthetic boards, no engine).

Renders a board with two squares tinted like a site's last-move highlight and checks
that detect_last_move recovers the pair + the side to move, and — importantly — that a
board with NO highlight ABSTAINS (returns None) so it can never mis-pin the side.

Run: python tests/test_highlight.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chess
import cv2
import numpy as np

from src.highlight import detect_last_move
from src.vision import cell_to_square

PASS = FAIL = 0
S = 48


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  PASS  {name}")
    else:
        FAIL += 1; print(f"  FAIL  {name}   {detail}")


def base_bgr(sq):
    light = (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 1
    return (np.array([181, 217, 240], np.float32) if light
            else np.array([99, 136, 181], np.float32))


def render(board, white_bottom=True, highlight=None, hl_bgr=(70, 235, 235)):
    """Render an 8x8 board (BGR) with optional last-move tint on `highlight` squares."""
    img = np.zeros((8 * S, 8 * S, 3), np.uint8)
    for r in range(8):
        for c in range(8):
            sq = cell_to_square(r, c, white_bottom)
            col = base_bgr(sq).copy()
            if highlight and sq in highlight:
                col = 0.55 * col + 0.45 * np.array(hl_bgr, np.float32)   # blend the tint
            img[r * S:(r + 1) * S, c * S:(c + 1) * S] = col.astype(np.uint8)
            pc = board.piece_at(sq)
            if pc is not None:                                          # a centered piece blob
                cy, cx = r * S + S // 2, c * S + S // 2
                cv2.circle(img, (cx, cy), S // 3,
                           (245, 245, 245) if pc.color else (25, 25, 25), -1)
                cv2.circle(img, (cx, cy), S // 3, (128, 128, 128), 2)
    return img


def after(moves):
    b = chess.Board()
    for uci in moves:
        b.push_uci(uci)
    return b


def main():
    # --- White just played e4: e2->e4 highlighted, Black to move ---
    b = after(["e2e4"])
    img = render(b, True, highlight={chess.E2, chess.E4})
    h = detect_last_move(img, True, b)
    check("detects a highlighted pair", h is not None)
    if h:
        check("pair is {e2,e4}", {h.from_square, h.to_square} == {chess.E2, chess.E4},
              f"{chess.square_name(h.from_square)}->{chess.square_name(h.to_square)}")
        check("from = e2 (empty), to = e4 (pawn)",
              h.from_square == chess.E2 and h.to_square == chess.E4)
        check("side to move = Black (White just moved)", h.side_to_move == chess.BLACK,
              str(h.side_to_move))

    # --- Black replied e5: e7->e5 highlighted, White to move ---
    b2 = after(["e2e4", "e7e5"])
    img2 = render(b2, True, highlight={chess.E7, chess.E5})
    h2 = detect_last_move(img2, True, b2)
    check("side to move = White (Black just moved)", h2 is not None and h2.side_to_move == chess.WHITE,
          str(h2.side_to_move if h2 else None))

    # --- a busy midgame, a knight capture highlighted: still finds the pair + side ---
    b3 = after(["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5c6", "d7c6"])
    # last move was d7c6 (Black), White to move
    img3 = render(b3, True, highlight={chess.D7, chess.C6})
    h3 = detect_last_move(img3, True, b3)
    check("midgame: side to move = White", h3 is not None and h3.side_to_move == chess.WHITE,
          str(h3.side_to_move if h3 else None))

    # --- same-colour-square move (Ra1-a3 style): both highlighted squares same class ---
    bd = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    bd.push_uci("a1a3")                                  # a1(dark)->a3(dark), Black to move
    img4 = render(bd, True, highlight={chess.A1, chess.A3})
    h4 = detect_last_move(img4, True, bd)
    check("same-colour move detected, side = Black",
          h4 is not None and h4.side_to_move == chess.BLACK, str(h4.side_to_move if h4 else None))

    # --- NO highlight: must ABSTAIN (never mis-pin) ---
    b5 = after(["e2e4", "e7e5"])
    img5 = render(b5, True, highlight=None)
    h5 = detect_last_move(img5, True, b5)
    check("no highlight -> abstains (None)", h5 is None, str(h5))

    # --- flipped orientation (Black on bottom) still works ---
    img6 = render(b, False, highlight={chess.E2, chess.E4})
    h6 = detect_last_move(img6, False, b)
    check("flipped board: side to move = Black",
          h6 is not None and h6.side_to_move == chess.BLACK, str(h6.side_to_move if h6 else None))

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


sys.exit(main())
