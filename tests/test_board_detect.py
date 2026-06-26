"""Auto board-alignment (second CALIBRATION mode) — tests on REAL pixels.

Embeds the real captured board (tests/whiteplayerstartingposition.png) into larger
canvases (varied margins, background, noise, fake UI chrome) to mimic a roughly
selected region, and checks that ``find_board`` snaps to the board AND that the
auto-detected region still recognises the position exactly. The piece recogniser
is unchanged; this only finds the board *rectangle*.
Run: python tests/test_board_detect.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
import cv2
import numpy as np
from src.board_detect import find_board
from src.vision import VisionModel

BOARD = cv2.imread(str(Path(__file__).resolve().parent / "whiteplayerstartingposition.png"))
BH, BW = BOARD.shape[:2]
PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}   {detail}")


def _canvas(ml, mt, mr, mb, bg=128, noise=0, ui=False):
    h, w = BH + mt + mb, BW + ml + mr
    c = np.full((h, w, 3), bg, np.uint8)
    if noise:
        c = (c.astype(int) + np.random.randint(-noise, noise + 1, (h, w, 3))).clip(0, 255).astype(np.uint8)
    if ui:
        cv2.rectangle(c, (2, 2), (w - 3, 18), (90, 90, 90), -1)
        cv2.putText(c, "lichess.org 1-0", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    c[mt:mt + BH, ml:ml + BW] = BOARD
    return c, (ml, mt, BW, BH)


CASES = {
    "tight": (0, 0, 0, 0, 128, 0, False),
    "even-margin": (20, 20, 20, 20, 128, 0, False),
    "uneven-margin": (40, 30, 15, 55, 150, 0, False),
    "noisy-bg": (25, 25, 25, 25, 120, 12, False),
    "ui-chrome": (18, 30, 18, 18, 135, 6, True),
    "dark-bg": (30, 30, 30, 30, 40, 0, False),
    "light-bg": (30, 30, 30, 30, 225, 0, False),
}


def test_finds_board_within_a_few_px():
    for name, args in CASES.items():
        c, (tx, ty, tw, th) = _canvas(*args)
        got = find_board(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY))
        if got is None:
            check(f"{name}: board found", False, "got None")
            continue
        x, y, w, h = got
        err = max(abs(x - tx), abs(y - ty), abs(x + w - tx - tw), abs(y + h - ty - th))
        check(f"{name}: edges within 6px (err={err})", err <= 6, f"region={got} true={(tx,ty,tw,th)}")


def test_detected_region_recognises_exactly():
    # The real point: whatever region it returns must recognise the position.
    for name, args in CASES.items():
        c, _ = _canvas(*args)
        got = find_board(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY))
        if got is None:
            check(f"{name}: end-to-end", False, "detect None")
            continue
        x, y, w, h = got
        crop = c[y:y + h, x:x + w]
        m = VisionModel()
        m.calibrate(crop)
        wrong = sum(1 for sq in chess.SQUARES if m.recognize(crop).piece_at(sq) != chess.Board().piece_at(sq))
        check(f"{name}: auto-region recognises start (wrong={wrong})", wrong == 0)


def test_rejects_non_board():
    # A flat image has no grid -> None (don't hand back a bogus region).
    check("uniform image -> None", find_board(np.full((400, 400), 128, np.uint8)) is None)
    rng = np.random.default_rng(0)
    out = find_board(rng.integers(0, 255, (400, 400), dtype=np.uint8))   # must not crash
    check("pure noise -> None or harmless", out is None or len(out) == 4)


if __name__ == "__main__":
    for fn in (test_finds_board_within_a_few_px, test_detected_region_recognises_exactly,
               test_rejects_non_board):
        print(fn.__name__)
        fn()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
