"""Board-orientation detection accuracy on real Lichess puzzle positions.

The fixture (tests/fixtures/orientation_puzzles.csv) is 12k positions HELD OUT from
the model's training data, spanning every rating and ~50% endgames (the hard case).
A Lichess FEN is white-normal, so the detector must return white_bottom=True for it
and white_bottom=False for its 180-degree rotation.

Two levels:
  * FEN-level: the chess-reasoning classifier on the exact placement.
  * end-to-end: render each position from the user's REAL board pixels, recognize it
    with the CV, then detect -- proving recognition + detection work together.

Run: python tests/test_orientation.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import chess
from src.orientation import detect_orientation, orientation_logit

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}   {detail}")


def rot180(b: chess.Board) -> chess.Board:
    return b.transform(chess.flip_vertical).transform(chess.flip_horizontal)


def _load_fixture():
    path = ROOT / "tests" / "fixtures" / "orientation_puzzles.csv"
    rows = []
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append((row["FEN"], row.get("Themes", "")))
    return rows


# --- sanity: unambiguous positions ------------------------------------------
def test_obvious_positions():
    start = chess.Board()
    wb, p = detect_orientation(start)
    check("start position: White on the bottom", wb is True, str(p))
    check("start position: high confidence (>0.99)", p > 0.99, f"{p:.4f}")
    wb2, p2 = detect_orientation(rot180(start))
    check("start rotated 180: White on top", wb2 is False, str(p2))
    # a normal middlegame (Italian)
    mid = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    check("italian middlegame: White on the bottom", detect_orientation(mid)[0] is True)
    check("italian rotated: White on top", detect_orientation(rot180(mid))[0] is False)
    # logit is antisymmetric under rotation (sign flips)
    check("logit flips sign under 180 rotation", orientation_logit(mid) * orientation_logit(rot180(mid)) < 0)


# --- the headline: accuracy on held-out Lichess puzzles ----------------------
def test_fixture_accuracy():
    rows = _load_fixture()
    correct = total = 0
    endgame_c = endgame_t = 0
    for fen, themes in rows:
        try:
            b = chess.Board(fen)
        except Exception:
            continue
        is_end = "endgame" in themes
        for board, expect in ((b, True), (rot180(b), False)):
            total += 1
            ok = detect_orientation(board)[0] is expect
            correct += ok
            if is_end:
                endgame_t += 1
                endgame_c += ok
    acc = 100 * correct / total
    end_acc = 100 * endgame_c / max(1, endgame_t)
    print(f"  fixture: {len(rows)} positions, {total} oriented cases")
    print(f"  accuracy: {acc:.4f}%   (endgames: {end_acc:.4f}%)")
    check(f"orientation accuracy >= 99% on held-out puzzles ({acc:.3f}%)", acc >= 99.0, f"{acc:.4f}%")


# --- end-to-end: real pixels -> recognize -> detect --------------------------
def test_end_to_end_pixels():
    try:
        import test_vision as tv  # reuses the user's real board fixture + composer
    except Exception as exc:
        print(f"  SKIP  end-to-end (vision composer unavailable: {exc})")
        return
    model = tv._calibrated()
    rows = _load_fixture()[:400]
    correct = total = 0
    for fen, _ in rows:
        try:
            b = chess.Board(fen)
        except Exception:
            continue
        for physical_wb in (True, False):
            img = tv.compose(b, white_bottom=physical_wb)
            # the app recognises with a fixed assumption (white_bottom=True); the
            # detector must recover the TRUE physical orientation from the pixels.
            rec = model.analyze(img, white_bottom=True)[0]
            total += 1
            correct += detect_orientation(rec)[0] is physical_wb
    acc = 100 * correct / max(1, total)
    print(f"  end-to-end recognise+detect on {total} rendered boards: {acc:.4f}%")
    check(f"end-to-end orientation accuracy >= 98.5% ({acc:.3f}%)", acc >= 98.5, f"{acc:.4f}%")


if __name__ == "__main__":
    for fn in (test_obvious_positions, test_fixture_accuracy, test_end_to_end_pixels):
        print(fn.__name__)
        fn()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
