"""Puzzle SOLVING on real Lichess puzzles (gated on Stockfish).

Fixture (tests/fixtures/puzzle_solves.csv): held-out puzzles with solution moves.
A Lichess Moves field starts with the opponent's setup move; pushing it yields the
puzzle position (player to move), whose solution is the next move.

Two checks:
  * raw solve -- the engine's best move IS the puzzle solution.
  * puzzle-mode end to end -- _analyze_puzzle converges on the side to move (it has
    no move history to go on) AND surfaces the solution as the move to play.

Run: python tests/test_puzzle_solve.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chess
import chess.engine
from PySide6 import QtCore
from src.engine import find_stockfish

app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
PASS, FAIL = 0, 0
DEPTH = 18


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}   {detail}")


def _load(n):
    rows = []
    with open(ROOT / "tests" / "fixtures" / "puzzle_solves.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            mv = row["Moves"].split()
            if len(mv) >= 2:
                rows.append((row["FEN"], mv))
            if len(rows) >= n:
                break
    return rows


def _puzzle_position(fen, mv):
    """Position the solver actually sees: after the opponent's setup move."""
    b = chess.Board(fen)
    b.push_uci(mv[0])
    return b, mv[1]


def main():
    sf = find_stockfish()
    if not sf:
        print("  SKIP  stockfish not installed")
        print(f"\n{PASS} passed, {FAIL} failed")
        return

    # --- raw solve: engine best move == the puzzle solution ---
    eng = chess.engine.SimpleEngine.popen_uci(sf)
    eng.configure({"Threads": 2, "Hash": 128})
    solved = total = 0
    for fen, mv in _load(120):
        try:
            board, sol = _puzzle_position(fen, mv)
        except Exception:
            continue
        best = eng.analyse(board, chess.engine.Limit(depth=DEPTH))["pv"][0].uci()
        total += 1
        solved += (best == sol)
    eng.quit()
    rate = 100 * solved / max(1, total)
    print(f"  raw solve @ depth {DEPTH}: {rate:.2f}%  ({solved}/{total})")
    check(f"engine solves >= 97% of puzzles ({rate:.1f}%)", rate >= 97.0, f"{rate:.2f}%")

    # --- puzzle mode end to end: converge on the side + surface the solution ---
    from src.analysis import EngineController
    ctrl = EngineController(sf, 2, 64)
    if not ctrl._spawn_engine():
        print("  SKIP  puzzle-mode (engine spawn failed)")
        print(f"\n{PASS} passed, {FAIL} failed")
        return
    cap = {}
    ctrl.updated.connect(lambda s, d, b, o, t: cap.update(s=s, b=b))
    side_ok = solve_ok = total2 = 0
    for fen, mv in _load(80):
        try:
            board, sol = _puzzle_position(fen, mv)
        except Exception:
            continue
        cap.clear()
        ctrl._analyze_puzzle(board, 1, DEPTH, total2 + 1)   # multipv=1 (as the menu does)
        total2 += 1
        if cap.get("b") is not None and cap["b"].turn == board.turn:
            side_ok += 1
        if cap.get("s") and cap["s"][0].move.uci() == sol:
            solve_ok += 1
    ctrl._quit_engine()
    side_rate = 100 * side_ok / max(1, total2)
    solve_rate = 100 * solve_ok / max(1, total2)
    print(f"  puzzle-mode converged on the side to move: {side_rate:.1f}%  ({side_ok}/{total2})")
    print(f"  puzzle-mode surfaced the solution move:     {solve_rate:.1f}%  ({solve_ok}/{total2})")
    check(f"puzzle mode picks the side to move >= 80% ({side_rate:.1f}%)", side_rate >= 80.0)
    check(f"puzzle mode surfaces the solution >= 80% ({solve_rate:.1f}%)", solve_rate >= 80.0)

    print(f"\n{PASS} passed, {FAIL} failed")


main()
sys.exit(1 if FAIL else 0)
