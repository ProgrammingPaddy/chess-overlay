"""Predictive engine mode — opponent's top moves + the best reply to EACH of them.

  1. Routing / collapse (no engine): the per-opponent-move replies are used ONLY on
     the opponent's turn; on the player's turn predictive falls through to the
     normal multipv flow — i.e. it collapses to live behaviour. Live/Fixed never
     use the per-move replies.
  2. Selection (needs Stockfish, else SKIP): the real _analyze_predictive emits the
     opponent's N candidates plus one legal best reply paired to each.

Run: python tests/test_predictive.py
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
import chess.engine
from PySide6 import QtCore

from src.analysis import EngineController
from src.engine import find_stockfish

app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}   {detail}")


# --- 1. routing / collapse, with a stub engine (no Stockfish needed) ----------
class _EmptyAnalysis:
    """Stands in for engine.analysis(...): an empty, immediately-exhausted stream."""
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter(())

    def stop(self):
        pass


def fake_controller():
    c = EngineController("dummy", 1, 16)
    c._engine = types.SimpleNamespace(
        analysis=lambda *a, **k: _EmptyAnalysis(),
        analyse=lambda *a, **k: [],          # _top_moves -> no opponent moves
        quit=lambda: None,
    )
    return c


c = fake_controller()
called = {"pred": False}
c._analyze_predictive = lambda *a, **k: called.__setitem__("pred", True)

# opponent to move: player is Black, White to move at the start
c._analyze(chess.Board(), 3, "predictive", 10, player_color=chess.BLACK, token=1)
check("predictive uses per-move replies on the opponent's turn", called["pred"] is True)

# player to move: after 1.e4 it is Black's (the player's) turn -> must collapse
called["pred"] = False
b = chess.Board()
b.push_san("e4")
c._analyze(b, 3, "predictive", 10, player_color=chess.BLACK, token=2)
check("predictive collapses to live on the player's turn", called["pred"] is False)

# live mode never uses the per-move replies, even on the opponent's turn
called["pred"] = False
c._analyze(chess.Board(), 3, "live", 10, player_color=chess.BLACK, token=3)
check("live mode never uses per-move replies", called["pred"] is False)

# fixed mode likewise
called["pred"] = False
c._analyze(chess.Board(), 3, "fixed", 10, player_color=chess.BLACK, token=4)
check("fixed mode never uses per-move replies", called["pred"] is False)


# --- 2. selection, with the real engine (skipped if Stockfish is absent) ------
path = find_stockfish()
if not path:
    print("  SKIP  no Stockfish found — selection test skipped")
else:
    c2 = EngineController(path, 1, 64)
    c2._engine = chess.engine.SimpleEngine.popen_uci(path)
    c2._engine.configure({"Threads": 1, "Hash": 64})
    captured = {}

    def cap(responses, depth, board, opp, token):
        captured.update(responses=responses, depth=depth, board=board, opp=opp, token=token)

    c2.updated.connect(cap)
    start = chess.Board()                      # White (opponent) to move; player is Black
    c2._analyze_predictive(start, 3, chess.BLACK, 77)
    c2._engine.quit()

    opp = captured.get("opp", [])
    resp = captured.get("responses", [])
    check("token preserved", captured.get("token") == 77, str(captured.get("token")))
    check("3 opponent candidates", len(opp) == 3, f"got {len(opp)}")
    check("3 paired replies (one per opponent move)", len(resp) == 3, f"got {len(resp)}")
    check("reply ranks pair to the opponent moves (1..3)",
          sorted(r.rank for r in resp) == [1, 2, 3], str([r.rank for r in resp]))
    check("opponent moves are legal at the start",
          all(m.move in start.legal_moves for m in opp))

    legal_after_its_move = True
    for r in resp:
        after = start.copy()
        after.push(opp[r.rank - 1].move)       # the opponent move this reply answers
        if r.move not in after.legal_moves:
            legal_after_its_move = False
    check("each reply is legal after its own opponent move", legal_after_its_move)
    check("replies are concrete moves", all(r.move for r in resp))

    tb = captured.get("board")
    check("emitted board is after the opponent's best move, player to move",
          tb is not None and tb.turn == chess.BLACK and tb.board_fen() != start.board_fen())

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
