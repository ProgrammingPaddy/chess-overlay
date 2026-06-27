"""Predictive + opponent look-ahead refinement.

  1. Routing / collapse (stub engine): the per-opponent-move replies are used ONLY
     on the opponent's turn in predictive mode; the player's turn collapses to the
     normal flow, and live/fixed never use them.
  2. Deepen schedule (pure): fencepost-correct depth ladders for live refinement.
  3. Selection (needs Stockfish, else SKIP): the opponent's N candidates plus one
     legal best reply paired to each; the emit carries the post-opponent board.

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
        quit=lambda: None,
    )
    return c


c = fake_controller()
called = {"pred": False}
c._predictive_turn = lambda *a, **k: called.__setitem__("pred", True)

# opponent to move: player is Black, White to move at the start
c._analyze(chess.Board(), 3, "predictive", 10, player_color=chess.BLACK, token=1)
check("predictive uses per-move replies on the opponent's turn", called["pred"] is True)

# player to move: after 1.e4 it is Black's (the player's) turn -> must collapse
called["pred"] = False
b = chess.Board()
b.push_san("e4")
c._analyze(b, 3, "predictive", 10, player_color=chess.BLACK, token=2)
check("predictive collapses to live on the player's turn", called["pred"] is False)

# live / fixed never use the per-move replies, even on the opponent's turn
for m in ("live", "fixed"):
    called["pred"] = False
    c._analyze(chess.Board(), 3, m, 10, player_color=chess.BLACK, token=3)
    check(f"{m} mode never uses per-move replies", called["pred"] is False)


# --- 2. deepen schedule (pure) ------------------------------------------------
sched = c._deepen_schedule
check("schedule 12->22 step 2", sched(12, 22) == [12, 14, 16, 18, 20, 22], str(sched(12, 22)))
check("schedule lands on the cap from an odd start",
      sched(13, 22) == [13, 15, 17, 19, 21, 22], str(sched(13, 22)))
check("schedule start==cap is a single round", sched(22, 22) == [22], str(sched(22, 22)))
check("schedule cap clamped up to start (preview deeper than ceiling)",
      sched(30, 22) == [30], str(sched(30, 22)))
for s, cap in ((8, 22), (12, 12), (1, 22), (15, 30)):
    ladder = sched(s, cap)
    check(f"schedule {s}->{cap} is sane",
          ladder[0] == s and ladder[-1] == max(s, cap)
          and all(ladder[i] < ladder[i + 1] for i in range(len(ladder) - 1)),
          str(ladder))


# --- 3. selection, with the real engine (skipped if Stockfish is absent) ------
path = find_stockfish()
if not path:
    print("  SKIP  no Stockfish found — selection test skipped")
else:
    c2 = EngineController(path, 1, 64)
    c2._engine = chess.engine.SimpleEngine.popen_uci(path)
    c2._engine.configure({"Threads": 1, "Hash": 64})
    start = chess.Board()                      # White (opponent) to move; player is Black

    opp = c2._top_moves(start, 3, 12)
    resp = c2._predictive_replies(start, opp, 12)
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

    # the emit path (one round, forced via a single-depth schedule)
    captured = {}

    def cap_emit(responses, depth, board, opp_s, token):
        captured.update(responses=responses, depth=depth, board=board, opp=opp_s, token=token)

    c2.updated.connect(cap_emit)
    c2._deepen_schedule = lambda *a, **k: [12]
    c2._predictive_turn(start, 3, False, 12, 22, 99)
    check("emit token preserved", captured.get("token") == 99, str(captured.get("token")))
    check("emit carries 3 replies", len(captured.get("responses", [])) == 3)
    tb = captured.get("board")
    check("emit board is after the opponent's best move, player to move",
          tb is not None and tb.turn == chess.BLACK and tb.board_fen() != start.board_fen())

    # live opponent look-ahead: candidates + responses refine over increasing depth,
    # then a deep stream is layered on the settled line (stubbed here so it can't
    # block). Ceiling pinned low (14) to keep the test quick.
    c2._deepen_schedule = EngineController._deepen_schedule.__get__(c2)   # restore the real one
    rounds = []
    c2.updated.disconnect(cap_emit)
    c2.updated.connect(lambda r, d, bd, o, t: rounds.append((d, r, bd)))
    c2._stream_player = lambda *a, **k: rounds.append(("post-stream", None, None))
    c2._analyze_opponent_turn(start, 3, "live", 14, True, 12, 14, 88)
    refine = [r for r in rounds if r[0] != "post-stream"]
    depths = [r[0] for r in refine]
    check("live look-ahead refines over increasing depth (>=2 rounds, ascending)",
          len(depths) >= 2 and depths == sorted(depths), str(depths))
    check("live look-ahead reaches the ceiling", depths and max(depths) == 14, str(depths))
    legal = all(r.move in bd.legal_moves for _, resp, bd in refine for r in resp)
    check("live look-ahead responses are legal on the analysed board", legal)
    check("deep stream layered on the settled line", any(r[0] == "post-stream" for r in rounds))
    c2._engine.quit()

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
