"""Combined (multi-engine) mode.

  1. build_combined_annotations (pure): each engine's picks in its own colour, SOLID on
     your turn / DASHED on the opponent's, per-engine ring, eval vs human-% labels, fade.
  2. Config.combined_* normalization (pure): partial/oversized saved dicts are healed.
  3. MultiController fan-out (stub children): player_color forced to None so each child
     analyses the current side to move; per-engine arrow counts; Maia 2 Elo swap on the
     opponent's turn; visibility spawns/kills a single child; results relayed tagged.

Run: python tests/test_combined.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
from PySide6 import QtCore

from src.config import Config
from src.engine import MoveSuggestion
from src.overlay import (ENGINE_COLORS, ENGINE_RING_SCALE, build_combined_annotations)

app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}   {detail}")


def sug(uci, cp=None, mate=None, rank=1, policy=None, win_prob=None):
    return MoveSuggestion(chess.Move.from_uci(uci), cp, mate, rank=rank,
                          win_prob=win_prob, policy=policy)


# --- 1. build_combined_annotations -------------------------------------------
b = chess.Board()                                   # White to move
sf = [sug("e2e4", cp=30, rank=1)]
lc = [sug("d2d4", cp=25, rank=1)]
anns = build_combined_annotations([("stockfish", sf, b), ("leela", lc, b)], opponent_turn=False)
by_move = {a.move.uci(): a for a in anns}
check("one arrow per engine move", len(anns) == 2, str(len(anns)))
check("stockfish arrow is cyan", by_move["e2e4"].color == ENGINE_COLORS["stockfish"])
check("leela arrow is green", by_move["d2d4"].color == ENGINE_COLORS["leela"])
check("your turn -> solid arrows", all(not a.dashed for a in anns))
check("eval label is absolute (+White)", by_move["e2e4"].label == "+0.30", by_move["e2e4"].label)
check("per-engine ring scale carried", by_move["d2d4"].ring == ENGINE_RING_SCALE["leela"])

# opponent's turn -> dashed
anns_o = build_combined_annotations([("stockfish", sf, b)], opponent_turn=True)
check("opponent turn -> dashed arrows", anns_o and anns_o[0].dashed is True)

# Maia 2 (policy) labels are human-% and fade with likelihood
maia = [sug("e2e4", policy=0.41, win_prob=0.55, rank=1),
        sug("g1f3", policy=0.18, win_prob=0.55, rank=2)]
anns_m = build_combined_annotations([("maia2", maia, b)], opponent_turn=False)
check("maia labels are human %", [a.label for a in anns_m] == ["41%", "18%"],
      str([a.label for a in anns_m]))
check("maia arrow is pink", anns_m[0].color == ENGINE_COLORS["maia2"])
check("more-likely human move is more opaque", anns_m[0].strength > anns_m[1].strength)

# a Black-to-move board flips the eval sign for the label (absolute)
bb = chess.Board(); bb.push_uci("e2e4")             # Black to move
anns_b = build_combined_annotations([("stockfish", [sug("e7e5", cp=20)], bb)], opponent_turn=False)
check("black-to-move eval label negated to absolute", anns_b[0].label == "-0.20", anns_b[0].label)


# --- 2. config normalization --------------------------------------------------
c = Config()
check("combined defaults present", set(c.combined_visible) == {"stockfish", "leela", "maia2"})
c.combined_visible = {"stockfish": False}           # a partial/old save
c.combined_lines = {"maia2": 99}                    # out of range
c._normalize_combined()
check("missing visibility keys filled from defaults",
      c.combined_visible == {"stockfish": False, "leela": True, "maia2": True},
      str(c.combined_visible))
check("arrow counts clamped to 1..5",
      c.combined_lines == {"stockfish": 1, "leela": 1, "maia2": 5}, str(c.combined_lines))


# --- 3. MultiController fan-out (stub children) -------------------------------
import src.multi_engine as me


class StubChild(QtCore.QObject):
    updated = QtCore.Signal(list, int, object, object, int)
    failed = QtCore.Signal(str)
    ready = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.reqs = []
        self.started = self.shut = False

    def start(self):
        self.started = True
        self.ready.emit()

    def request(self, board, multipv, mode, depth, player_color=None, token=0,
                opp_live=False, opp_depth=12, opp_max=22, limit_strength=False,
                player_elo=1500, opp_elo=1500, puzzle=False, puzzle_side=None):
        self.reqs.append(dict(multipv=multipv, mode=mode, player_color=player_color,
                              token=token, player_elo=player_elo, opp_elo=opp_elo,
                              limit=limit_strength))

    def clear(self):
        pass

    def reconfigure(self, *a):
        pass

    def shutdown(self):
        self.shut = True


made: dict = {}


def fake_build_single(key, cfg):
    ch = StubChild()
    made[key] = ch
    return ch, ""


me.build_single = fake_build_single                 # patch the names MultiController uses
me.availability = lambda k: (True, "")

cfg = Config()
cfg.engine = "combined"
cfg.combined_visible = {"stockfish": True, "leela": True, "maia2": True}
cfg.combined_lines = {"stockfish": 1, "leela": 2, "maia2": 3}
cfg.maia_player_elo, cfg.maia_opp_elo = 1400, 1800

mc = me.MultiController(cfg)
relayed = []
mc.combined_updated.connect(lambda k, s, d, b, o, t: relayed.append((k, s, d, o, t)))
mc.start()
check("one child spawned per visible engine", set(made) == {"stockfish", "leela", "maia2"},
      str(set(made)))
check("every child started", all(ch.started for ch in made.values()))

# player's turn (White to move, player is White): each child asked for the current STM
made["stockfish"].reqs.clear(); made["leela"].reqs.clear(); made["maia2"].reqs.clear()
mc.request(chess.Board(), 3, "live", 18, player_color=chess.WHITE, token=7)
check("fan-out forces player_color=None (analyse current STM)",
      all(ch.reqs[-1]["player_color"] is None for ch in made.values()))
check("per-engine arrow counts used as multipv",
      (made["stockfish"].reqs[-1]["multipv"], made["leela"].reqs[-1]["multipv"],
       made["maia2"].reqs[-1]["multipv"]) == (1, 2, 3))
check("searchers run full strength in combined (limiter off)",
      made["stockfish"].reqs[-1]["limit"] is False)
check("maia uses your Elo on your turn",
      made["maia2"].reqs[-1]["player_elo"] == 1400 and made["maia2"].reqs[-1]["opp_elo"] == 1800)

# opponent's turn (Black to move, player White): Maia predicts the mover at the mover's Elo
made["maia2"].reqs.clear()
ob = chess.Board(); ob.push_uci("e2e4")             # Black (opponent) to move
mc.request(ob, 3, "live", 18, player_color=chess.WHITE, token=8)
check("maia Elos swap on the opponent's turn (predict them at their rating)",
      made["maia2"].reqs[-1]["player_elo"] == 1800 and made["maia2"].reqs[-1]["opp_elo"] == 1400,
      str(made["maia2"].reqs[-1]))

# relay tags each child's emit with its engine key
made["leela"].updated.emit([sug("d2d4", cp=20)], 12, chess.Board(), [], 8)
check("relayed emit is tagged with the engine key", relayed and relayed[-1][0] == "leela",
      str(relayed[-1] if relayed else None))

# visibility toggle spawns / tears down a single child
me.availability = lambda k: (True, "")
mc.set_visible("leela", False)
check("hiding an engine shuts its child down", made["leela"].shut is True)
check("hidden engine removed from active set", "leela" not in mc.active_engines())
prev = made["stockfish"]
mc.set_visible("leela", True)
check("re-showing an engine respawns a fresh child",
      made["leela"] is not prev and made["leela"].started is True)

mc.shutdown()
check("shutdown joins every child", all(ch.shut for ch in made.values()))

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
