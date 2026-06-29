"""Multi-engine backend: the normalized result schema, win-prob<->cp, WDL parsing,
engine discovery, and (gated on the engines being installed) that lc0 and the Maia 2
worker drive through the SAME interface and populate the engine-agnostic fields.

Run: python tests/test_engine_backends.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
import chess.engine
from PySide6 import QtCore

from src.engine import (ENGINES_DIR, MoveSuggestion, find_lc0, find_leela_network,
                        find_maia2_python, list_maia_nets, win_prob_to_cp)

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


# --- schema: the new fields default None so Stockfish is unaffected -----------
s = MoveSuggestion(chess.Move.from_uci("e2e4"), 30, None)
check("win_prob/policy default to None", s.win_prob is None and s.policy is None)

# --- win_prob -> cp: signed, monotonic, clamped ------------------------------
check("wp 0.5 -> ~0", abs(win_prob_to_cp(0.5)) <= 1)
check("wp 0.8 -> positive", win_prob_to_cp(0.8) > 0)
check("wp 0.2 -> negative", win_prob_to_cp(0.2) < 0)
check("monotonic in wp",
      win_prob_to_cp(0.95) > win_prob_to_cp(0.6) > win_prob_to_cp(0.5) > win_prob_to_cp(0.4))
check("clamped at the extremes", -2000 <= win_prob_to_cp(0.0) and win_prob_to_cp(1.0) <= 2000)


# --- from_info: pull win_prob from a WDL, else leave it None ------------------
class _Wdl:
    def __init__(self, w, d, l):
        self.wins, self.draws, self.losses = w, d, l


class _PovWdl:
    def __init__(self, wdl):
        self._wdl = wdl

    def pov(self, _color):
        return self._wdl


score = chess.engine.PovScore(chess.engine.Cp(30), chess.WHITE)
info = {"pv": [chess.Move.from_uci("e2e4")], "score": score,
        "wdl": _PovWdl(_Wdl(600, 300, 100)), "depth": 10}
s = MoveSuggestion.from_info(info, chess.Board(), 1)
check("from_info derives win_prob from WDL", s is not None and abs(s.win_prob - 0.75) < 1e-6,
      str(None if s is None else s.win_prob))
s2 = MoveSuggestion.from_info({"pv": [chess.Move.from_uci("e2e4")], "score": score, "depth": 10},
                             chess.Board(), 1)
check("from_info without WDL -> win_prob None", s2 is not None and s2.win_prob is None)


# --- engine discovery ---------------------------------------------------------
print(f"  (engines dir: {ENGINES_DIR})")
maia_nets = list_maia_nets()
check("maia rating nets discovered (or none installed)",
      len(maia_nets) >= 5 or not (ENGINES_DIR / "networks" / "maia").is_dir(),
      f"{len(maia_nets)} nets")


# --- lc0 via the SAME EngineController (gated on install) --------------------
lc0, net = find_lc0(), find_leela_network()
if lc0 and net:
    from src.analysis import EngineController
    c = EngineController(lc0, 2, 256, extra_options={"WeightsFile": net, "UCI_ShowWDL": True})
    if c._spawn_engine():
        sugg = c._top_moves(chess.Board(), 3, 6)
        check("lc0 yields multipv suggestions", len(sugg) >= 1, f"{len(sugg)}")
        check("lc0 populates win_prob (from WDL)", bool(sugg) and all(x.win_prob is not None for x in sugg))
        check("lc0 still produces a cp eval (search path unchanged)",
              bool(sugg) and all(x.score_cp is not None for x in sugg))
        c._quit_engine()
    else:
        print("  SKIP  lc0 present but failed to spawn")
else:
    print("  SKIP  lc0 / network not installed")


# --- Maia 2 via the worker (gated on install) --------------------------------
py = find_maia2_python()
if py:
    from src.maia2_engine import Maia2Controller
    c = Maia2Controller(py, str(ENGINES_DIR / "maia2_models"), "rapid", "gpu")
    if c._spawn():
        cap = {}
        c.updated.connect(lambda s, d, b, o, t: cap.update(s=s, o=o))
        c._analyze(chess.Board(), 3, "live", chess.WHITE, 1, 1500, 1500)
        greens = cap.get("s") or []
        check("maia2 yields human moves", len(greens) >= 1, f"{len(greens)}")
        check("maia2 populates policy (human likelihood)", bool(greens) and all(x.policy is not None for x in greens))
        check("maia2 populates win_prob + cp", bool(greens)
              and all(x.win_prob is not None and x.score_cp is not None for x in greens))
        # opponent turn (predictive): reds + one paired reply each
        cap.clear()
        c._analyze(chess.Board(), 3, "predictive", chess.BLACK, 2, 1300, 1700)
        check("maia2 opponent reds present", len(cap.get("o") or []) >= 1)
        # robustness: a terminal position (no legal moves — Maia raises on it) must
        # NOT crash the worker or fail the rest of the batch. (Fool's mate position.)
        mate_fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        res = c._query([{"fen": mate_fen, "elo_self": 1500, "elo_oppo": 1500},
                        {"fen": chess.Board().fen(), "elo_self": 1500, "elo_oppo": 1500}], 3)
        check("terminal query returns an empty (not failed) result", len(res) == 2 and res[0]["moves"] == [])
        check("a normal query in the same batch still works", len(res[1]["moves"]) >= 1)
        check("the worker is still alive after a terminal query", c._proc.poll() is None)
        c._quit()
    else:
        print("  SKIP  maia2 env present but worker failed to spawn")
else:
    print("  SKIP  maia2 env not installed")

# --- puzzle mode: converge on the decisive side (gated on Stockfish) ---------
from src.engine import find_stockfish

sf = find_stockfish()
if sf:
    from src.analysis import EngineController as _Sf
    cpz = _Sf(sf, 2, 64)
    if cpz._spawn_engine():
        cap = {}
        cpz.updated.connect(lambda s, d, b, o, t: cap.update(s=s, b=b, o=o))
        wmate = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")   # White mates with Ra8
        cpz._analyze_puzzle(wmate, 3, 12, 1)
        check("puzzle converges on White (who is mating)",
              cap.get("b") is not None and cap["b"].turn == chess.WHITE)
        check("puzzle shows White's mating move", cap.get("s")
              and cap["s"][0].move == chess.Move.from_uci("a1a8"))
        cap.clear()
        cpz._analyze_puzzle(chess.Board("r5k1/8/8/8/8/8/5PPP/6K1 b - - 0 1"), 3, 12, 2)
        check("puzzle converges on Black (who is mating)",
              cap.get("b") is not None and cap["b"].turn == chess.BLACK)
        check("puzzle shows Black's mating move", cap.get("s")
              and cap["s"][0].move == chess.Move.from_uci("a8a1"))
        # the 'whose turn' override pins the side even against the eval (force Black on
        # a White-to-mate position): greens become a Black move, reds stay White's mate.
        cap.clear()
        cpz._analyze_puzzle(wmate, 1, 12, 3, forced_side=chess.BLACK)
        check("forced-side puzzle solves for the chosen side (Black)",
              cap.get("b") is not None and cap["b"].turn == chess.BLACK)
        check("forced-side puzzle shows a Black move", cap.get("s")
              and cap["b"].piece_at(cap["s"][0].move.from_square).color == chess.BLACK)
        check("forced-side puzzle keeps the other side (White) as the reds",
              cap.get("o") and cap["o"][0].move == chess.Move.from_uci("a1a8"))
        # a puzzle is a single answer: multipv=1 yields exactly one green.
        cap.clear()
        cpz._analyze_puzzle(wmate, 1, 12, 4)
        check("puzzle with multipv=1 shows a single best move", len(cap.get("s") or []) == 1)
        cpz._quit_engine()
    else:
        print("  SKIP  stockfish puzzle (spawn failed)")
else:
    print("  SKIP  stockfish not found for the puzzle test")


# --- the engine registry / factory ------------------------------------------
from src.config import Config
from src.engine_profiles import ENGINE_ORDER, PROFILES, availability, make_controller
from src.maia2_engine import Maia2Controller as _Maia2

check("three engines registered", set(ENGINE_ORDER) == {"stockfish", "leela", "maia2"})
check("stockfish/leela display eval, maia2 displays policy",
      PROFILES["stockfish"].display == "eval" and PROFILES["leela"].display == "eval"
      and PROFILES["maia2"].display == "policy")
check("stockfish exposes the Elo limiter, maia2 exposes player/opp Elo",
      "strength_elo" in PROFILES["stockfish"].features
      and {"player_elo", "opp_elo"} <= PROFILES["maia2"].features)
check("stockfish is available", availability("stockfish")[0])

cfg = Config()
cfg.engine = "stockfish"
ctrl, _ = make_controller(cfg)
from src.analysis import EngineController as _Uci
check("factory builds Stockfish as a UCI controller", isinstance(ctrl, _Uci))
if availability("leela")[0]:
    cfg.engine = "leela"
    ctrl, _ = make_controller(cfg)
    check("factory builds Leela as a UCI controller", isinstance(ctrl, _Uci))
if availability("maia2")[0]:
    cfg.engine = "maia2"
    ctrl, _ = make_controller(cfg)
    check("factory builds Maia 2 as the Maia controller", isinstance(ctrl, _Maia2))

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
