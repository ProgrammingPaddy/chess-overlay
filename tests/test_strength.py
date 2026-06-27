"""Player-eval strength limiter.

  1. No engine: capability gating (no-op when unsupported) and that _analyze records
     the requested strength.
  2. Real Stockfish (else SKIP): UCI_LimitStrength/UCI_Elo are configured correctly,
     redundant sets are cached, 'full' disables the limit, the Elo is clamped, and a
     limited analysis still returns a legal move.

The opponent prediction always runs at full strength — the analysis code calls
_strength_full() before the reds and _strength_player() before the greens; that
split is covered structurally here (full toggles back off) and by the predictive
look-ahead tests.

Run: python tests/test_strength.py
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


class _EmptyAnalysis:
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
    c._engine = types.SimpleNamespace(analysis=lambda *a, **k: _EmptyAnalysis(), quit=lambda: None)
    return c


# --- 1. capability gating + plumbing (no engine) ------------------------------
c = fake_controller()
check("strength unsupported by default", c._strength_supported is False)
c._set_strength(True, 1500)
check("unsupported _set_strength is a no-op", c._applied_strength is None)
c._strength_full()
check("unsupported _strength_full is a no-op", c._applied_strength is None)

# _analyze records the requested strength (player to move after 1.e4, player = Black)
b = chess.Board()
b.push_san("e4")
c._analyze(b, 3, "live", 10, player_color=chess.BLACK, token=1, limit_strength=True, player_elo=1234)
check("_analyze records the requested strength", c._strength == (True, 1234))
c._analyze(chess.Board(), 3, "live", 10, player_color=chess.WHITE, token=2,
           limit_strength=False, player_elo=1234)
check("full-strength request records (False, ...)", c._strength[0] is False)


# --- 2. real engine: option wiring + caching + clamp --------------------------
path = find_stockfish()
if not path:
    print("  SKIP  no Stockfish found — engine strength test skipped")
else:
    c2 = EngineController(path, 1, 64)
    check("engine spawned", c2._spawn_engine() is True)
    check("Stockfish exposes strength limiting", c2._strength_supported is True)
    check("fresh engine forces re-apply (applied = None)", c2._applied_strength is None)

    configs = []
    real_configure = c2._engine.configure
    c2._engine.configure = lambda d: (configs.append(dict(d)), real_configure(d))[1]

    c2._set_strength(True, 1400)
    check("limit applied state", c2._applied_strength == (True, 1400))
    check("limit configured UCI options",
          bool(configs) and configs[-1].get("UCI_LimitStrength") is True
          and configs[-1].get("UCI_Elo") == 1400, str(configs[-1] if configs else None))

    n = len(configs)
    c2._set_strength(True, 1400)
    check("redundant set is cached (no extra configure)", len(configs) == n)

    c2._strength_full()
    check("full applied state", c2._applied_strength == (False, 0))
    check("full disables the limit", configs[-1].get("UCI_LimitStrength") is False)

    lo, hi = c2._elo_range
    c2._set_strength(True, lo - 500)
    check("Elo clamped up to the engine minimum", configs[-1].get("UCI_Elo") == lo, str((lo, hi)))
    c2._set_strength(True, hi + 500)
    check("Elo clamped down to the engine maximum", configs[-1].get("UCI_Elo") == hi)

    # a limited analysis still produces a legal move
    c2._set_strength(True, lo)
    tactical = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1")
    moves = c2._top_moves(tactical, 1, 8)
    check("limited analysis returns a legal move",
          bool(moves) and moves[0].move in tactical.legal_moves)
    c2._quit_engine()

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
