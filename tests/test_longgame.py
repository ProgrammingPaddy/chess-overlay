"""Long-game stress test — drive the continuously-running GUI pipeline over a full
game (tracking, opening ID, SAN line, annotation building, arrow/eval colouring) to
shake out any crash that only shows up deep into a game (the 'died around move 70'
class of bug). Also exercises endgame/mate evals through the colour functions and,
if Stockfish is present, the engine on mate/stalemate positions.

Run: python tests/test_longgame.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
from src.engine import MoveSuggestion, find_stockfish
from src.openings import identify
from src.overlay import (_arrow_color, _eval_text_color, build_annotations,
                         visible_annotations)
from src.tracker import GameTracker

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}   {detail}")


def synth(board, n, base):
    """Assorted MoveSuggestions over a board's legal moves — mates, being-mated,
    0.00, and a spread of +/- centipawns — to stress every colour/opacity path."""
    out = []
    for i, mv in enumerate(list(board.legal_moves)[:n]):
        k = (base + i) % 7
        if k == 0:
            s = MoveSuggestion(mv, None, (i % 3) + 1)        # player mates in 1..3
        elif k == 1:
            s = MoveSuggestion(mv, None, -((i % 3) + 1))     # player being mated
        elif k == 2:
            s = MoveSuggestion(mv, 0, None)                  # dead level
        else:
            s = MoveSuggestion(mv, (i - 1) * 173, None)      # spread incl. negative
        s.rank = i + 1
        out.append(s)
    return out


def render(board, base):
    """Everything the GUI does per position; must never raise."""
    identify(board)
    player = synth(board, 3, base)
    opp = synth(board, 3, base + 3)
    anns = build_annotations(player, opp, True, board.turn == chess.WHITE, True)
    for a in anns:
        _arrow_color(a)
        _eval_text_color(a)
    visible_annotations(board, anns)


# --- a long, tracked game (random but seeded) --------------------------------
random.seed(20240607)
errors = []
plies = 0
for game_no in range(3):                       # a few independent games to vary endgames
    tracker = GameTracker()
    tracker.reset()
    live = chess.Board()
    render(tracker.board, plies)
    for _ in range(140):
        if live.is_game_over() or not live.legal_moves:
            break
        mv = random.choice(list(live.legal_moves))
        live.push(mv)
        try:
            applied = tracker.update_to(live.copy())   # vision feeds the new placement
            if applied is None:                        # unreachable jump -> resync like the app
                tracker.reset(live.copy(), previous=tracker.board.copy())
            render(tracker.board, plies)
            tracker.san_line()
        except Exception as exc:                       # capture, keep going
            import traceback
            errors.append(f"game{game_no} ply{plies}: {exc!r}\n{traceback.format_exc()}")
        plies += 1

check(f"no exception across {plies} tracked plies", not errors, errors[0] if errors else "")
check("reached a deep game (>=120 plies)", plies >= 120, f"only {plies}")

# --- explicit endgame / mate evals through the colour path -------------------
edge = [
    MoveSuggestion(chess.Move.from_uci("d1h5"), None, 1),    # mate in 1
    MoveSuggestion(chess.Move.from_uci("e2e4"), None, -1),   # mated in 1
    MoveSuggestion(chess.Move.from_uci("a2a4"), 0, None),    # 0.00
    MoveSuggestion(chess.Move.from_uci("b2b4"), 3000, None), # winning endgame
    MoveSuggestion(chess.Move.from_uci("c2c4"), -3000, None),
]
for i, s in enumerate(edge):
    s.rank = i + 1
edge_ok = True
for wtm in (True, False):
    try:
        for a in build_annotations(edge, edge, True, wtm, True):
            _arrow_color(a)
            _eval_text_color(a)
    except Exception as exc:
        edge_ok = False
        print("   edge fail:", repr(exc))
check("endgame/mate evals colour cleanly (both POVs)", edge_ok)

# --- the engine on terminal-ish positions (skipped without Stockfish) --------
path = find_stockfish()
if not path:
    print("  SKIP  no Stockfish — engine endgame smoke skipped")
else:
    import chess.engine
    eng = chess.engine.SimpleEngine.popen_uci(path)
    try:
        positions = [
            "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",          # black is checkmated (no legal moves)
            "7k/5Q2/5K2/8/8/8/8/8 b - - 0 1",          # black stalemated (no legal moves)
            "8/8/8/8/8/5k2/6q1/7K w - - 0 1",          # white to move, getting mated
            "8/P7/8/8/8/8/8/k1K5 w - - 0 1",           # promotion endgame
        ]
        eok = True
        for fen in positions:
            b = chess.Board(fen)
            if b.legal_moves.count() == 0:
                continue                                # engine isn't asked to search a finished game
            try:
                eng.analyse(b, chess.engine.Limit(depth=10), multipv=3)
            except Exception as exc:
                eok = False
                print("   engine fail:", fen, repr(exc))
        check("engine analyses endgame positions without error", eok)
    finally:
        eng.quit()

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
