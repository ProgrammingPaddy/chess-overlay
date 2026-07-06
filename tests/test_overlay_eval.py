"""Eval styling is from the PLAYER's POV — proves 'playing as Black' is not broken.

The eval NUMBER is ABSOLUTE (negative = Black winning, per the agreed contract),
but the green/grey arrow and the green/red number colour must reflect whether a
move is good FOR THE PLAYER. Before the fix, a winning Black player saw their best
moves greyed out and their evals in red (in absolute terms Black's advantage is
negative), which read as 'no eval for the player as Black'. These tests lock the
player-POV colouring while keeping the number absolute, for White and Black alike.

Run: python tests/test_overlay_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
from src.engine import MoveSuggestion, win_prob_to_cp
from src.overlay import (DARK_RED, GOLD, GREEN, GREY, RED, _arrow_color,
                         _eval_text_color, build_annotations, build_puzzle_line)

GREEN_NUM = (120, 240, 150)
RED_NUM = (255, 95, 95)
PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}   {detail}")


def sug(uci, cp=None, mate=None, rank=1):
    return MoveSuggestion(move=chess.Move.from_uci(uci), score_cp=cp, mate_in=mate, rank=rank)


def rgb(c):
    return (c.red(), c.green(), c.blue())


def player_top(suggestions, white_to_move, gold=False):
    """Top player-side annotation (opponent reds disabled)."""
    anns = build_annotations(suggestions, opp_suggestions=None, show_opponent=False,
                             white_to_move=white_to_move, gold_enabled=gold)
    return anns[0]


# --- the player is WHITE -----------------------------------------------------
a = player_top([sug("e2e4", cp=150)], white_to_move=True)
check("white winning: green arrow", rgb(_arrow_color(a)) == GREEN, rgb(_arrow_color(a)))
check("white winning: absolute +1.50", a.label == "+1.50", a.label)
check("white winning: green number", rgb(_eval_text_color(a)) == GREEN_NUM)

a = player_top([sug("e2e4", cp=-150)], white_to_move=True)
check("white losing: grey arrow", rgb(_arrow_color(a)) == GREY, rgb(_arrow_color(a)))
check("white losing: absolute -1.50", a.label == "-1.50", a.label)
check("white losing: red number", rgb(_eval_text_color(a)) == RED_NUM)

# --- the player is BLACK (the regression) ------------------------------------
a = player_top([sug("e7e5", cp=150)], white_to_move=False)   # +150 from Black's POV => Black winning
check("black winning: green arrow (was grey)", rgb(_arrow_color(a)) == GREEN, rgb(_arrow_color(a)))
check("black winning: absolute -1.50 (number stays absolute)", a.label == "-1.50", a.label)
check("black winning: green number (was red)", rgb(_eval_text_color(a)) == GREEN_NUM,
      rgb(_eval_text_color(a)))

a = player_top([sug("e7e5", cp=-150)], white_to_move=False)  # Black is worse
check("black losing: grey arrow", rgb(_arrow_color(a)) == GREY, rgb(_arrow_color(a)))
check("black losing: absolute +1.50", a.label == "+1.50", a.label)
check("black losing: red number", rgb(_eval_text_color(a)) == RED_NUM)

# --- symmetry: a winning move looks the same for either colour ---------------
white_win = _arrow_color(player_top([sug("e2e4", cp=150)], white_to_move=True))
black_win = _arrow_color(player_top([sug("e7e5", cp=150)], white_to_move=False))
check("winning is green for both colours", rgb(white_win) == rgb(black_win) == GREEN)

# --- mate is player-POV too --------------------------------------------------
a = player_top([sug("d8h4", mate=2)], white_to_move=False)   # Black has the mate
check("black mating: green arrow", rgb(_arrow_color(a)) == GREEN, rgb(_arrow_color(a)))
check("black mating: absolute #-2", a.label == "#-2", a.label)

a = player_top([sug("e7e5", mate=-1)], white_to_move=False)  # Black is being mated
check("black mated: grey arrow", rgb(_arrow_color(a)) == GREY, rgb(_arrow_color(a)))
check("black mated: absolute #1", a.label == "#1", a.label)

a = player_top([sug("d8h4", mate=1)], white_to_move=False, gold=True)
check("black forced mate: gold when enabled", rgb(_arrow_color(a)) == GOLD, rgb(_arrow_color(a)))

# --- opponent reds are independent of the player's colour --------------------
anns = build_annotations([sug("e7e5", cp=20)], opp_suggestions=[sug("e2e4", cp=50)],
                         show_opponent=True, white_to_move=False, gold_enabled=False)
opp = [x for x in anns if x.opponent][0]
me = [x for x in anns if not x.opponent][0]
check("opponent move is red (player is black)", rgb(_arrow_color(opp)) == RED, rgb(_arrow_color(opp)))
check("opponent eval absolute +0.50 (their white move)", opp.label == "+0.50", opp.label)
check("my reply still green (player black, slightly better)", rgb(_arrow_color(me)) == GREEN,
      rgb(_arrow_color(me)))

anns = build_annotations([sug("e7e5", cp=20)], opp_suggestions=[sug("e2e4", cp=400, rank=1),
                                                                sug("d2d4", cp=10, rank=2)],
                         show_opponent=True, white_to_move=False, gold_enabled=True)
opp_dom = [x for x in anns if x.opponent and x.gold]
check("overwhelming opponent move -> dark red", bool(opp_dom)
      and rgb(_arrow_color(opp_dom[0])) == DARK_RED,
      rgb(_arrow_color(opp_dom[0])) if opp_dom else "no gold opp")

# --- policy mode (Maia 2): opacity = human likelihood, label = probability --------
def psug(uci, policy, win_prob, rank):
    return MoveSuggestion(chess.Move.from_uci(uci), win_prob_to_cp(win_prob), None,
                          rank=rank, win_prob=win_prob, policy=policy)

# player is White, slightly winning (win_prob 0.6); 3 human moves of decreasing odds
pol = [psug("e2e4", 0.45, 0.6, 1), psug("d2d4", 0.3, 0.6, 2), psug("g1f3", 0.1, 0.6, 3)]
anns = build_annotations(pol, None, show_opponent=False, white_to_move=True,
                         gold_enabled=True, policy_mode=True)
check("policy labels are percentages", all(a.label.endswith("%") for a in anns),
      str([a.label for a in anns]))
check("top human move labelled 45%", anns[0].label == "45%")
check("most-likely human move is fully opaque", anns[0].strength == 1.0)
check("opacity scales with human likelihood", anns[1].strength < anns[0].strength)
check("policy arrow green when the player is winning (no dominant move)",
      rgb(_arrow_color(anns[0])) == GREEN, rgb(_arrow_color(anns[0])))

# a clearly dominant human move (>=50%) goes gold
dom = build_annotations([psug("e2e4", 0.7, 0.6, 1), psug("d2d4", 0.1, 0.6, 2)], None,
                        show_opponent=False, white_to_move=True, gold_enabled=True, policy_mode=True)
check("dominant human move (>=50%) is gold", rgb(_arrow_color(dom[0])) == GOLD)

# Black player in a losing line: win_prob 0.3 (side-to-move = Black) -> grey
pol_b = build_annotations([psug("e7e5", 0.4, 0.3, 1)], None, show_opponent=False,
                          white_to_move=False, gold_enabled=True, policy_mode=True)
check("policy arrow grey when the player is worse", rgb(_arrow_color(pol_b[0])) == GREY,
      rgb(_arrow_color(pol_b[0])))

# --- puzzle solution-line move numbers (label 1,2,3…; same-square moves merge) ---
_pb = chess.Board()                                                  # White to move
_pv = [chess.Move.from_uci(u) for u in ("e2e4", "d7d5", "e4d5")]     # moves 2 & 3 both land on d5
_top = MoveSuggestion(_pv[0], 50, None, pv=_pv, rank=1)
_num = build_puzzle_line(_top, _pb, True, show_opponent=True, max_plies=8, move_numbers=True)
check("move numbers: first arrow labelled 1", _num[0].label == "1", _num[0].label)
check("move numbers: same-square moves merge to '2,3'",
      [a.label for a in _num] == ["1", "2,3", ""], str([a.label for a in _num]))
_ev = build_puzzle_line(_top, _pb, True, show_opponent=True, max_plies=8, move_numbers=False)
check("numbers off: only the current move carries an eval",
      [a.label for a in _ev] == ["+0.50", "", ""], str([a.label for a in _ev]))
_won = build_puzzle_line(_top, _pb, True, show_opponent=False, max_plies=8, move_numbers=True)
check("winning-only numbers by DISPLAYED order (opponent hidden -> 1,2)",
      [a.label for a in _won] == ["1", "2"], str([a.label for a in _won]))

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
