"""Opening identification — every curated line must be legal, and identify() must
name openings (and upgrade to the most specific one as the game deepens).

Run: python tests/test_openings.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
from src.openings import _OPENINGS, identify

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}   {detail}")


def play(*sans):
    b = chess.Board()
    for s in sans:
        b.push_san(s)
    return b


# every curated line is legal SAN
bad = []
for nm, line in _OPENINGS.items():
    b = chess.Board()
    try:
        for tok in line.split():
            b.push_san(tok)
    except Exception as exc:
        bad.append(f"{nm}: {exc}")
check(f"all {len(_OPENINGS)} curated lines are legal", not bad, "; ".join(bad[:5]))

# core identifications
check("1.e4 c5 -> Sicilian", identify(play("e4", "c5")) == "Sicilian Defense")
check("1.e4 e6 -> French", identify(play("e4", "e6")) == "French Defense")
check("1.d4 d5 2.c4 -> Queen's Gambit", identify(play("d4", "d5", "c4")) == "Queen's Gambit")
check("Ruy Lopez named",
      identify(play("e4", "e5", "Nf3", "Nc6", "Bb5")) == "Ruy Lopez")
check("Italian named",
      identify(play("e4", "e5", "Nf3", "Nc6", "Bc4")) == "Italian Game")
check("Grünfeld named",
      identify(play("d4", "Nf6", "c4", "g6", "Nc3", "d5")) == "Grünfeld Defense")

# deepest-match upgrade: Ruy Lopez -> Morphy Defense once ...a6 is played
check("upgrades to the most specific opening",
      identify(play("e4", "e5", "Nf3", "Nc6", "Bb5", "a6")) == "Ruy Lopez: Morphy Defense")

# name is retained after leaving book (deepest known position sticks)
deep = play("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7")
check("known name retained after leaving book", deep is not None and identify(deep) is not None)

# transposition by move order still resolves (EPD-based, order-independent):
# both orders reach the identical King's Indian position.
kid_a = play("d4", "Nf6", "c4", "g6", "Nc3", "Bg7")
kid_b = play("c4", "Nf6", "d4", "g6", "Nc3", "Bg7")
check("transposition reaches the same position", kid_a.epd() == kid_b.epd())
check("transposition resolves to the same opening",
      identify(kid_a) == identify(kid_b) == "King's Indian Defense",
      f"{identify(kid_a)} vs {identify(kid_b)}")

# the bare start position and a random early move are not falsely named
check("start position has no opening name", identify(chess.Board()) is None)
check("1.a3 is unnamed (out of book)", identify(play("a3")) is None)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
