"""Board-orientation detection from a recognized position — the chess-reasoning
way a strong player infers which side is which (pawn directions, king and piece
placement, pawn confrontations), rather than pixel brightness.

It is a logistic-regression classifier (learned from ~320k real Lichess puzzle
positions across every rating) over piece-position features, embedded here as a
plain weight vector so scoring is a single dot product — no ML runtime dependency.
Given a ``chess.Board`` it returns whether White's army sits on the LOW ranks (i.e.
the board, as represented, has White on the bottom) and a probability.

Validation: 99.06% on held-out positions (50% of which are endgames, the hard
case). On normally-tracked games (openings/midgames) it is well above that. See
tests/test_orientation.py. The model itself lives in orientation_model.json.

Why this matters: ``white_bottom`` drives both vision (pixel->square) and the
overlay (square->pixel); getting it wrong rotates everything 180 degrees. This lets
the app reason about orientation like a player and show what it believes (see the
overlay direction indicator) instead of relying on a brittle brightness heuristic.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import chess

_MODEL_PATH = Path(__file__).resolve().parent / "orientation_model.json"
_M = json.loads(_MODEL_PATH.read_text(encoding="utf-8"))
_W = _M["weights"]
_B = float(_M["bias"])
_PAIRS = [tuple(p) for p in _M["inter_pairs"]]

# piece-type order used to index the per-(type, colour, rank) count features
_PT = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
_PTIDX = {pt: i for i, pt in enumerate(_PT)}
_R, _F = chess.square_rank, chess.square_file


def _signals(board: chess.Board) -> list[float]:
    """The directional / relational signals a player reads: king ranks, pawn
    advancement and confrontations, piece centre-of-mass, back-rank occupancy.
    Must match the training feature function exactly."""
    pm = board.piece_map()
    wp: dict[int, list[int]] = {}
    bp: dict[int, list[int]] = {}
    wpr, bpr, wr, br = [], [], [], []
    w_r0 = b_r7 = b_r0 = w_r7 = 0
    for sq, p in pm.items():
        r, f = _R(sq), _F(sq)
        (wr if p.color else br).append(r)
        if r == 0:
            w_r0 += int(bool(p.color)); b_r0 += int(not p.color)
        elif r == 7:
            b_r7 += int(not p.color); w_r7 += int(bool(p.color))
        if p.piece_type == chess.PAWN:
            (wpr if p.color else bpr).append(r)
            (wp if p.color else bp).setdefault(f, []).append(r)
    wk, bk = board.king(chess.WHITE), board.king(chess.BLACK)
    wkr = _R(wk) if wk is not None else 3.5
    bkr = _R(bk) if bk is not None else 3.5
    conf = 0
    for f in range(8):
        if f in wp and f in bp:
            conf += 1 if max(wp[f]) < min(bp[f]) else -1
    diag = 0
    for f in wp:
        for nb in (f - 1, f + 1):
            if nb in bp:
                for rw in wp[f]:
                    diag += sum(1 if rb == rw + 1 else (-1 if rb == rw - 1 else 0) for rb in bp[nb])
    mwp = max(wpr) if wpr else 3.5
    mbp = min(bpr) if bpr else 3.5
    mwpr = (sum(wpr) / len(wpr)) if wpr else 3.5
    mbpr = (sum(bpr) / len(bpr)) if bpr else 3.5
    mwr = (sum(wr) / len(wr)) if wr else 3.5
    mbr = (sum(br) / len(br)) if br else 3.5
    return [
        bkr - wkr, float(wkr), float(bkr),
        mbpr - mwpr, mwp - 3.5, 3.5 - mbp,
        float(conf), float(diag),
        mbr - mwr,
        float(sum(1 for r in wpr if r >= 4) - sum(1 for r in bpr if r <= 3)),
        float((w_r0 + b_r7) - (b_r0 + w_r7)),
        float(len(wpr) + len(bpr)), float(len(wr) + len(br)),
    ]


def _counts(board: chess.Board) -> list[float]:
    c = [0.0] * 96
    for sq, p in board.piece_map().items():
        c[(_PTIDX[p.piece_type] * 2 + (0 if p.color else 1)) * 8 + _R(sq)] += 1.0
    return c


def _features(board: chess.Board) -> list[float]:
    s = _signals(board)
    inter = [s[i] * s[j] for i, j in _PAIRS]
    return _counts(board) + s + inter


def orientation_logit(board: chess.Board) -> float:
    """Signed score: > 0 => White's army is on the low ranks (White on the bottom).
    Magnitude is the model's confidence (a logit)."""
    f = _features(board)
    return sum(wi * fi for wi, fi in zip(_W, f)) + _B


def detect_orientation(board: chess.Board) -> tuple[bool, float]:
    """Return ``(white_bottom, probability)`` for a recognized position.

    ``white_bottom`` is True when White's army sits on the low ranks of ``board``;
    ``probability`` is P(white_bottom) in [0, 1] (so values near 0 or 1 are
    confident, ~0.5 is a genuinely ambiguous position — rare, mostly sparse
    endgames). Kings are required; without both, returns (True, 0.5)."""
    if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
        return True, 0.5
    z = orientation_logit(board)
    z = max(-60.0, min(60.0, z))
    p = 1.0 / (1.0 + math.exp(-z))
    return (z > 0.0), p
