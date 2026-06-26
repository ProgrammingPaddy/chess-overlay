"""Temporal per-square consensus over recent recognition frames.

Per-frame vision can wobble (a moving cursor, an animation frame, sensor noise).
Voting each square across a short rolling window — weighted by per-cell
confidence — yields a stable consensus board plus an 'agreement' score that is
high only when recent frames concur. Transient single-frame errors are outvoted,
which is the core of recovering from bad guesses and keeping readings consistent.
"""
from __future__ import annotations

from collections import defaultdict, deque

import chess


class ConsensusBuffer:
    def __init__(self, window: int = 3):
        self.window = window
        self._frames: deque[dict[str, tuple[str, float]]] = deque(maxlen=window)

    def push(self, debug: list[dict]) -> None:
        """Record one recognition frame (the vision debug list)."""
        self._frames.append(
            {d["square"]: (d["piece"] or ".", float(d.get("conf", 0.0))) for d in debug})

    def ready(self) -> bool:
        return len(self._frames) >= self.window

    def clear(self) -> None:
        self._frames.clear()

    def consensus(self) -> tuple[chess.Board, float]:
        """Return (confidence-weighted voted board, agreement in [0, 1])."""
        weight: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        count: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for frame in self._frames:
            for sq, (sym, conf) in frame.items():
                weight[sq][sym] += max(conf, 0.05)
                count[sq][sym] += 1

        board = chess.Board.empty()
        agree_total, n = 0.0, 0
        for sq, syms in weight.items():
            best = max(syms, key=syms.get)
            total = sum(count[sq].values())
            agree_total += count[sq][best] / total if total else 0.0
            n += 1
            if best != ".":
                board.set_piece_at(chess.parse_square(sq), chess.Piece.from_symbol(best))
        return board, (agree_total / n if n else 0.0)
