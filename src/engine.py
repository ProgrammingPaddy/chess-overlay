"""Stockfish (UCI) wrapper built on python-chess.

Provides ranked move suggestions with evaluations and principal-variation lines.
``MoveSuggestion.from_info`` is shared by the one-shot ``best_moves`` and the
streaming ``analysis.AnalysisWorker`` so both produce identical objects.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field

import chess
import chess.engine


def find_stockfish() -> str | None:
    """Locate a Stockfish binary: PATH, then ./engines/."""
    found = shutil.which("stockfish")
    if found:
        return found
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("stockfish.exe", "stockfish"):
        candidate = os.path.join(root, "engines", name)
        if os.path.isfile(candidate):
            return candidate
    return None


@dataclass
class MoveSuggestion:
    move: chess.Move
    score_cp: int | None        # centipawns from side-to-move POV (None on mate)
    mate_in: int | None         # moves-to-mate (None when not mate)
    pv: list[chess.Move] = field(default_factory=list)
    rank: int = 1               # 1 = best
    depth: int | None = None

    @property
    def uci(self) -> str:
        return self.move.uci()

    def eval_text(self) -> str:
        return self.eval_text_pov(False)

    def eval_text_pov(self, flip: bool) -> str:
        """Eval string; ``flip`` negates it to show the other side's POV
        (used so evals are always shown from the player's perspective)."""
        mate, cp = self.mate_in, self.score_cp
        if flip:
            mate = -mate if mate is not None else None
            cp = -cp if cp is not None else None
        if mate is not None:
            return f"#{mate}"
        if cp is None:
            return "?"
        return f"{cp / 100:+.2f}"

    @classmethod
    def from_info(cls, info: dict, board: chess.Board, rank: int) -> "MoveSuggestion | None":
        pv = info.get("pv") or []
        score = info.get("score")
        if not pv or score is None:
            return None
        pov = score.pov(board.turn)
        return cls(move=pv[0], score_cp=pov.score(), mate_in=pov.mate(),
                   pv=list(pv), rank=rank, depth=info.get("depth"))


class ChessEngine:
    """Context-manager-friendly wrapper around a UCI engine process."""

    def __init__(self, engine_path: str | None = None, *,
                 threads: int = 2, hash_mb: int = 256):
        self.engine_path = engine_path or self._find_stockfish()
        if not self.engine_path:
            raise FileNotFoundError(
                "Stockfish not found. Put the binary in the 'engines' folder "
                "(named stockfish.exe), on PATH, or set engine_path in config.json.")
        self._engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
        self._engine.configure({"Threads": threads, "Hash": hash_mb})

    @property
    def raw(self) -> chess.engine.SimpleEngine:
        """The underlying SimpleEngine (used by the streaming worker)."""
        return self._engine

    @staticmethod
    def _find_stockfish() -> str | None:
        return find_stockfish()

    def best_moves(self, board: chess.Board, *, multipv: int = 3,
                   depth: int | None = 18,
                   movetime: float | None = None) -> list[MoveSuggestion]:
        limit = chess.engine.Limit(depth=depth, time=movetime)
        infos = self._engine.analyse(board, limit, multipv=multipv)
        if isinstance(infos, dict):
            infos = [infos]
        out: list[MoveSuggestion] = []
        for rank, info in enumerate(infos, start=1):
            s = MoveSuggestion.from_info(info, board, rank)
            if s:
                out.append(s)
        return out

    def close(self) -> None:
        try:
            self._engine.quit()
        except Exception:
            pass

    def __enter__(self) -> "ChessEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
