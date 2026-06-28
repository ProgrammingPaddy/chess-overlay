"""Stockfish (UCI) wrapper built on python-chess.

Provides ranked move suggestions with evaluations and principal-variation lines.
``MoveSuggestion.from_info`` is shared by the one-shot ``best_moves`` and the
streaming ``analysis.AnalysisWorker`` so both produce identical objects.
"""
from __future__ import annotations

import math
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import chess
import chess.engine

# All alternative engines (lc0 binary, networks, the Maia 2 venv) live in a single
# sibling folder so the app's own repo stays clean. Override with $CHESS_ENGINES_DIR.
ENGINES_DIR = Path(os.environ.get(
    "CHESS_ENGINES_DIR", str(Path(__file__).resolve().parents[2] / "Chess Engines")))


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


def find_lc0() -> str | None:
    """Locate the lc0 binary (GPU build preferred, then CPU, then PATH)."""
    for p in (ENGINES_DIR / "lc0" / "lc0.exe", ENGINES_DIR / "lc0" / "cpu" / "lc0.exe",
              ENGINES_DIR / "lc0" / "lc0", ENGINES_DIR / "lc0" / "cpu" / "lc0"):
        if p.is_file():
            return str(p)
    return shutil.which("lc0")


def find_leela_network() -> str | None:
    """A general (strong) lc0 network: any .pb.gz under networks/ or bundled with lc0.
    Excludes the Maia rating nets (those are human-specific)."""
    cands = sorted((ENGINES_DIR / "networks").glob("*.pb.gz")) + \
        sorted((ENGINES_DIR / "lc0").glob("*.pb.gz"))
    return str(cands[0]) if cands else None


def list_maia_nets() -> dict[int, str]:
    """Maia human rating nets discovered on disk: {elo: path}, sorted by elo."""
    out: dict[int, str] = {}
    d = ENGINES_DIR / "networks" / "maia"
    if d.is_dir():
        for p in d.glob("maia-*.pb.gz"):
            m = re.search(r"maia-(\d+)", p.name)
            if m:
                out[int(m.group(1))] = str(p)
    return dict(sorted(out.items()))


def find_maia2_python() -> str | None:
    """The isolated Python interpreter for the Maia 2 (PyTorch) worker, if provisioned."""
    for p in (ENGINES_DIR / "maia2-env" / "Scripts" / "python.exe",
              ENGINES_DIR / "maia2-env" / "bin" / "python"):
        if p.is_file():
            return str(p)
    return None


def win_prob_to_cp(win_prob: float) -> int:
    """Map a side-to-move win probability (0..1) to a signed centipawn value so a
    probability-based engine (Leela WDL, Maia 2) flows through the SAME eval-colour
    machinery the search engines use. Logistic inverse, clamped."""
    wp = min(1 - 1e-4, max(1e-4, float(win_prob)))
    cp = -173.72 * math.log(1.0 / wp - 1.0)
    return int(max(-2000, min(2000, round(cp))))


@dataclass
class MoveSuggestion:
    move: chess.Move
    score_cp: int | None        # centipawns from side-to-move POV (None on mate)
    mate_in: int | None         # moves-to-mate (None when not mate)
    pv: list[chess.Move] = field(default_factory=list)
    rank: int = 1               # 1 = best
    depth: int | None = None
    # Engine-agnostic extras (None for plain Stockfish, so it is unaffected):
    win_prob: float | None = None   # side-to-move win probability 0..1 (Leela WDL, Maia 2)
    policy: float | None = None     # move probability 0..1 (Maia 2 human likelihood)

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
        # Leela (with UCI_ShowWDL) reports a WDL; carry it as a side-to-move win
        # probability. Stockfish (WDL off here) leaves this None -> unchanged.
        win_prob = None
        wdl = info.get("wdl")
        if wdl is not None:
            try:
                w = wdl.pov(board.turn)
                total = w.wins + w.draws + w.losses
                if total > 0:
                    win_prob = (w.wins + 0.5 * w.draws) / total
            except Exception:
                pass
        return cls(move=pv[0], score_cp=pov.score(), mate_in=pov.mate(),
                   pv=list(pv), rank=rank, depth=info.get("depth"), win_prob=win_prob)


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
