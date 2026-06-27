"""Runtime configuration, persisted to ``config.json`` at the project root.

Board and vision *calibration* are intentionally NOT persisted — they are
redone each session. (Stale capture coordinates from a previous run would
silently misalign the overlay, which is worse than asking to recalibrate.)
Only stable preferences live here.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"


@dataclass
class Config:
    # --- engine ---
    engine_path: str | None = None
    engine_depth: int = 18           # fixed-mode search depth
    engine_threads: int = 2
    engine_hash_mb: int = 256
    multipv: int = 3                 # number of best moves to show
    engine_mode: str = "live"        # "live" (streaming) | "fixed" (static depth) | "predictive" (reply to each likely opp move)
    analyze_for: str = "auto"        # "auto" (side on bottom) | "white" | "black"

    # --- preferences (stable across sessions) ---
    board_monitor: int = 0
    white_bottom: bool = True
    show_arrows: bool = True
    gold_moves: bool = True           # highlight a clearly-best / forced-mate move in gold
    show_border: bool = False         # draw the calibrated board outline in the overlay
    allow_illegal: bool = False      # accept recognized positions even if not a legal move
    show_predicted: bool = True      # show the opponent's best move (red) on their turn
    pause_on_drag: bool = True       # freeze the eval while a piece is held on the board
    auto_track: bool = True          # auto-track the live board (engine analysis on by default)

    @classmethod
    def load(cls) -> "Config":
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
            return cls(**known)
        return cls()

    def save(self) -> None:
        CONFIG_PATH.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
