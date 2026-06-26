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
    engine_mode: str = "live"        # "live" (streaming, instant) or "fixed" (static depth)
    analyze_for: str = "auto"        # "auto" (side on bottom) | "white" | "black"

    # --- preferences (stable across sessions) ---
    board_monitor: int = 0
    white_bottom: bool = True
    show_arrows: bool = True
    allow_illegal: bool = False      # accept recognized positions even if not a legal move
    show_predicted: bool = True      # show the opponent's best move (red) on their turn

    @classmethod
    def load(cls) -> "Config":
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
            return cls(**known)
        return cls()

    def save(self) -> None:
        CONFIG_PATH.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
