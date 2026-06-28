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
    # Opponent look-ahead (live & predictive modes): a one-shot preview at
    # opp_lookahead_depth by default; when opp_lookahead_live is on it refines from
    # that depth up to opp_lookahead_max over time. Predictive ALWAYS refines its
    # per-move replies regardless of the toggle.
    opp_lookahead_live: bool = False
    opp_lookahead_depth: int = 12    # one-shot / preview depth (also the live-refine start)
    opp_lookahead_max: int = 22      # live-refine ceiling
    # Player-eval strength limiter (simulated Elo). Default OFF = full strength,
    # which must stay completely unaffected. Limits ONLY the player's eval
    # (greens); the opponent prediction (reds) always stays full strength.
    limit_player_strength: bool = False
    player_elo: int = 1500
    # --- engine selection (Stockfish default; Leela/Maia 2 are optional add-ons) ---
    engine: str = "stockfish"        # "stockfish" | "leela" | "maia2"
    maia_player_elo: int = 1500      # Maia 2: your rating (the moves IT predicts for you)
    maia_opp_elo: int = 1500         # Maia 2: opponent rating (predicts THEIR moves)
    maia_model: str = "rapid"        # Maia 2 model: "rapid" | "blitz"
    maia_device: str = "gpu"         # Maia 2 device: "gpu" | "cpu"
    leela_network: str = ""          # Leela weights path ("" = auto-pick the strongest)

    # --- preferences (stable across sessions) ---
    board_monitor: int = 0
    white_bottom: bool = True        # orientation: is the WHITE army on the bottom? (from vision)
    player_side: str = "bottom"      # which SEAT you're in: "bottom" | "top" (your colour is derived)
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
