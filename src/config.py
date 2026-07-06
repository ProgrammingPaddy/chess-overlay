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
    # Orientation and player colour are INDEPENDENT (decoupled on purpose):
    #   * white_bottom = which army is on the bottom of the SCREEN. Drives vision
    #     (pixel->square) and the overlay (square->pixel). Detected by vision; the
    #     "Flip board orientation" button is the manual fallback. It is about the
    #     screen, never about who you are.
    #   * player_colour_mode = who YOU play, for the green (mine) / red (opponent)
    #     split ONLY. "auto" = whoever is on the bottom; "white"/"black" force it.
    #     Changing it never rotates the board; flipping the board never changes it.
    white_bottom: bool = True        # orientation: is the WHITE army on the bottom? (from vision)
    player_colour_mode: str = "auto"  # "auto" (bottom army) | "white" | "black"
    show_arrows: bool = True
    gold_moves: bool = True           # highlight a clearly-best / forced-mate move in gold
    show_border: bool = False         # draw the calibrated board outline in the overlay
    show_orientation: bool = True     # draw the board-direction indicator in the overlay
    auto_orient: bool = True          # auto-correct orientation when the CV confidently disagrees
    allow_illegal: bool = False      # accept recognized positions even if not a legal move
    show_predicted: bool = True      # show the opponent's best move (red) on their turn
    pause_on_drag: bool = True       # freeze the eval while a piece is held on the board
    # Play mode is a top-level switch:
    #   * "live"   = normal play (track the game, show your moves + opponent reds).
    #   * "puzzle" = treat the on-screen position as an ISOLATED puzzle: analyse both
    #     sides, converge on the decisive side (the one whose best move wins), and
    #     show only THAT side's single best move. Eval engines only (not Maia).
    play_mode: str = "live"          # "live" | "puzzle"
    puzzle_winning_only: bool = False  # puzzle: show only the winning side's move (hide the other side's)
    # Puzzle look-ahead: how many half-moves PAST the current move to also draw (the
    # solution line — your moves green, the current one gold, opponent replies red,
    # fainter the deeper in the line). 0 = just the move to play now. Default 1.
    puzzle_lookahead: int = 1
    # Label the solution arrows with their move number in the line (1 = move to play now,
    # 2 = next, …) instead of the eval. Moves landing on the same square share a label.
    puzzle_move_numbers: bool = False
    # Move-highlight side detection: an OPTIONAL, gated layer that reads the site's
    # last-move square highlight to fix whose-move-it-is with certainty (off by default;
    # never touches piece recognition — a wrong/absent read simply abstains).
    puzzle_use_highlight: bool = False
    # Mover-on-bottom: puzzle boards are shown from the side-to-move's perspective, so the
    # BOTTOM army IS the side to move. Deriving the side from the (~99%) orientation beats
    # the ~91% engine standout, so this is the authoritative side source for a FRESH puzzle
    # when on (highlight/standout defer). Only the initial position — parity governs after.
    puzzle_mover_on_bottom: bool = True
    auto_track: bool = True          # auto-track the live board (engine analysis on by default)

    @classmethod
    def load(cls) -> "Config":
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            cls._migrate(data)
            known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
            return cls(**known)
        return cls()

    @staticmethod
    def _migrate(data: dict) -> None:
        """Carry forward settings from older config layouts (unknown keys are dropped
        on load, so legacy ones must be translated here before that)."""
        # The old puzzle_mode bool became the "puzzle" play_mode.
        if "play_mode" not in data and data.get("puzzle_mode"):
            data["play_mode"] = "puzzle"
        # The old bottom/top seat became an explicit colour. A bottom seat is exactly
        # what "auto" already does (you're the bottom army); only a top seat needs a
        # forced colour to preserve the prior derived colour (= the top army).
        if "player_colour_mode" not in data and data.get("player_side") == "top":
            data["player_colour_mode"] = "black" if data.get("white_bottom", True) else "white"

    def save(self) -> None:
        CONFIG_PATH.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
