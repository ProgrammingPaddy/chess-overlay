"""Engine registry — what each selectable engine is, which GUI controls apply to it,
how it's built, and whether it's installed. Selecting an engine loads its profile,
which the menu uses to show/hide the relevant option groups and to build the right
controller. Stockfish is the default and always available.

The ``features`` set is the contract with the GUI: a control group is shown iff its
flag is in the active engine's features.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.analysis import EngineController
from src.engine import (ENGINES_DIR, find_lc0, find_leela_network, find_maia2_python,
                        find_stockfish, list_maia_nets)
from src.maia2_engine import Maia2Controller


@dataclass(frozen=True)
class EngineProfile:
    key: str
    label: str
    display: str            # "eval" (cp/mate) | "policy" (human move %)
    features: frozenset      # GUI feature flags this engine exposes
    blurb: str = ""


# feature flags: mode, multipv, depth, threads, hash, opp_lookahead, strength_elo,
#                leela_network, player_elo, opp_elo, maia_model
PROFILES: dict[str, EngineProfile] = {
    "stockfish": EngineProfile(
        "stockfish", "Stockfish (default)", "eval",
        frozenset({"mode", "multipv", "depth", "threads", "hash",
                   "opp_lookahead", "strength_elo"}),
        "Strong tactical search. Strength limiter via simulated Elo."),
    "leela": EngineProfile(
        "leela", "Leela (lc0)", "eval",
        frozenset({"mode", "multipv", "depth", "threads",
                   "opp_lookahead", "leela_network"}),
        "Neural-net engine, positional/human style; WDL eval. "
        "Strength = the chosen network."),
    "maia2": EngineProfile(
        "maia2", "Maia 2 (human)", "policy",
        frozenset({"mode", "multipv", "player_elo", "opp_elo", "maia_model"}),
        "Predicts the move a human of a given Elo would actually play "
        "(probabilities, no search)."),
}

ENGINE_ORDER = ["stockfish", "leela", "maia2"]


def availability(key: str) -> tuple[bool, str]:
    """(installed?, reason-if-not) — the menu uses this to gate selection."""
    if key == "stockfish":
        return (find_stockfish() is not None, "Stockfish not found in engines/.")
    if key == "leela":
        if find_lc0() is None:
            return (False, "lc0 not found — run setup/download_engines.sh.")
        if find_leela_network() is None:
            return (False, "No Leela network in 'Chess Engines/networks'.")
        return (True, "")
    if key == "maia2":
        if find_maia2_python() is None:
            return (False, "Maia 2 env not set up — run setup/provision_maia2.sh.")
        return (True, "")
    return (False, "Unknown engine.")


def leela_networks() -> list[tuple[str, str]]:
    """(label, path) choices for the Leela engine: the strong general net plus the
    Maia rating nets (which run on lc0 too, as a human-strength option)."""
    out: list[tuple[str, str]] = []
    strong = find_leela_network()
    if strong:
        import os
        out.append((f"Strong — {os.path.basename(strong)}", strong))
    for elo, path in list_maia_nets().items():
        out.append((f"Human ~{elo} (Maia)", path))
    return out


def make_controller(cfg):
    """Build the controller for the configured engine. Returns (controller, error).
    All controllers share the EngineController signal/request interface."""
    key = cfg.engine
    ok, reason = availability(key)
    if not ok:
        return None, reason
    try:
        if key == "leela":
            net = cfg.leela_network or find_leela_network()
            return EngineController(find_lc0(), cfg.engine_threads, cfg.engine_hash_mb,
                                    extra_options={"WeightsFile": net, "UCI_ShowWDL": True}), ""
        if key == "maia2":
            return Maia2Controller(find_maia2_python(), str(ENGINES_DIR / "maia2_models"),
                                   cfg.maia_model, cfg.maia_device), ""
        path = cfg.engine_path or find_stockfish()
        return EngineController(path, cfg.engine_threads, cfg.engine_hash_mb), ""
    except Exception as exc:
        return None, f"{key} failed to start: {exc}"
