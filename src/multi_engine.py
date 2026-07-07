"""Combined-mode controller — runs several engines at once and relays each one's
result tagged with which engine produced it, so the overlay can draw them together
in distinct colours.

Design (kept deliberately simple so the single-engine controllers are reused verbatim):

  * It owns one child controller per VISIBLE engine (built by engine_profiles.build_single),
    so an unchecked engine is never spawned and costs nothing — that is the compute lever.
  * Each request is fanned out with ``player_color=None``, which makes every child analyse
    the CURRENT side to move on the given board directly (no look-ahead, no reply line).
    So each child returns its best move(s) for whoever is to move — the player's moves on
    your turn, the opponent's predicted moves on theirs. The menu draws them solid vs
    dashed accordingly. This sidesteps the per-engine 'reply on a different predicted
    board' tangle a real look-ahead would create, and needs no change to the children.
  * Maia 2 is Elo-conditioned, so on the opponent's turn its self/oppo Elos are swapped
    (predict the mover at the mover's rating). Searchers ignore Elo.

Threading: this is a plain QObject living on the GUI thread; the children are the same
QThread/subprocess controllers as always. Child signals cross back to the GUI thread as
usual. Shutdown clears then joins every child.
"""
from __future__ import annotations

import chess
from PySide6 import QtCore

from src.engine_profiles import COMBINED_ENGINES, availability, build_single


class MultiController(QtCore.QObject):
    # (engine_key, player suggestions, depth, analysed Board, opponent suggestions, token)
    combined_updated = QtCore.Signal(str, list, int, object, object, int)
    # (checker_engine_key, evals of the Maia moves, analysed Board, token) — 'check Maia lines'
    check_updated = QtCore.Signal(str, list, object, int)
    failed = QtCore.Signal(str)
    ready = QtCore.Signal()

    CHECKERS = ["stockfish", "leela"]          # engines that can grade the Maia moves

    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self._cfg = cfg
        self._children: dict[str, object] = {}
        self._checkers: dict[str, object] = {}   # separate instances that eval the Maia moves
        self._ready_emitted = False

    # ----- lifecycle (mirrors a QThread controller's start/shutdown) -----
    def start(self) -> None:
        for key in self._active_keys():
            self._spawn(key)
        for key in self._active_check_keys():
            self._spawn_checker(key)
        if not self._children:                 # nothing to run — settle the UI anyway
            self._child_ready()

    def _active_keys(self) -> list[str]:
        return [k for k in COMBINED_ENGINES
                if self._cfg.combined_visible.get(k) and availability(k)[0]]

    def _active_check_keys(self) -> list[str]:
        # Checking only makes sense when Maia is on screen (its moves are what we grade).
        if not (self._cfg.combined_check_maia and self._cfg.combined_visible.get("maia2")):
            return []
        return [k for k in self.CHECKERS
                if self._cfg.combined_check_with.get(k) and availability(k)[0]]

    def _spawn(self, key: str) -> None:
        ctrl, err = build_single(key, self._cfg)
        if ctrl is None:
            self.failed.emit(f"{key}: {err}")
            return
        # Re-tag each child's result with its engine key, then relay. Maia's result also
        # triggers the checkers. (The lambda runs in the child's thread; the signal then
        # queues to the GUI-thread menu slot.)
        if key == "maia2":
            ctrl.updated.connect(self._on_maia_updated)
        else:
            ctrl.updated.connect(
                lambda s, d, b, o, t, k=key: self.combined_updated.emit(k, s, d, b, o, t))
        ctrl.failed.connect(lambda m, k=key: self.failed.emit(f"{k}: {m}"))
        ctrl.ready.connect(self._child_ready)
        ctrl.start()
        self._children[key] = ctrl

    def _spawn_checker(self, key: str) -> None:
        ctrl, err = build_single(key, self._cfg)
        if ctrl is None:
            self.failed.emit(f"{key} (check): {err}")
            return
        ctrl.updated.connect(lambda s, d, b, o, t, k=key: self.check_updated.emit(k, s, b, t))
        ctrl.failed.connect(lambda m, k=key: self.failed.emit(f"{k} (check): {m}"))
        ctrl.start()
        self._checkers[key] = ctrl

    def _on_maia_updated(self, suggestions, depth, board, opp, token) -> None:
        """Relay Maia's picks, then have each checker grade EXACTLY those moves (root_moves)
        on the same position — so the strong-engine evals land on the Maia arrows."""
        self.combined_updated.emit("maia2", suggestions, depth, board, opp, token)
        if self._checkers and suggestions and board is not None:
            moves = [s.move for s in suggestions]
            depth = int(getattr(self._cfg, "engine_depth", 18))
            for c in self._checkers.values():
                c.request(board, len(moves), "fixed", depth, None, token, root_moves=moves)

    def refresh_checkers(self) -> None:
        """Spawn/kill checker instances to match the current check settings (toggle,
        engine choice, or Maia visibility changed)."""
        want = set(self._active_check_keys())
        for key in list(self._checkers):
            if key not in want:
                ctrl = self._checkers.pop(key)
                try:
                    ctrl.clear()
                    ctrl.shutdown()
                except Exception:
                    pass
        for key in want:
            if key not in self._checkers:
                self._spawn_checker(key)

    def _child_ready(self) -> None:
        if not self._ready_emitted:
            self._ready_emitted = True
            self.ready.emit()

    def active_engines(self) -> list[str]:
        """Engine keys currently running (for the menu's status line)."""
        return [k for k in COMBINED_ENGINES if k in self._children]

    # ----- visibility toggle: spawn / tear down a single engine on demand -----
    def set_visible(self, key: str, on: bool) -> None:
        self._cfg.combined_visible[key] = bool(on)
        if on and key not in self._children and availability(key)[0]:
            self._spawn(key)
        elif not on and key in self._children:
            ctrl = self._children.pop(key)
            try:
                ctrl.clear()
                ctrl.shutdown()
            except Exception:
                pass
        if key == "maia2":                     # checkers only run while Maia is on screen
            self.refresh_checkers()

    # ----- request fan-out -----
    def request(self, board: chess.Board, multipv: int, mode: str, depth: int,
                player_color: bool | None = None, token: int = 0,
                opp_live: bool = False, opp_depth: int = 12, opp_max: int = 22,
                limit_strength: bool = False, player_elo: int = 1500,
                opp_elo: int = 1500, puzzle: bool = False,
                puzzle_side: bool | None = None) -> None:
        opp_turn = (player_color is not None and board.is_valid()
                    and board.turn != player_color)
        smode = mode if mode in ("live", "fixed") else "live"   # predictive N/A in combined
        for key, ctrl in list(self._children.items()):
            n = max(1, int(self._cfg.combined_lines.get(key, 1)))
            if key == "maia2":
                # Predict the side to move at the mover's OWN rating (swap on the opp turn).
                p, o = ((self._cfg.maia_opp_elo, self._cfg.maia_player_elo) if opp_turn
                        else (self._cfg.maia_player_elo, self._cfg.maia_opp_elo))
                ctrl.request(board, n, "live", depth, None, token, player_elo=p, opp_elo=o)
            else:
                # player_color=None => analyse the current side to move directly; full
                # strength so each engine shows its real opinion (limiter is a solo-mode aid).
                ctrl.request(board, n, smode, depth, None, token, opp_live=False,
                             limit_strength=False, player_elo=player_elo, opp_elo=opp_elo)
        # New position: drop any in-flight checks; they re-trigger when Maia next responds.
        for c in self._checkers.values():
            try:
                c.clear()
            except Exception:
                pass

    def clear(self) -> None:
        for c in list(self._children.values()) + list(self._checkers.values()):
            try:
                c.clear()
            except Exception:
                pass

    def reconfigure(self, threads: int, hash_mb: int) -> None:
        for c in list(self._children.values()):
            try:
                c.reconfigure(threads, hash_mb)
            except Exception:
                pass

    def shutdown(self) -> None:
        everyone = list(self._children.values()) + list(self._checkers.values())
        for c in everyone:                           # stop searching first…
            try:
                c.clear()
            except Exception:
                pass
        for c in everyone:                           # …then join each worker/process
            try:
                c.shutdown()
            except Exception:
                pass
        self._children.clear()
        self._checkers.clear()
