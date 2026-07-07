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
    failed = QtCore.Signal(str)
    ready = QtCore.Signal()

    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self._cfg = cfg
        self._children: dict[str, object] = {}
        self._ready_emitted = False

    # ----- lifecycle (mirrors a QThread controller's start/shutdown) -----
    def start(self) -> None:
        for key in self._active_keys():
            self._spawn(key)
        if not self._children:                 # nothing to run — settle the UI anyway
            self._child_ready()

    def _active_keys(self) -> list[str]:
        return [k for k in COMBINED_ENGINES
                if self._cfg.combined_visible.get(k) and availability(k)[0]]

    def _spawn(self, key: str) -> None:
        ctrl, err = build_single(key, self._cfg)
        if ctrl is None:
            self.failed.emit(f"{key}: {err}")
            return
        # Re-tag each child's result with its engine key, then relay. (The lambda runs in
        # the child's thread; combined_updated then queues to the GUI-thread menu slot.)
        ctrl.updated.connect(
            lambda s, d, b, o, t, k=key: self.combined_updated.emit(k, s, d, b, o, t))
        ctrl.failed.connect(lambda m, k=key: self.failed.emit(f"{k}: {m}"))
        ctrl.ready.connect(self._child_ready)
        ctrl.start()
        self._children[key] = ctrl

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

    def clear(self) -> None:
        for c in list(self._children.values()):
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
        for c in list(self._children.values()):     # stop searching first…
            try:
                c.clear()
            except Exception:
                pass
        for c in list(self._children.values()):     # …then join each worker/process
            try:
                c.shutdown()
            except Exception:
                pass
        self._children.clear()
