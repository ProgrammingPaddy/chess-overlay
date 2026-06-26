"""Persistent engine controller — one long-lived thread that owns Stockfish.

Instead of spawning a thread and opening a fresh analysis per move (the source
of the lag and the engine-desync crashes), the engine lives in ONE thread for
the whole session and you just feed it the current position via ``request()``.
Switching positions interrupts the running search in milliseconds and starts the
new one immediately — so the overlay updates feel instantaneous.

Emits ``updated(list[MoveSuggestion], depth)`` as the search refines. The engine
is created inside the worker thread and auto-respawned if it ever dies.
"""
from __future__ import annotations

import threading

import chess
import chess.engine
from PySide6 import QtCore

from src.engine import MoveSuggestion


class EngineController(QtCore.QThread):
    # (suggestions, depth, analysed Board, opponent_move | None, request token)
    updated = QtCore.Signal(list, int, object, object, int)
    failed = QtCore.Signal(str)
    ready = QtCore.Signal()
    LOOKAHEAD_DEPTH = 14   # depth for guessing the opponent's best move

    def __init__(self, engine_path: str, threads: int, hash_mb: int, parent=None):
        super().__init__(parent)
        self._path = engine_path
        self._threads = threads
        self._hash = hash_mb
        self._reconfigure = False
        self._lock = threading.Lock()
        self._pending: tuple | None = None     # (board, multipv, mode, depth)
        self._wake = threading.Event()
        self._shutdown = False
        self._analysis = None
        self._engine: chess.engine.SimpleEngine | None = None

    # ----- called from the GUI thread -----
    def request(self, board: chess.Board, multipv: int, mode: str, depth: int,
                player_color: bool | None = None, token: int = 0) -> None:
        with self._lock:
            self._pending = (board.copy(), multipv, mode, depth, player_color, token)
        self._interrupt()

    def clear(self) -> None:
        with self._lock:
            self._pending = None
        self._interrupt()

    def reconfigure(self, threads: int, hash_mb: int) -> None:
        with self._lock:
            self._threads, self._hash, self._reconfigure = threads, hash_mb, True
        self._interrupt()

    def shutdown(self) -> None:
        self._shutdown = True
        self._interrupt()
        self.wait(5000)

    def _interrupt(self) -> None:
        self._wake.set()
        if self._analysis is not None:
            try:
                self._analysis.stop()
            except Exception:
                pass

    # ----- worker thread -----
    def run(self) -> None:
        if not self._spawn_engine():
            return
        self.ready.emit()
        while not self._shutdown:
            self._wake.wait(0.5)
            self._wake.clear()
            if self._shutdown:
                break
            with self._lock:
                job, self._pending = self._pending, None
                reconf, self._reconfigure = self._reconfigure, False
                threads, hash_mb = self._threads, self._hash
            if reconf and self._engine is not None:
                try:
                    self._engine.configure({"Threads": threads, "Hash": hash_mb})
                except Exception:
                    pass
            if job is None:
                continue
            try:
                self._analyze(*job)
            except Exception as exc:
                self.failed.emit(str(exc))
                self._spawn_engine()      # respawn after a crash
        self._quit_engine()

    def _spawn_engine(self) -> bool:
        self._quit_engine()
        try:
            self._engine = chess.engine.SimpleEngine.popen_uci(self._path)
            self._engine.configure({"Threads": self._threads, "Hash": self._hash})
            return True
        except Exception as exc:
            self.failed.emit(f"engine failed to start: {exc}")
            return False

    def _quit_engine(self) -> None:
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None

    def _analyze(self, board: chess.Board, multipv: int, mode: str, depth: int,
                 player_color: bool | None = None, token: int = 0) -> None:
        # Tempo model:
        #   * player to move  -> analyse the CURRENT position for the player.
        #   * opponent to move -> take the opponent's rank-1 move, play it, and
        #     analyse the PLAYER's responses ("what I should be ready to play
        #     after their best move"). The opponent's move is reported as opp_move.
        opp_move, target_board = None, board
        looking = (player_color is not None and board.is_valid()
                   and board.turn != player_color and board.legal_moves.count() > 0)
        if looking:
            try:
                info = self._engine.analyse(board, chess.engine.Limit(depth=self.LOOKAHEAD_DEPTH))
                pv = info.get("pv")
                if pv:
                    opp_move = pv[0]
                    target_board = board.copy()
                    target_board.push(opp_move)
            except Exception:
                opp_move, target_board = None, board

        # If the opponent's best move ends the game, there are no player responses
        # — still surface the predicted move (red) so it is visible.
        if opp_move is not None and target_board.legal_moves.count() == 0:
            self.updated.emit([], 0, target_board, opp_move, token)
            return

        n = min(multipv, max(1, target_board.legal_moves.count()))
        limit = chess.engine.Limit(depth=depth) if mode == "fixed" else None
        latest: dict[int, dict] = {}
        last_emit = -1
        with self._engine.analysis(target_board, multipv=multipv, limit=limit) as analysis:
            self._analysis = analysis
            for info in analysis:
                if self._wake.is_set() or self._shutdown:
                    break
                if info.get("pv"):
                    latest[info.get("multipv", 1)] = info
                if len(latest) >= n:
                    coherent = min(int(v.get("depth", 0)) for v in latest.values())
                    if coherent > last_emit:
                        last_emit = coherent
                        self._emit(latest, coherent, target_board, opp_move, token)
            if not (self._wake.is_set() or self._shutdown) and len(latest) >= n:
                coherent = min(int(v.get("depth", 0)) for v in latest.values())
                if coherent != last_emit:
                    self._emit(latest, coherent, target_board, opp_move, token)
        self._analysis = None

    def _emit(self, latest, depth, board, opp_move, token) -> None:
        out = []
        for rank, key in enumerate(sorted(latest), start=1):
            s = MoveSuggestion.from_info(latest[key], board, rank)
            if s:
                out.append(s)
        if out:
            self.updated.emit(out, depth, board, opp_move, token)
