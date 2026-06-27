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
    # (player suggestions, depth, analysed Board, opponent suggestions list, token)
    updated = QtCore.Signal(list, int, object, object, int)
    failed = QtCore.Signal(str)
    ready = QtCore.Signal()
    LOOKAHEAD_DEPTH = 12   # depth for the opponent's candidate moves (kept shallow for speed)

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

    def _top_moves(self, board: chess.Board, multipv: int, depth: int) -> list:
        """One-shot multipv analysis -> ranked MoveSuggestions (opponent's reds)."""
        try:
            infos = self._engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
        except Exception:
            return []
        if isinstance(infos, dict):
            infos = [infos]
        out = []
        for rank, info in enumerate(infos, start=1):
            s = MoveSuggestion.from_info(info, board, rank)
            if s:
                out.append(s)
        return out

    def _analyze(self, board: chess.Board, multipv: int, mode: str, depth: int,
                 player_color: bool | None = None, token: int = 0) -> None:
        # Tempo model:
        #   * player to move  -> analyse the CURRENT position for the player.
        #   * opponent to move -> get the opponent's top candidate moves (reds),
        #     play their best, and analyse the PLAYER's responses to it (greens).
        # Game over (no legal moves) -> emit an empty result so the UI shows the
        # result cleanly instead of going silent on an unsearchable position.
        if board.legal_moves.count() == 0:
            self.updated.emit([], 0, board, [], token)
            return

        opponent_to_move = (player_color is not None and board.is_valid()
                            and board.turn != player_color)

        # Predictive mode (opponent to move): instead of three replies to the
        # opponent's single best move, show ONE best reply to EACH of the
        # opponent's top moves — so the player can prepare for every likely move,
        # not just the most likely one. Once the opponent actually moves it is the
        # player's turn and execution falls through to the normal multipv flow
        # below: i.e. it collapses to the same behaviour as live mode. (The live
        # and fixed modes are untouched.)
        if mode == "predictive" and opponent_to_move:
            self._analyze_predictive(board, multipv, player_color, token)
            return

        opp_suggestions, target_board = [], board
        if opponent_to_move:
            opp_suggestions = self._top_moves(board, multipv, self.LOOKAHEAD_DEPTH)
            if opp_suggestions:
                target_board = board.copy()
                target_board.push(opp_suggestions[0].move)

        if target_board.legal_moves.count() == 0:        # opponent's best move ends the game
            self.updated.emit([], 0, target_board, opp_suggestions, token)
            return

        n = min(multipv, target_board.legal_moves.count())
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
                        self._emit(latest, coherent, target_board, opp_suggestions, token)
            if not (self._wake.is_set() or self._shutdown) and len(latest) >= n:
                coherent = min(int(v.get("depth", 0)) for v in latest.values())
                if coherent != last_emit:
                    self._emit(latest, coherent, target_board, opp_suggestions, token)
        self._analysis = None

    def _analyze_predictive(self, board: chess.Board, multipv: int,
                            player_color: bool, token: int) -> None:
        """Opponent to move, Predictive mode: emit the opponent's top moves (reds)
        plus the player's single best reply to EACH of them (greens), so the player
        can prepare for every likely opponent move. Emitted once at the look-ahead
        depth — the deep, refining analysis happens for real once it is the player's
        turn (the normal live flow). Reply ``rank`` is paired with its opponent move
        (reply #1 answers the opponent's best move, #2 their 2nd, ...)."""
        opp_suggestions = self._top_moves(board, multipv, self.LOOKAHEAD_DEPTH)
        if not opp_suggestions:
            self.updated.emit([], 0, board, [], token)
            return
        responses = []
        for rank, opp in enumerate(opp_suggestions, start=1):
            if self._wake.is_set() or self._shutdown:
                return                          # a newer position arrived — abandon this one
            reply_board = board.copy()
            reply_board.push(opp.move)
            if reply_board.legal_moves.count() == 0:
                continue                        # this opponent move ends the game (no reply)
            best = self._top_moves(reply_board, 1, self.LOOKAHEAD_DEPTH)
            if best:
                reply = best[0]
                reply.rank = rank               # pair the reply with its opponent move
                responses.append(reply)
        if self._wake.is_set() or self._shutdown:
            return
        # Pass the position after the opponent's best move (turn == player) so the
        # eval POV and player/opponent colours resolve exactly as the normal flow.
        target_board = board.copy()
        target_board.push(opp_suggestions[0].move)
        self.updated.emit(responses, self.LOOKAHEAD_DEPTH, target_board, opp_suggestions, token)

    def _emit(self, latest, depth, board, opp_suggestions, token) -> None:
        out = []
        for rank, key in enumerate(sorted(latest), start=1):
            s = MoveSuggestion.from_info(latest[key], board, rank)
            if s:
                out.append(s)
        if out:
            self.updated.emit(out, depth, board, opp_suggestions, token)
