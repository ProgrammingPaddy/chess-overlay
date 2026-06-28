"""Maia 2 engine controller — same interface as EngineController, backed by the
isolated-Python worker (src/maia2_worker.py) over a JSON pipe.

Maia 2 is a single forward pass (no search), conditioned on player/opponent Elo, so
this controller is much simpler than the UCI one: each request is one (batched)
round-trip to the warm worker (~5 ms/inference on GPU), producing normalized
MoveSuggestions that carry the human-move probability (``policy``) and the position
win probability (``win_prob``). It maps into the same player/opponent tempo:

  * player to move    -> the player's likely human moves (at the player's Elo).
  * opponent to move  -> the opponent's likely moves (at the opponent's Elo) as the
    reds, plus the player's human reply — one per opponent move in Predictive mode,
    else replies to the opponent's most-likely move.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

import chess
from PySide6 import QtCore

from src.engine import MoveSuggestion, win_prob_to_cp

WORKER = Path(__file__).resolve().parent / "maia2_worker.py"


class Maia2Controller(QtCore.QThread):
    # Same signal shape as EngineController so the GUI is engine-agnostic.
    updated = QtCore.Signal(list, int, object, object, int)
    failed = QtCore.Signal(str)
    ready = QtCore.Signal()

    def __init__(self, worker_python: str, save_root: str, model_type: str = "rapid",
                 device: str = "gpu", parent=None):
        super().__init__(parent)
        self._worker_python = worker_python
        self._save_root = save_root
        self._type = model_type
        self._device = device
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._pending: tuple | None = None
        self._wake = threading.Event()
        self._shutdown = False
        self._rid = 0

    # ----- GUI-thread interface (mirrors EngineController) -----
    def request(self, board: chess.Board, multipv: int, mode: str, depth: int,
                player_color: bool | None = None, token: int = 0,
                opp_live: bool = False, opp_depth: int = 12, opp_max: int = 22,
                limit_strength: bool = False, player_elo: int = 1500,
                opp_elo: int = 1500) -> None:
        with self._lock:
            self._pending = (board.copy(), multipv, mode, player_color, token,
                             int(player_elo), int(opp_elo))
        self._wake.set()

    def clear(self) -> None:
        with self._lock:
            self._pending = None
        self._wake.set()

    def reconfigure(self, *_a, **_k) -> None:
        pass                                    # no threads/hash for a single forward pass

    def shutdown(self) -> None:
        self._shutdown = True
        self._wake.set()
        self.wait(6000)

    # ----- worker thread -----
    def run(self) -> None:
        if not self._spawn():
            return
        self.ready.emit()
        while not self._shutdown:
            self._wake.wait(0.5)
            self._wake.clear()
            if self._shutdown:
                break
            with self._lock:
                job, self._pending = self._pending, None
            if job is None:
                continue
            try:
                self._analyze(*job)
            except Exception as exc:
                self.failed.emit(f"Maia 2: {exc}")
                self._respawn()
        self._quit()

    def _popen_kwargs(self) -> dict:
        kw = dict(stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        if sys.platform == "win32":
            kw["creationflags"] = 0x08000000   # CREATE_NO_WINDOW — no console popup
        return kw

    def _spawn(self) -> bool:
        try:
            self._proc = subprocess.Popen(
                [self._worker_python, str(WORKER), "--type", self._type,
                 "--device", self._device, "--save-root", self._save_root],
                **self._popen_kwargs())
            banner = json.loads(self._proc.stdout.readline() or "{}")
            if not banner.get("ready"):
                self.failed.emit(f"Maia 2 worker failed: {banner.get('error', 'no response')}")
                return False
            return True
        except Exception as exc:
            self.failed.emit(f"Maia 2 failed to start: {exc}")
            return False

    def _respawn(self) -> None:
        self._quit()
        self._spawn()

    def _quit(self) -> None:
        if self._proc is not None:
            try:
                self._proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=2)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None

    def _query(self, queries: list[dict], top_k: int) -> list[dict]:
        self._rid += 1
        rid = self._rid
        self._proc.stdin.write(json.dumps({"id": rid, "top_k": top_k, "queries": queries}) + "\n")
        self._proc.stdin.flush()
        while True:                              # skip any stale lines until our id
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("worker closed the pipe")
            resp = json.loads(line)
            if resp.get("id") == rid:
                if "error" in resp:
                    raise RuntimeError(resp["error"])
                return resp.get("results", [])

    @staticmethod
    def _suggestions(result: dict, board: chess.Board, start_rank: int = 1) -> list:
        wp = result.get("win_prob")
        cp = win_prob_to_cp(wp) if wp is not None else None
        out = []
        for i, (uci, prob) in enumerate(result.get("moves", []), start=start_rank):
            try:
                mv = chess.Move.from_uci(uci)
            except Exception:
                continue
            if mv not in board.legal_moves:      # safety; Maia should only emit legal moves
                continue
            out.append(MoveSuggestion(move=mv, score_cp=cp, mate_in=None, rank=i,
                                      win_prob=wp, policy=float(prob)))
        return out

    def _superseded(self) -> bool:
        return self._wake.is_set() or self._shutdown

    def _analyze(self, board, multipv, mode, player_color, token, player_elo, opp_elo) -> None:
        if board.legal_moves.count() == 0:
            self.updated.emit([], 0, board, [], token)
            return

        opponent_to_move = (player_color is not None and board.is_valid()
                            and board.turn != player_color)
        if not opponent_to_move:
            res = self._query([{"fen": board.fen(), "elo_self": player_elo,
                                "elo_oppo": opp_elo}], multipv)
            self.updated.emit(self._suggestions(res[0], board) if res else [], 0, board, [], token)
            return

        # Opponent to move: their likely moves (reds) at the OPPONENT's Elo.
        res = self._query([{"fen": board.fen(), "elo_self": opp_elo,
                            "elo_oppo": player_elo}], multipv)
        opp_sugg = self._suggestions(res[0], board) if res else []
        if not opp_sugg:
            self.updated.emit([], 0, board, [], token)
            return
        if self._superseded():
            return

        if mode == "predictive":
            # one human reply (at the player's Elo) to EACH opponent move
            reply_boards, queries = [], []
            for opp in opp_sugg:
                b2 = board.copy()
                b2.push(opp.move)
                reply_boards.append(b2)
                queries.append({"fen": b2.fen(), "elo_self": player_elo, "elo_oppo": opp_elo})
            results = self._query(queries, 1)
            greens = []
            for rank, (b2, r) in enumerate(zip(reply_boards, results), start=1):
                top = self._suggestions(r, b2, start_rank=rank)
                if top:
                    greens.append(top[0])        # the single best human reply, paired to its opp move
            target = board.copy()
            target.push(opp_sugg[0].move)
            self.updated.emit(greens, 0, target, opp_sugg, token)
        else:
            target = board.copy()
            target.push(opp_sugg[0].move)
            if target.legal_moves.count() == 0:
                self.updated.emit([], 0, target, opp_sugg, token)
                return
            res2 = self._query([{"fen": target.fen(), "elo_self": player_elo,
                                 "elo_oppo": opp_elo}], multipv)
            greens = self._suggestions(res2[0], target) if res2 else []
            self.updated.emit(greens, 0, target, opp_sugg, token)
