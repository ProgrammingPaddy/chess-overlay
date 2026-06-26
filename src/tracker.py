"""Game tracker — the rules/temporal reconciliation core.

Maintains an authoritative game state and accepts a new observed position only
when it is reachable from the current one by a short sequence of LEGAL moves.
This rejects per-frame vision noise (illegal readings simply don't connect) and
yields the move list in chess notation for free.

Hardened for real play:
  * Whose turn it is can't be read from one frame, so when the baseline turn is
    unknown (after a mid-game reseed) ``update_to`` tries BOTH sides and adopts
    whichever explains the change — the moved piece's color is unambiguous.
  * Castling rights are inferred from piece positions on a reseed so castling
    moves can still be recognized.
  * ``update_to`` returns ``None`` (not ``[]``) when nothing connects, so the
    caller can resync after the board has clearly moved on (fast/illegal jumps).
"""
from __future__ import annotations

import chess


def mover_color(prev: chess.Board | None, new: chess.Board) -> bool | None:
    """Whose move produced ``new`` from ``prev`` — the colour of the piece(s) that
    appeared on changed squares. Robust for normal moves, captures, castling and
    en passant (only one side's piece lands on a new square). None if unknown."""
    if prev is None:
        return None
    appeared = [new.piece_at(sq).color for sq in chess.SQUARES
                if new.piece_at(sq) is not None and new.piece_at(sq) != prev.piece_at(sq)]
    if not appeared:
        return None
    return chess.WHITE if appeared.count(chess.WHITE) >= appeared.count(chess.BLACK) else chess.BLACK


class GameTracker:
    def __init__(self):
        self.reset()

    def reset(self, observed: chess.Board | None = None,
              previous: chess.Board | None = None) -> None:
        """Seed the baseline from ``observed`` (placement only), or the start.

        If ``previous`` is given, the side to move is inferred from the diff (the
        opposite of whoever just moved) — so a resync is never 'turn unknown'."""
        if observed is None or observed.board_fen() == chess.Board().board_fen():
            self.board = chess.Board()
            self.turn_known = True          # standard start is unambiguously White
        else:
            self.board = observed.copy()
            self._infer_castling()
            mover = mover_color(previous, observed)
            if mover is not None:
                self.board.turn = not mover
                self.turn_known = True
            else:
                self.turn_known = False     # truly unknown (e.g. first read, no prior)
        self._base = self.board.copy()

    def _infer_castling(self) -> None:
        b = self.board

        def at(square: chess.Square, symbol: str) -> bool:
            pc = b.piece_at(square)
            return pc is not None and pc.symbol() == symbol

        rights = ""
        if at(chess.E1, "K"):
            rights += "K" if at(chess.H1, "R") else ""
            rights += "Q" if at(chess.A1, "R") else ""
        if at(chess.E8, "k"):
            rights += "k" if at(chess.H8, "r") else ""
            rights += "q" if at(chess.A8, "r") else ""
        try:
            b.set_castling_fen(rights or "-")
        except Exception:
            b.set_castling_fen("-")

    def set_turn(self, white_to_move: bool) -> None:
        """Manually pin whose turn it is (the 'flip side to move' button).

        Rebuilds from the current placement so the move history can't become
        inconsistent with the new side to move (which could crash SAN replay)."""
        placement = self.board.board_fen()
        self.board = chess.Board(f"{placement} {'w' if white_to_move else 'b'} - - 0 1")
        self._base = self.board.copy()
        self.turn_known = True

    def update_to(self, observed: chess.Board, max_plies: int = 2):
        """Advance toward ``observed`` (compared by placement).

        Returns the SAN of moves applied, ``[]`` if unchanged, or ``None`` if the
        target isn't reachable within ``max_plies`` (caller may resync)."""
        target = observed.board_fen()
        if target == self.board.board_fen():
            return []
        turns = [self.board.turn] if self.turn_known else [chess.WHITE, chess.BLACK]
        for depth in range(1, max_plies + 1):
            for turn in turns:
                trial = self.board.copy()
                trial.turn = turn
                moves = self._search(trial, target, depth)
                if moves is not None:
                    if not self.turn_known:
                        self._base.turn = turn
                    self.board.turn = turn
                    sans = []
                    for mv in moves:
                        sans.append(self.board.san(mv))
                        self.board.push(mv)
                    self.turn_known = True
                    return sans
        return None

    @staticmethod
    def _search(state: chess.Board, target_fen: str, depth: int):
        for mv in state.legal_moves:
            state.push(mv)
            if state.board_fen() == target_fen:
                state.pop()
                return [mv]
            deeper = GameTracker._search(state, target_fen, depth - 1) if depth > 1 else None
            state.pop()
            if deeper is not None:
                return [mv] + deeper
        return None

    def san_line(self) -> str:
        """The game so far as '1. e4 e5 2. Nf3 ...' (from the current baseline)."""
        replay = self._base.copy()
        parts = []
        for i, mv in enumerate(self.board.move_stack):
            if replay.turn == chess.WHITE:
                parts.append(f"{replay.fullmove_number}.")
            elif i == 0:
                parts.append(f"{replay.fullmove_number}...")
            parts.append(replay.san(mv))
            replay.push(mv)
        return " ".join(parts)
