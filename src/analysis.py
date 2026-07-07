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
    LOOKAHEAD_DEPTH = 12   # default one-shot / preview opponent look-ahead depth
    LOOKAHEAD_STEP = 2     # depth increment per 'live' refinement round
    PUZZLE_PICK_DEPTH = 12      # shallow both-sides pass to pick the side (once per puzzle)

    def __init__(self, engine_path: str, threads: int, hash_mb: int, parent=None,
                 extra_options: dict | None = None):
        super().__init__(parent)
        self._path = engine_path
        self._threads = threads
        self._hash = hash_mb
        # Engine-specific UCI options (e.g. lc0's WeightsFile / UCI_ShowWDL). They
        # are capability-filtered on apply, so Stockfish (which has none of them) is
        # entirely unaffected and this same class drives lc0 too.
        self._extra_options = dict(extra_options or {})
        self._reconfigure = False
        self._lock = threading.Lock()
        self._pending: tuple | None = None     # (board, multipv, mode, depth)
        self._wake = threading.Event()
        self._shutdown = False
        self._analysis = None
        self._engine: chess.engine.SimpleEngine | None = None
        # Set when a helper catches a DEAD engine (EngineTerminatedError). The blocking /
        # look-ahead helpers (_top_moves, _analyse_blocking) swallow errors to keep the
        # thread alive, so without this flag a dead engine would go unnoticed on the
        # OPPONENT path (its only caller) and the reds would stay broken for the rest of
        # the session, while the player path silently respawns. run() respawns before the
        # next job when this is set, so both paths recover identically.
        self._engine_broken = False
        # Player-eval strength limiter (applied per-analysis; opponent stays full).
        self._strength = (False, 0)            # (limit, elo) requested for THIS job
        self._applied_strength: tuple | None = None   # what the engine is configured to now
        self._strength_supported = False
        self._elo_range = (1320, 3190)

    # ----- called from the GUI thread -----
    def request(self, board: chess.Board, multipv: int, mode: str, depth: int,
                player_color: bool | None = None, token: int = 0,
                opp_live: bool = False, opp_depth: int = 12, opp_max: int = 22,
                limit_strength: bool = False, player_elo: int = 1500,
                opp_elo: int = 1500, puzzle: bool = False,
                puzzle_side: bool | None = None) -> None:
        # opp_elo is part of the shared engine interface (used by the Maia 2
        # controller); the search engines ignore it. puzzle_side (None = auto-pick the
        # decisive side; True/False = solve for White/Black) only applies when puzzle.
        with self._lock:
            self._pending = (board.copy(), multipv, mode, depth, player_color, token,
                             opp_live, opp_depth, opp_max, limit_strength, player_elo,
                             puzzle, puzzle_side)
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
            # The ENTIRE loop body is guarded: nothing here (a job, an engine death,
            # a stop()/configure() race) may ever kill the analysis thread, or the
            # overlay would silently stop updating for the rest of the session.
            try:
                self._wake.wait(0.5)
                self._wake.clear()
                if self._shutdown:
                    break
                with self._lock:
                    job, self._pending = self._pending, None
                    reconf, self._reconfigure = self._reconfigure, False
                    threads, hash_mb = self._threads, self._hash
                if reconf and self._engine is not None:
                    self._safe_configure({"Threads": threads, "Hash": hash_mb})
                if job is None:
                    continue
                # Respawn if the engine is absent OR a prior job caught it dead. The
                # latter is what lets the opponent (reds) path recover from an engine
                # death — its helpers swallow the error, so without this it would never
                # come back (see _engine_broken).
                if ((self._engine is None or self._engine_broken)
                        and not self._spawn_engine()):
                    self._wake.wait(1.0)      # engine down — back off, don't busy-loop
                    continue
                self._analyze(*job)
            except Exception as exc:
                self.failed.emit(str(exc))
                try:
                    self._spawn_engine()      # recover the engine after any error
                except Exception:
                    pass
        self._quit_engine()

    def _spawn_engine(self) -> bool:
        self._quit_engine()
        try:
            self._engine = chess.engine.SimpleEngine.popen_uci(self._path)
            self._safe_configure({"Threads": self._threads, "Hash": self._hash,
                                  **self._extra_options})
            self._detect_strength_support()
            self._applied_strength = None       # a fresh engine is full strength; force re-apply
            self._engine_broken = False          # a fresh process is healthy
            return True
        except Exception as exc:
            self.failed.emit(f"engine failed to start: {exc}")
            return False

    def _safe_configure(self, want: dict) -> None:
        """Configure only the options the engine actually exposes — so sending
        Stockfish's Hash to lc0 (which has no Hash) is silently skipped instead of
        erroring, and the same controller drives both."""
        if self._engine is None:
            return
        opts = getattr(self._engine, "options", {}) or {}
        cfg = {k: v for k, v in want.items() if k in opts}
        if cfg:
            try:
                self._engine.configure(cfg)
            except Exception:
                pass

    def _detect_strength_support(self) -> None:
        """Note whether the engine exposes UCI_LimitStrength/UCI_Elo and its range."""
        opts = getattr(self._engine, "options", {}) or {}
        self._strength_supported = "UCI_LimitStrength" in opts and "UCI_Elo" in opts
        if self._strength_supported:
            o = opts["UCI_Elo"]
            lo = int(getattr(o, "min", None) or 1320)
            hi = int(getattr(o, "max", None) or 3190)
            self._elo_range = (lo, hi)

    def _set_strength(self, limited: bool, elo: int) -> None:
        """Configure the engine's playing strength, caching to avoid redundant
        setoptions. Off => UCI_LimitStrength false (native full strength, so the
        default full-strength path is never altered). No-op if unsupported."""
        if not self._strength_supported:
            return
        want = (bool(limited), int(elo) if limited else 0)
        if want == self._applied_strength:
            return
        try:
            if limited:
                lo, hi = self._elo_range
                self._engine.configure(
                    {"UCI_LimitStrength": True, "UCI_Elo": max(lo, min(hi, int(elo)))})
            else:
                self._engine.configure({"UCI_LimitStrength": False})
            self._applied_strength = want
        except Exception:
            pass

    def _strength_full(self) -> None:
        """Full strength — for the OPPONENT prediction (reds), always."""
        self._set_strength(False, 0)

    def _strength_player(self) -> None:
        """The player's configured strength — for the PLAYER's eval (greens)."""
        self._set_strength(*self._strength)

    def _quit_engine(self) -> None:
        if self._engine is not None:
            try:
                self._engine.quit()      # ask the engine process to exit (blocks briefly)
            except Exception:
                pass
            try:
                self._engine.close()     # end python-chess's OWN background asyncio thread
            except Exception:            # (quit alone leaves it to end on pipe-EOF)
                pass
            self._engine = None

    def _top_moves(self, board: chess.Board, multipv: int, depth: int) -> list:
        """Ranked MoveSuggestions for ``board`` at ``depth`` — one look-ahead step.

        Streams to the depth limit (rather than a blocking analyse) so a new request
        can interrupt it promptly, which is what lets the engine collapse to the live
        analysis the instant the opponent actually moves. Returns the deepest lines
        reached (possibly shallower if interrupted). Used for the opponent's
        candidates (reds) and the player's look-ahead replies."""
        latest: dict[int, dict] = {}
        try:
            # python-chess parses PVs against this board in ITS OWN background thread;
            # hand it a private copy so that thread never shares a chess.Board with this
            # controller (concurrent push/pop corrupts the move stack -> segfault).
            with self._engine.analysis(board.copy(), multipv=multipv,
                                       limit=chess.engine.Limit(depth=depth)) as analysis:
                self._analysis = analysis
                for info in analysis:
                    if self._wake.is_set() or self._shutdown:
                        break
                    if info.get("pv"):
                        latest[info.get("multipv", 1)] = info
        except chess.engine.EngineTerminatedError:
            # The engine process died. Only the opponent/look-ahead path calls this, and
            # it swallows the error to stay alive — so flag a respawn (unless we're just
            # being interrupted), or the reds would never come back. See _engine_broken.
            if not self._aborting():
                self._engine_broken = True
            return []
        except Exception:
            return []
        finally:
            self._analysis = None
        out = []
        for rank, key in enumerate(sorted(latest), start=1):
            s = MoveSuggestion.from_info(latest[key], board, rank)
            if s:
                out.append(s)
        return out

    def _deepen_schedule(self, start: int, cap: int) -> list[int]:
        """Depths to refine a 'live' look-ahead over: ``start`` up to ``cap``
        (never below start), always landing exactly on the ceiling."""
        start = max(1, int(start))
        cap = max(start, int(cap))
        sched = list(range(start, cap + 1, self.LOOKAHEAD_STEP))
        if not sched or sched[-1] != cap:
            sched.append(cap)
        return sched

    def _analyze(self, board: chess.Board, multipv: int, mode: str, depth: int,
                 player_color: bool | None = None, token: int = 0,
                 opp_live: bool = False, opp_depth: int = 12, opp_max: int = 22,
                 limit_strength: bool = False, player_elo: int = 1500,
                 puzzle: bool = False, puzzle_side: bool | None = None) -> None:
        # Tempo model:
        #   * player to move  -> analyse the CURRENT position for the player.
        #   * opponent to move -> look ahead (see _analyze_opponent_turn): the
        #     opponent's top moves (reds) + the player's reply(ies) (greens), either
        #     a one-shot preview or refined live over increasing depth.
        # Game over (no legal moves) -> emit empty so the UI shows the result
        # cleanly instead of going silent on an unsearchable position.
        # The player's eval can be strength-limited (simulated Elo); the opponent
        # prediction always stays full strength. Applied per-analysis below.
        self._strength = (bool(limit_strength), int(player_elo))
        if puzzle:
            self._analyze_puzzle(board, multipv, depth, token, puzzle_side)
            return
        if board.legal_moves.count() == 0:
            self._emit_update([], 0, board, [], token)
            return

        opponent_to_move = (player_color is not None and board.is_valid()
                            and board.turn != player_color)
        if opponent_to_move:
            self._analyze_opponent_turn(board, multipv, mode, depth, opp_live,
                                        opp_depth, opp_max, token)
            return

        # Player to move: analyse the real position (streaming live / fixed depth).
        self._strength_player()
        self._stream_player(board, multipv, mode, depth, [], token)

    @staticmethod
    def _rel_value(s) -> float:
        """A single suggestion's value from the mover's POV (mate dominates)."""
        if s.mate_in is not None:
            return 1e6 - s.mate_in if s.mate_in > 0 else -1e6 - s.mate_in
        return float(s.score_cp) if s.score_cp is not None else 0.0

    @classmethod
    def _standout(cls, suggestions: list) -> float:
        """How much the best move STANDS OUT from the next-best, from the mover's POV.
        A puzzle's solution is uniquely good, so the side that holds the tactic shows a
        large standout while the other side (given a free move) has many near-equal
        options. This identifies the side to move ~92% of the time on real puzzles --
        far better than 'who is winning' (~60%), which fails because in sharp positions
        whoever moves first wins. Mate makes the standout huge automatically."""
        if not suggestions:
            return -1e18
        best = cls._rel_value(suggestions[0])
        second = cls._rel_value(suggestions[1]) if len(suggestions) > 1 else best - 1000.0
        return best - second

    def _aborting(self) -> bool:
        """A newer request arrived (or we're shutting down) — abandon this job."""
        return self._wake.is_set() or self._shutdown

    @staticmethod
    def _side_in_check(board: chess.Board, color: bool) -> bool:
        """Is ``color``'s king attacked? In a legal position only the side TO MOVE can
        be in check (you can't leave/​move your own king into check, and you can't be
        in check on the opponent's move), so a lone king-in-check pins the side to move
        with certainty — a free, deterministic tell that beats any heuristic."""
        king = board.king(color)
        return king is not None and board.is_attacked_by(not color, king)

    def _legal_side_board(self, board: chess.Board, color: bool) -> chess.Board | None:
        """A copy of ``board`` with ``color`` to move and an empty history — or None if
        that side is illegal to move (opponent left in check) or has no legal move."""
        b = board.copy()
        b.turn = color
        b.clear_stack()
        if not b.is_valid() or b.legal_moves.count() == 0:
            return None
        return b

    def _pick_side(self, board: chess.Board, white_sugg: list, black_sugg: list) -> bool:
        """Whose move it is in an isolated puzzle. A king in check is DETERMINISTIC
        (see _side_in_check), so it decides outright; otherwise the side whose best move
        stands out most from its alternatives holds the puzzle's uniquely-good tactic
        (see _standout — ~91% on real puzzles, where 'who is winning' is a coin flip).
        The side is then LOCKED for the rest of the puzzle and tracked by parity, so a
        cold miss self-corrects the moment a move is played (handled in the menu)."""
        wc, bc = self._side_in_check(board, chess.WHITE), self._side_in_check(board, chess.BLACK)
        if wc and not bc:
            return chess.WHITE
        if bc and not wc:
            return chess.BLACK
        if not white_sugg:
            return chess.BLACK
        if not black_sugg:
            return chess.WHITE
        return chess.WHITE if self._standout(white_sugg) >= self._standout(black_sugg) else chess.BLACK

    def _analyse_blocking(self, board: chess.Board, multipv: int, depth: int) -> list:
        """Ranked MoveSuggestions via a BLOCKING ``engine.analyse`` (runs to ``depth``
        and returns). Unlike the streaming ``_top_moves``, python-chess consumes the
        whole analysis internally and tears it down cleanly — there is no half-drained
        streaming analysis left for its background parser thread to keep working on.
        Puzzle mode uses this: its rapid both-sides + refine cycling, abruptly
        interrupted as you flick through puzzles, corrupted python-chess's async parser
        and segfaulted. Hands the engine a private board copy (see _top_moves)."""
        n = max(1, min(multipv, board.legal_moves.count()))
        try:
            infos = self._engine.analyse(board.copy(), chess.engine.Limit(depth=depth),
                                         multipv=n)
        except chess.engine.EngineTerminatedError:
            if not self._aborting():        # dead engine -> respawn next job (see _engine_broken)
                self._engine_broken = True
            return []
        except Exception:
            return []
        if isinstance(infos, dict):                  # may collapse to a single InfoDict
            infos = [infos]
        out = []
        for rank, info in enumerate(infos, start=1):
            s = MoveSuggestion.from_info(info, board, rank)
            if s:
                out.append(s)
        return out

    def _analyze_puzzle(self, board: chess.Board, multipv: int, depth: int, token: int,
                        forced_side: bool | None = None) -> None:
        """Puzzle mode: solve ONE side and STREAM its best line exactly like live mode — a
        shallow move appears in milliseconds and refines up to ``depth``, then stops (a
        solved puzzle is not re-evaluated). Its PV is the forced sequence the overlay draws.

        ``forced_side`` is the FAST, steady-state path: the menu determines the side once
        per puzzle and tracks it by parity (so it never flips mid-puzzle), so here we skip
        straight to streaming that side — no both-sides search, no wait. Only a brand-new
        puzzle arrives with ``forced_side=None``; then we pick the side ONCE via a quick
        shallow both-sides pass (check-constraint then best-move standout, see _pick_side).

        Streaming — NOT the old blocking preview+refine — is what makes this instant. It is
        a SINGLE streamed analysis, interrupted cleanly when the next request arrives: the
        exact live-mode pattern. (The segfault was the old per-request churn of three
        blocking/streamed analyses torn down mid-flight, not streaming itself — see
        [[crash-safety-and-v1]].) Always full strength; the opponent's forced replies are
        read straight from the PV, so there is no separate both-sides 'reds' pass."""
        self._strength_full()
        mpv = max(1, multipv)
        if forced_side is not None:
            # Known side (parity-tracked / highlight override): stream it straight away.
            target = (self._legal_side_board(board, bool(forced_side))
                      or self._legal_side_board(board, not forced_side))
        else:
            target = self._pick_target(board, mpv)      # fresh puzzle: quick one-time pick
        if target is None:
            self._emit_update([], 0, board, [], token)
            return
        # Fixed-depth stream: instant shallow move, refines to `depth`, then ends (idle —
        # no re-evaluation of a static solved position). Interrupted on the next request.
        self._stream_player(target, mpv, "fixed", depth, [], token)

    def _pick_target(self, board: chess.Board, mpv: int) -> chess.Board | None:
        """Fresh puzzle: pick the side (check-constraint + standout) from a quick, shallow
        both-sides pass and return that side's board to stream. Cheap — runs once per
        puzzle, never in the steady state (the side is parity-tracked after)."""
        best = {}
        for color in (chess.WHITE, chess.BLACK):
            if self._aborting():
                return None
            sb = self._legal_side_board(board, color)
            best[color] = self._analyse_blocking(sb, max(2, mpv), self.PUZZLE_PICK_DEPTH) if sb else []
        if not best[chess.WHITE] and not best[chess.BLACK]:
            return None
        return self._legal_side_board(
            board, self._pick_side(board, best[chess.WHITE], best[chess.BLACK]))

    def _stream_player(self, target_board: chess.Board, multipv: int, mode: str,
                       depth: int, opp_suggestions: list, token: int,
                       min_emit_depth: int = -1) -> None:
        """Stream a multipv analysis of ``target_board`` (player to move) and emit as
        it refines. ``opp_suggestions`` ride along to the UI unchanged. Emits only
        once the coherent depth exceeds ``min_emit_depth`` (so a deep stream layered
        after a refinement loop doesn't briefly regress to shallow lines)."""
        if target_board.legal_moves.count() == 0:
            self._emit_update([], 0, target_board, opp_suggestions, token)
            return
        n = min(multipv, target_board.legal_moves.count())
        limit = chess.engine.Limit(depth=depth) if mode == "fixed" else None
        latest: dict[int, dict] = {}
        last_emit = min_emit_depth
        # private copy for python-chess's background PV parser (see _top_moves).
        with self._engine.analysis(target_board.copy(), multipv=multipv, limit=limit) as analysis:
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
                if coherent > last_emit:
                    self._emit(latest, coherent, target_board, opp_suggestions, token)
        self._analysis = None

    def _analyze_opponent_turn(self, board: chess.Board, multipv: int, mode: str,
                               depth: int, opp_live: bool, opp_depth: int,
                               opp_max: int, token: int) -> None:
        """Opponent to move. Predictive splits off to its own handler. Live/fixed
        show the opponent's top moves (reds) and the player's responses (greens) to
        the opponent's best move. With ``opp_live`` (live mode only) the candidates
        and responses refine together over increasing depth, then the responses keep
        streaming deep on the settled line; otherwise the candidates are a one-shot
        preview and the responses stream/fix as before (unchanged default)."""
        if mode == "predictive":
            self._predictive_turn(board, multipv, opp_live, opp_depth, opp_max, token)
            return

        if opp_live and mode == "live":
            schedule = self._deepen_schedule(opp_depth, opp_max)
            opp = []
            for d in schedule:
                if self._wake.is_set() or self._shutdown:
                    return
                self._strength_full()                      # opponent prediction: full strength
                opp = self._top_moves(board, multipv, d)
                if not opp:
                    self._emit_update([], 0, board, [], token)
                    return
                target = board.copy()
                target.push(opp[0].move)
                if target.legal_moves.count() == 0:        # opponent's best ends the game
                    self._emit_update([], 0, target, opp, token)
                    return
                if self._wake.is_set() or self._shutdown:
                    return
                self._strength_player()                    # player's responses: limited
                responses = self._top_moves(target, multipv, d)
                self._emit_update(responses, d, target, opp, token)
            # Candidates settled at the ceiling; now refine the responses deeply on
            # that line (only emitting past the ceiling, so they don't regress).
            if not (self._wake.is_set() or self._shutdown) and opp:
                target = board.copy()
                target.push(opp[0].move)
                self._strength_player()
                self._stream_player(target, multipv, mode, depth, opp, token,
                                    min_emit_depth=schedule[-1])
            return

        # Default: one-shot opponent candidates (fast preview), responses streamed
        # (live) or fixed-depth — exactly the prior behaviour.
        self._strength_full()                              # opponent prediction: full strength
        opp = self._top_moves(board, multipv, opp_depth)
        target = board.copy()
        if opp:
            target.push(opp[0].move)
        self._strength_player()                            # player's responses: limited
        self._stream_player(target, multipv, mode, depth, opp, token)

    def _predictive_turn(self, board: chess.Board, multipv: int, opp_live: bool,
                         opp_depth: int, opp_max: int, token: int) -> None:
        """Predictive, opponent to move: the opponent's top moves (reds) plus the
        player's single best reply to EACH of them (greens). The replies ALWAYS
        refine over increasing depth (live) — even when the opponent candidates are
        a one-shot preview; with ``opp_live`` the candidates refine too. Collapses to
        the normal live flow once the opponent actually moves."""
        fixed_opp = None
        if not opp_live:
            self._strength_full()                                    # opponent candidates: full strength
            fixed_opp = self._top_moves(board, multipv, opp_depth)   # fast preview, then deepen replies
            if not fixed_opp:
                self._emit_update([], 0, board, [], token)
                return
        for d in self._deepen_schedule(opp_depth, opp_max):
            if self._wake.is_set() or self._shutdown:
                return
            if fixed_opp is not None:
                opp = fixed_opp
            else:
                self._strength_full()                                # opponent candidates: full strength
                opp = self._top_moves(board, multipv, d)
            if not opp:
                self._emit_update([], 0, board, [], token)
                return
            self._strength_player()                                  # player's replies: limited
            responses = self._predictive_replies(board, opp, d)
            if responses is None:                 # interrupted mid-round
                return
            target = board.copy()
            target.push(opp[0].move)              # turn == player: correct eval POV downstream
            self._emit_update(responses, d, target, opp, token)

    def _predictive_replies(self, board: chess.Board, opp_suggestions: list,
                            depth: int) -> list | None:
        """One best reply per opponent candidate, at ``depth``. Reply ``rank`` is
        paired with its opponent move (reply #1 answers the opponent's best move).
        Returns None if interrupted, so the caller abandons the round."""
        responses = []
        for rank, opp in enumerate(opp_suggestions, start=1):
            if self._wake.is_set() or self._shutdown:
                return None
            reply_board = board.copy()
            reply_board.push(opp.move)
            if reply_board.legal_moves.count() == 0:       # this opponent move ends the game
                continue
            best = self._top_moves(reply_board, 1, depth)
            if best:
                best[0].rank = rank
                responses.append(best[0])
        return responses

    def _emit_update(self, suggestions, depth, board, opp_suggestions, token) -> None:
        """Emit to the GUI thread with a COPY of the board (position only). The worker
        keeps using ITS board right after this returns — handing it to the engine and
        pushing moves — while the GUI calls board.san()/push() on what it receives;
        sharing one chess.Board across the two threads lets their push/pop race and
        corrupt its internal stack, which segfaults. The GUI needs the placement and
        side to move, not the history, so stack=False keeps the copy cheap."""
        safe = board.copy(stack=False) if board is not None else board
        self.updated.emit(suggestions, depth, safe, opp_suggestions, token)

    def _emit(self, latest, depth, board, opp_suggestions, token) -> None:
        out = []
        for rank, key in enumerate(sorted(latest), start=1):
            s = MoveSuggestion.from_info(latest[key], board, rank)
            if s:
                out.append(s)
        if out:
            self._emit_update(out, depth, board, opp_suggestions, token)
