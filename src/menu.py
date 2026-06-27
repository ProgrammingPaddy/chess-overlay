"""Options menu / control panel — the application hub (tabbed).

Owns the overlay, the persistent engine controller, screen capture, vision, and
the game tracker. Board + vision calibration are in memory only (see config.py).
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes
import os
from pathlib import Path

import chess
from PySide6 import QtCore, QtGui, QtWidgets

from src.analysis import EngineController
from src.calibrate import auto_calibrate, calibrate
from src.capture import ScreenCapture, save_image
from src.config import Config
from src.consensus import ConsensusBuffer
from src.engine import find_stockfish
from src.overlay import (Annotation, BoardGeometry, OverlayManager,
                         build_annotations, visible_annotations)
from src.tracker import GameTracker
from src.vision import VisionModel, certainty, dump_calibration, dump_recognition
from src.vision_worker import VisionWorker

ROOT = Path(__file__).resolve().parent.parent
DEBUG_DIR = ROOT / "debug"
TICK_MS = 50             # background vision interval (ms) — fast, off the GUI thread
CONSENSUS_WINDOW = 3     # frames voted per square for a stable reading
AGREEMENT_MIN = 0.55     # min cross-frame agreement before a read is acted on
FRAME_MIN = 0.45         # min per-frame match quality to treat it as a real board
NO_BOARD_CERT = 0.30     # sustained reads below this => clearly no board (clear arrows)
NO_BOARD_FRAMES = 4
RESYNC_CONFIRM = 2       # stable non-legal reads before resyncing (filters drags)


def _left_mouse_down() -> bool:
    """Is the physical left mouse button currently held (anywhere)?"""
    try:
        return bool(ctypes.windll.user32.GetAsyncKeyState(0x01) & 0x8000)
    except Exception:
        return False


def _cursor_xy() -> tuple[int, int]:
    pt = ctypes.wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return int(pt.x), int(pt.y)


_GLYPHS = {"k": "♚", "q": "♛", "r": "♜",
           "b": "♝", "n": "♞", "p": "♟"}


class MiniBoard(QtWidgets.QWidget):
    """A small live render of the position the CV currently believes it sees."""

    def __init__(self):
        super().__init__()
        self._board = chess.Board.empty()
        self._white_bottom = True
        self.setMinimumSize(220, 220)

    def set_board(self, board: chess.Board, white_bottom: bool = True) -> None:
        self._board = board
        self._white_bottom = white_bottom
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        s = min(self.width(), self.height()) / 8.0
        light, dark = QtGui.QColor(240, 217, 181), QtGui.QColor(181, 136, 99)
        for r in range(8):
            for c in range(8):
                p.fillRect(QtCore.QRectF(c * s, r * s, s, s),
                           light if (r + c) % 2 == 0 else dark)
        font = p.font()
        font.setPixelSize(int(s * 0.82))
        p.setFont(font)
        for r in range(8):
            for c in range(8):
                sq = chess.square(c, 7 - r) if self._white_bottom else chess.square(7 - c, r)
                pc = self._board.piece_at(sq)
                if pc is None:
                    continue
                glyph = _GLYPHS[pc.symbol().lower()]
                rect = QtCore.QRectF(c * s, r * s, s, s)
                if pc.color == chess.WHITE:           # outline white pieces for contrast
                    p.setPen(QtGui.QColor(30, 30, 30))
                    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        p.drawText(rect.translated(dx, dy), QtCore.Qt.AlignCenter, glyph)
                    p.setPen(QtGui.QColor(250, 250, 250))
                else:
                    p.setPen(QtGui.QColor(20, 20, 20))
                p.drawText(rect, QtCore.Qt.AlignCenter, glyph)


class MenuWindow(QtWidgets.QWidget):
    def __init__(self, app: QtWidgets.QApplication):
        super().__init__()
        self._app = app
        self.cfg = Config.load()
        self._capture: ScreenCapture | None = None
        self._controller: EngineController | None = None
        self._eng_sig = (self.cfg.engine_threads, self.cfg.engine_hash_mb)
        self._geometry: BoardGeometry | None = None
        self._cap_region: tuple[int, int, int, int] | None = None
        self._loading = True

        self.overlay = OverlayManager(app)
        self.tracker = GameTracker()
        self.vision = VisionModel()
        self._resync_fen: str | None = None
        self._resync_count = 0
        self._cert_ema = 0.0
        self._no_board = 0
        self._analyzing_key: str | None = None     # placement+turn currently sent to the engine
        self._req_id = 0                            # token for the latest engine request
        self._suggestions: list[Annotation] = []    # latest engine arrows (pre-filter)
        self._believed: chess.Board | None = None    # what the CV currently sees
        self._consensus = ConsensusBuffer(CONSENSUS_WINDOW)

        self._vision_worker: VisionWorker | None = None

        self.setWindowTitle("Chess Overlay")
        self.setMinimumWidth(420)
        self._build_ui()

        self.overlay.show()
        self.overlay.set_overlay_visible(self.cfg.show_arrows)
        self.overlay.set_show_border(self.cfg.show_border)
        self._refresh_board_status()
        self._refresh_vision_status()
        self._init_controller()
        self._reconcile_orientation_controls()
        self._loading = False

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        tabs = QtWidgets.QTabWidget()
        root.addWidget(tabs)
        tabs.addTab(self._tab_setup(), "Setup")
        tabs.addTab(self._tab_vision(), "Vision")
        tabs.addTab(self._tab_play(), "Play")

        self.status_label = QtWidgets.QLabel("Ready. Calibrate the board, then vision.")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)
        self._connect_signals()

    def _tab_setup(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        board = QtWidgets.QGroupBox("Board")
        bf = QtWidgets.QFormLayout(board)
        self.monitor_combo = QtWidgets.QComboBox()
        for i, s in enumerate(self._app.screens()):
            g = s.geometry()
            self.monitor_combo.addItem(
                f"[{i}] {s.name()} {g.width()}x{g.height()} @{s.devicePixelRatio():g}x", i)
        self.monitor_combo.setCurrentIndex(
            max(0, min(self.cfg.board_monitor, self.monitor_combo.count() - 1)))
        bf.addRow("Monitor", self.monitor_combo)
        self.calibrate_btn = QtWidgets.QPushButton("Calibrate board — manual (drag a box)…")
        self.auto_calibrate_btn = QtWidgets.QPushButton("Calibrate board — auto-align (rough box)…")
        bf.addRow(self.calibrate_btn)
        bf.addRow(self.auto_calibrate_btn)
        self.board_status = QtWidgets.QLabel()
        self.board_status.setWordWrap(True)
        bf.addRow(self.board_status)
        self.show_border_cb = QtWidgets.QCheckBox("Show calibrated board border")
        self.show_border_cb.setChecked(self.cfg.show_border)
        self.show_arrows_cb = QtWidgets.QCheckBox("Show arrows on board")
        self.show_arrows_cb.setChecked(self.cfg.show_arrows)
        self.gold_moves_cb = QtWidgets.QCheckBox("Highlight a clearly-best / mate move in gold")
        self.gold_moves_cb.setChecked(self.cfg.gold_moves)
        self.white_bottom_cb = QtWidgets.QCheckBox("White on bottom")
        self.white_bottom_cb.setChecked(self.cfg.white_bottom)
        bf.addRow(self.show_border_cb)
        bf.addRow(self.show_arrows_cb)
        bf.addRow(self.gold_moves_cb)
        bf.addRow(self.white_bottom_cb)
        v.addWidget(board)

        eng = QtWidgets.QGroupBox("Engine")
        ef = QtWidgets.QFormLayout(eng)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem("Live (instant, refines)", "live")
        self.mode_combo.addItem("Fixed depth (strong)", "fixed")
        self.mode_combo.addItem("Predictive (a reply to each likely move)", "predictive")
        self.mode_combo.setCurrentIndex(max(0, self.mode_combo.findData(self.cfg.engine_mode)))
        ef.addRow("Mode", self.mode_combo)
        self.depth_spin = self._spin(1, 60, self.cfg.engine_depth)
        self.lines_spin = self._spin(1, 5, self.cfg.multipv)
        self.threads_spin = self._spin(1, max(1, os.cpu_count() or 1), self.cfg.engine_threads)
        self.hash_spin = self._spin(16, 8192, self.cfg.engine_hash_mb, step=64)
        ef.addRow("Depth (fixed)", self.depth_spin)
        ef.addRow("Lines", self.lines_spin)
        # Opponent look-ahead (live & predictive modes): a fast one-shot preview by
        # default, or refine it live from the preview depth up to the ceiling.
        self.opp_live_cb = QtWidgets.QCheckBox("Refine opponent look-ahead live (deepen over time)")
        self.opp_live_cb.setChecked(self.cfg.opp_lookahead_live)
        self.opp_depth_spin = self._spin(2, 40, self.cfg.opp_lookahead_depth)
        self.opp_max_spin = self._spin(2, 60, self.cfg.opp_lookahead_max)
        ef.addRow("Opponent look-ahead", self.opp_live_cb)
        ef.addRow("Opp preview depth", self.opp_depth_spin)
        ef.addRow("Opp refine ceiling", self.opp_max_spin)
        ef.addRow("Threads", self.threads_spin)
        ef.addRow("Hash (MB)", self.hash_spin)
        self.engine_status = QtWidgets.QLabel("Engine: starting…")
        self.engine_status.setWordWrap(True)
        ef.addRow(self.engine_status)
        v.addWidget(eng)
        v.addStretch(1)
        return w

    def _tab_vision(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        row = QtWidgets.QHBoxLayout()
        self.calib_vision_btn = QtWidgets.QPushButton("Calibrate vision (start position)")
        self.recognize_btn = QtWidgets.QPushButton("Recognize now")
        row.addWidget(self.calib_vision_btn)
        row.addWidget(self.recognize_btn)
        v.addLayout(row)
        self.allow_illegal_cb = QtWidgets.QCheckBox(
            "Allow illegal moves (accept any read, skip legality check)")
        self.allow_illegal_cb.setChecked(self.cfg.allow_illegal)
        v.addWidget(self.allow_illegal_cb)
        self.certainty_bar = QtWidgets.QProgressBar()
        self.certainty_bar.setRange(0, 100)
        self.certainty_bar.setFormat("Read certainty: %p%")
        v.addWidget(self.certainty_bar)
        self.vision_status = QtWidgets.QLabel()
        self.vision_status.setWordWrap(True)
        v.addWidget(self.vision_status)
        v.addWidget(QtWidgets.QLabel("What the CV sees:"))
        self.mini_board = MiniBoard()
        v.addWidget(self.mini_board, 1)
        return w

    def _tab_play(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        top = QtWidgets.QFormLayout()
        self.side_combo = QtWidgets.QComboBox()
        for text, data in (("Auto (side on bottom)", "auto"), ("White", "white"), ("Black", "black")):
            self.side_combo.addItem(text, data)
        self.side_combo.setCurrentIndex(max(0, self.side_combo.findData(self.cfg.analyze_for)))
        top.addRow("I play as", self.side_combo)
        self.track_cb = QtWidgets.QCheckBox("Auto-track game (live board → engine)")
        self.track_cb.setChecked(self.cfg.auto_track)
        top.addRow(self.track_cb)
        self.predict_cb = QtWidgets.QCheckBox("Show opponent's likely moves (red) on their turn")
        self.predict_cb.setChecked(self.cfg.show_predicted)
        top.addRow(self.predict_cb)
        self.pause_drag_cb = QtWidgets.QCheckBox("Pause eval while dragging a piece")
        self.pause_drag_cb.setChecked(self.cfg.pause_on_drag)
        top.addRow(self.pause_drag_cb)
        self.turn_label = QtWidgets.QLabel("To move: —")
        self.flip_turn_btn = QtWidgets.QPushButton("Flip side to move")
        top.addRow(self.turn_label, self.flip_turn_btn)
        v.addLayout(top)

        self.fen_edit = QtWidgets.QLineEdit(chess.STARTING_FEN)
        v.addWidget(self.fen_edit)
        row = QtWidgets.QHBoxLayout()
        self.analyze_btn = QtWidgets.QPushButton("Analyze FEN")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.newgame_btn = QtWidgets.QPushButton("New game")
        self.reset_game_btn = QtWidgets.QPushButton("Re-read board")
        self.snapshot_btn = QtWidgets.QPushButton("Snapshot")
        for b in (self.analyze_btn, self.stop_btn, self.newgame_btn,
                  self.reset_game_btn, self.snapshot_btn):
            row.addWidget(b)
        v.addLayout(row)
        self.results_list = QtWidgets.QListWidget()
        self.results_list.setFixedHeight(84)
        v.addWidget(self.results_list)
        self.opening_label = QtWidgets.QLabel("Opening: —")
        v.addWidget(self.opening_label)
        self.moves_view = QtWidgets.QPlainTextEdit()
        self.moves_view.setReadOnly(True)
        self.moves_view.setFixedHeight(64)
        v.addWidget(self.moves_view)
        return w

    @staticmethod
    def _spin(lo, hi, val, step=1) -> QtWidgets.QSpinBox:
        s = QtWidgets.QSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setValue(val)
        return s

    def _connect_signals(self) -> None:
        self.calibrate_btn.clicked.connect(self._on_calibrate)
        self.auto_calibrate_btn.clicked.connect(self._on_auto_calibrate)
        self.analyze_btn.clicked.connect(self._on_analyze)
        self.stop_btn.clicked.connect(self._on_stop)
        self.snapshot_btn.clicked.connect(self._on_snapshot)
        self.calib_vision_btn.clicked.connect(self._on_calibrate_vision)
        self.recognize_btn.clicked.connect(self._on_recognize)
        self.track_cb.toggled.connect(self._on_track_toggled)
        self.reset_game_btn.clicked.connect(self._on_reset_game)
        self.newgame_btn.clicked.connect(self._on_new_game)
        self.flip_turn_btn.clicked.connect(self._on_flip_turn)
        # Orientation == player colour (side on bottom is the player), so the
        # 'White on bottom' checkbox and the 'I play as' combo get dedicated
        # handlers that flip the board AND re-run analysis immediately.
        self.white_bottom_cb.toggled.connect(self._on_white_bottom_toggled)
        self.side_combo.currentIndexChanged.connect(self._on_side_changed)
        for wdg in (self.show_arrows_cb, self.gold_moves_cb, self.show_border_cb,
                    self.allow_illegal_cb, self.predict_cb, self.pause_drag_cb,
                    self.opp_live_cb):
            wdg.toggled.connect(self._on_settings_changed)
        for cb in (self.monitor_combo, self.mode_combo):
            cb.currentIndexChanged.connect(self._on_settings_changed)
        for sp in (self.depth_spin, self.lines_spin, self.threads_spin, self.hash_spin,
                   self.opp_depth_spin, self.opp_max_spin):
            sp.valueChanged.connect(self._on_settings_changed)

    # ---------------------------------------------------------------- helpers
    def _monitor_index(self) -> int:
        data = self.monitor_combo.currentData()
        return int(data) if data is not None else 0

    def _ensure_capture(self) -> ScreenCapture:
        if self._capture is None:
            self._capture = ScreenCapture()
        return self._capture

    def _grab_board(self):
        if self._cap_region is None:
            return None
        return self._ensure_capture().grab(*self._cap_region)

    def _init_controller(self) -> None:
        path = self.cfg.engine_path or find_stockfish()
        if not path:
            self.engine_status.setText("Engine: Stockfish not found (put it in engines/).")
            return
        c = EngineController(path, self.cfg.engine_threads, self.cfg.engine_hash_mb)
        c.updated.connect(self._on_analysis_updated)
        c.failed.connect(self._on_analysis_failed)
        c.ready.connect(lambda: self.engine_status.setText(f"Engine: {Path(path).name} ready."))
        c.start()
        self._controller = c

    def _start_analysis(self, board: chess.Board) -> None:
        if not board.is_valid():
            kings_ok = (board.king(chess.WHITE) is not None
                        and board.king(chess.BLACK) is not None)
            if not (self.cfg.allow_illegal and kings_ok):
                # Can't analyse (illegal read). Keep the last arrows (they're
                # filtered to the live board) instead of flickering them off, and
                # leave it un-analysed so a valid read re-triggers analysis.
                self._analyzing_key = None
                self.status_label.setText("Position not analysable yet — showing last moves.")
                return
            # Allow-illegal + both kings present: try anyway; the controller
            # auto-respawns the engine if Stockfish rejects the position.
        if self._controller is None:
            return
        self._analyzing_key = self._pos_key(board)
        self._req_id += 1
        self._controller.request(board, self.cfg.multipv, self.cfg.engine_mode,
                                 self.cfg.engine_depth, self._player_color(), self._req_id,
                                 self.cfg.opp_lookahead_live, self.cfg.opp_lookahead_depth,
                                 self.cfg.opp_lookahead_max)
        self.stop_btn.setEnabled(True)

    def _player_color(self) -> bool:
        """Which colour the human plays. The contract is 'side on bottom is the
        player', so this is derived from the board orientation alone — a single
        source of truth shared with vision and the overlay. The 'I play as' control
        sets the orientation (see ``_on_side_changed``), so they can never disagree
        and the look-ahead always knows whose turn it is."""
        return chess.WHITE if self.cfg.white_bottom else chess.BLACK

    def _stop_analysis(self) -> None:
        if self._controller is not None:
            self._controller.clear()
        self.stop_btn.setEnabled(False)

    def _reanalyze_current(self) -> None:
        """Force a re-analysis of the current position. Used when something that
        changes the engine output or the player/opponent split (orientation, side,
        lines, gold, reds) changes — the position is the same, so the normal
        'unchanged position' guard would otherwise suppress the update."""
        if self._controller is None:
            return
        board = self.tracker.board
        if board is None:
            return
        self._analyzing_key = None        # defeat the unchanged-position guard
        self._start_analysis(board)

    # ----------------------------------------------------- orientation / side
    def _sync_white_bottom_cb(self) -> None:
        self.white_bottom_cb.blockSignals(True)
        self.white_bottom_cb.setChecked(self.cfg.white_bottom)
        self.white_bottom_cb.blockSignals(False)

    def _sync_side_combo(self) -> None:
        self.side_combo.blockSignals(True)
        self.side_combo.setCurrentIndex(max(0, self.side_combo.findData(self.cfg.analyze_for)))
        self.side_combo.blockSignals(False)

    def _apply_orientation(self, white_bottom: bool) -> None:
        """Adopt a board orientation. 'Side on bottom is the player', so this also
        decides which colour the engine analyses for. Re-runs analysis so the
        green/red (player/opponent) split flips immediately, not on the next move."""
        self.cfg.white_bottom = white_bottom
        self._sync_white_bottom_cb()
        self.cfg.save()
        if self._geometry is not None:
            self._geometry.white_bottom = white_bottom
            self.overlay.set_board_geometry(self._geometry)
        if self._believed is not None:
            self.mini_board.set_board(self._believed, white_bottom)
        self._refresh_turn_label()
        self._reanalyze_current()

    def _on_side_changed(self, *_) -> None:
        """'I play as' — Auto follows the calibrated orientation; White/Black force
        it (and so flip the board), keeping a single source of truth."""
        if self._loading:
            return
        data = self.side_combo.currentData()
        self.cfg.analyze_for = data
        if data == "white":
            wb = True
        elif data == "black":
            wb = False
        else:                              # auto: the side calibration found on the bottom
            wb = self.vision.white_bottom if self.vision.calibrated else self.cfg.white_bottom
        self._apply_orientation(wb)

    def _on_white_bottom_toggled(self, on: bool) -> None:
        """Manually flipping the board is an explicit colour choice (you are the
        side on the bottom), so keep 'I play as' in step with it."""
        if self._loading:
            return
        self.cfg.analyze_for = "white" if on else "black"
        self._sync_side_combo()
        self._apply_orientation(on)

    def _reconcile_orientation_controls(self) -> None:
        """At startup make orientation and 'I play as' agree (side on bottom is the
        player) — a config from before this contract could have them disagree."""
        if self.cfg.analyze_for == "white":
            self.cfg.white_bottom = True
        elif self.cfg.analyze_for == "black":
            self.cfg.white_bottom = False
        self._sync_white_bottom_cb()
        self._sync_side_combo()

    def _refresh_board_status(self) -> None:
        if self._cap_region is None:
            self.board_status.setText("Board: not calibrated.")
        else:
            self.board_status.setText(
                f"Board: monitor [{self.cfg.board_monitor}], {self._cap_region[2]}px (physical).")

    def _refresh_vision_status(self) -> None:
        if self.vision.calibrated:
            side = "White" if self.vision.white_bottom else "Black"
            self.vision_status.setText(
                f"Vision: calibrated ({len(self.vision.sprites)} pieces, "
                f"{side} on bottom).")
        else:
            self.vision_status.setText(
                "Vision: not calibrated. Show the START position, then 'Calibrate vision'.")

    # ---------------------------------------------------------------- slots
    def _on_analysis_updated(self, suggestions: list, depth: int, board: chess.Board,
                             opp_suggestions: list, token: int) -> None:
        if token != self._req_id:        # stale emit from a superseded request
            return
        try:
            self.results_list.clear()
            flip = board.turn != chess.WHITE     # show eval ABSOLUTE (white +, black -)
            for s in suggestions:
                try:
                    san = board.san(s.move)
                except Exception:
                    san = s.uci
                self.results_list.addItem(f"#{s.rank}  {san:7s} {s.eval_text_pov(flip)}")
            self._suggestions = build_annotations(suggestions, opp_suggestions, self.cfg.show_predicted,
                                                  board.turn == chess.WHITE, self.cfg.gold_moves)
            self._draw_arrows()
            if not suggestions:
                self.status_label.setText(self._terminal_status(board, opp_suggestions))
            else:
                note = ""
                if opp_suggestions:
                    if self.cfg.engine_mode == "predictive":
                        note = f"  (a reply to each of their {len(opp_suggestions)} likely moves)"
                    else:
                        try:
                            opp_san = self.tracker.board.san(opp_suggestions[0].move)
                        except Exception:
                            opp_san = opp_suggestions[0].move.uci()
                        note = f"  (prep vs their best: {opp_san})"
                self.status_label.setText(f"Depth {depth}, {len(suggestions)} line(s).{note}")
        except Exception as exc:
            self.status_label.setText(f"render error: {exc}")

    @staticmethod
    def _terminal_status(board: chess.Board, opp_suggestions: list) -> str:
        """Status when there are no player moves to show — a real game over vs a
        predicted opponent finish (so the engine never just goes silent)."""
        if opp_suggestions:                     # look-ahead: their best move ends it
            mv = opp_suggestions[0].move.uci()
            return f"Opponent's {mv} would finish the game."
        if board.is_checkmate():
            return f"Checkmate — {'Black' if board.turn else 'White'} wins."
        if board.is_stalemate():
            return "Stalemate — draw."
        if board.is_game_over(claim_draw=True):
            return f"Game over — {board.result(claim_draw=True)}."
        return "No analysis (position not searchable)."

    def _on_analysis_failed(self, message: str) -> None:
        self.status_label.setText(f"Engine recovered after error: {message}")

    def _on_settings_changed(self, *_) -> None:
        if self._loading:
            return
        # Orientation / player colour are owned by the dedicated handlers
        # (_on_side_changed / _on_white_bottom_toggled), not read here.
        before = (self.cfg.multipv, self.cfg.engine_depth, self.cfg.engine_mode,
                  self.cfg.gold_moves, self.cfg.show_predicted, self.cfg.opp_lookahead_live,
                  self.cfg.opp_lookahead_depth, self.cfg.opp_lookahead_max)
        self.cfg.engine_depth = self.depth_spin.value()
        self.cfg.multipv = self.lines_spin.value()
        self.cfg.engine_threads = self.threads_spin.value()
        self.cfg.engine_hash_mb = self.hash_spin.value()
        self.cfg.engine_mode = self.mode_combo.currentData()
        self.cfg.board_monitor = self._monitor_index()
        self.cfg.show_arrows = self.show_arrows_cb.isChecked()
        self.cfg.gold_moves = self.gold_moves_cb.isChecked()
        self.cfg.show_border = self.show_border_cb.isChecked()
        self.cfg.allow_illegal = self.allow_illegal_cb.isChecked()
        self.cfg.show_predicted = self.predict_cb.isChecked()
        self.cfg.pause_on_drag = self.pause_drag_cb.isChecked()
        self.cfg.opp_lookahead_live = self.opp_live_cb.isChecked()
        self.cfg.opp_lookahead_depth = self.opp_depth_spin.value()
        self.cfg.opp_lookahead_max = self.opp_max_spin.value()
        self.cfg.save()
        self.overlay.set_overlay_visible(self.cfg.show_arrows)
        self.overlay.set_show_border(self.cfg.show_border)
        sig = (self.cfg.engine_threads, self.cfg.engine_hash_mb)
        if self._controller is not None and sig != self._eng_sig:
            self._eng_sig = sig
            self._controller.reconfigure(*sig)
        # A change that affects the engine output or the arrows (lines, depth,
        # mode, gold, opponent-reds, opponent look-ahead) re-runs analysis so the
        # change takes effect now instead of only on the next move.
        after = (self.cfg.multipv, self.cfg.engine_depth, self.cfg.engine_mode,
                 self.cfg.gold_moves, self.cfg.show_predicted, self.cfg.opp_lookahead_live,
                 self.cfg.opp_lookahead_depth, self.cfg.opp_lookahead_max)
        if after != before:
            self._reanalyze_current()

    def _on_calibrate(self) -> None:
        self.hide()
        QtWidgets.QApplication.processEvents()
        result = calibrate(self._app, self._monitor_index(), self.cfg.white_bottom)
        self.show()
        self.raise_()
        if result is None:
            self.status_label.setText("Calibration cancelled.")
            return
        self._apply_calibration(result, "manual")

    def _on_auto_calibrate(self) -> None:
        self.hide()
        QtWidgets.QApplication.processEvents()
        result = auto_calibrate(self._app, self._monitor_index(), self.cfg.white_bottom)
        self.show()
        self.raise_()
        if result is None:
            self.status_label.setText(
                "Auto-align: no board found — drag a tighter box around just the board, "
                "or use manual calibration.")
            return
        self._apply_calibration(result, "auto-aligned")
        if not self.cfg.show_border:                 # reveal what it found so you can verify
            self.show_border_cb.setChecked(True)

    def _apply_calibration(self, result, how: str) -> None:
        """Adopt a calibration result (manual or auto) — identical downstream state."""
        self.cfg.board_monitor = result.screen_index
        self.cfg.save()
        self._geometry = result.geometry
        self._cap_region = (result.phys_left, result.phys_top, result.phys_side, result.phys_side)
        self.overlay.set_board_geometry(self._geometry)
        self._refresh_board_status()
        # A (re)calibration is a clean slate: this may be a different board, so
        # drop the old piece templates, tracking state, and arrows.
        self._stop_tracking()
        self._set_track_checkbox(False)
        self.vision = VisionModel()
        self.tracker.reset()
        self._reset_tracking_state()
        self.overlay.clear()
        self.results_list.clear()
        self._refresh_moves()
        self._refresh_vision_status()
        self._on_snapshot()
        self.status_label.setText(
            f"Board calibrated ({how}, {result.phys_side}px). Now Calibrate vision for THIS board.")

    def _on_analyze(self) -> None:
        try:
            board = chess.Board(self.fen_edit.text().strip())
        except ValueError:
            self.status_label.setText("Invalid FEN.")
            return
        # Manual analysis: pause auto-track so the typed position isn't instantly
        # overwritten by the live board.
        if self._vision_worker is not None:
            self._stop_tracking()
            self._set_track_checkbox(False)
            self.cfg.auto_track = False
            self.cfg.save()
        self._believed = board
        self._start_analysis(board)
        self.status_label.setText("Analyzing the typed FEN (auto-track paused).")

    def _on_stop(self) -> None:
        self._stop_analysis()
        self.status_label.setText("Stopped.")

    def _on_snapshot(self) -> None:
        img = self._grab_board()
        if img is None:
            self.status_label.setText("Calibrate the board first.")
            return
        DEBUG_DIR.mkdir(exist_ok=True)
        save_image(DEBUG_DIR / "board.png", img)
        self.status_label.setText(f"Saved debug/board.png ({img.shape[1]}x{img.shape[0]}).")

    def _on_calibrate_vision(self) -> None:
        img = self._grab_board()
        if img is None:
            self.status_label.setText("Calibrate the board first.")
            return
        self._stop_tracking()        # don't write the model while the worker reads it
        # Calibration reads the true orientation from the board pixels (the white
        # army is brighter). Adopt it so a stale 'White on bottom' setting can't
        # 180-rotate the board (which inverts colours and swaps K<->Q). This is the
        # auto path, so reflect the detected side in BOTH controls (side on bottom
        # is the player) — tracking restarts right after and re-runs the analysis.
        warning = self.vision.calibrate(img, self.cfg.white_bottom)
        detected = self.vision.white_bottom
        note = ""
        if detected != self.cfg.white_bottom:
            note = f" (auto-detected {'White' if detected else 'Black'} on the bottom)"
        self.cfg.analyze_for = "auto"
        self._sync_side_combo()
        self.cfg.white_bottom = detected
        self._sync_white_bottom_cb()
        self.cfg.save()
        if self._geometry is not None:
            self._geometry.white_bottom = detected
            self.overlay.set_board_geometry(self._geometry)
        dump_calibration(DEBUG_DIR, img, self.vision)
        self._refresh_vision_status()
        self._set_track_checkbox(True)
        self._start_tracking()
        if warning:
            self.status_label.setText(f"Vision calibrated{note}, tracking on — note: {warning}.")
        else:
            self.status_label.setText(f"Vision calibrated{note}. Auto-tracking on — just play.")

    def _on_recognize(self) -> None:
        img = self._grab_board()
        if img is None:
            self.status_label.setText("Calibrate the board first.")
            return
        if not self.vision.calibrated:
            self.status_label.setText("Calibrate vision first (start position).")
            return
        board, debug = self.vision.analyze(img, self.cfg.white_bottom)
        conf = certainty(debug)
        self._update_certainty(conf)
        self.mini_board.set_board(board, self.cfg.white_bottom)
        self._believed = board
        dump_recognition(DEBUG_DIR, img, debug)
        self.fen_edit.setText(board.fen())
        self.status_label.setText(
            f"Recognized {len(board.piece_map())} pieces ({conf * 100:.0f}% certain). "
            f"Debug: debug/recognition_overlay.png.")
        self._start_analysis(board)

    # ---------------------------------------------------------------- tracking
    def _set_track_checkbox(self, on: bool) -> None:
        self.track_cb.blockSignals(True)
        self.track_cb.setChecked(on)
        self.track_cb.blockSignals(False)

    def _start_tracking(self) -> bool:
        if self._cap_region is None or not self.vision.calibrated:
            self.status_label.setText("Calibrate the board and vision before tracking.")
            return False
        self._stop_tracking()
        self.tracker.reset()
        self._reset_tracking_state()
        worker = VisionWorker(self.vision, lambda: self._cap_region,
                              lambda: self.cfg.white_bottom, interval_ms=TICK_MS)
        worker.frame.connect(self._on_frame)     # queued to the GUI thread
        self._vision_worker = worker
        worker.start()
        return True

    def _stop_tracking(self) -> None:
        if self._vision_worker is not None:
            self._vision_worker.stop()
            self._vision_worker = None

    def _reset_tracking_state(self) -> None:
        self._resync_fen = self._analyzing_key = None
        self._resync_count = 0
        self._cert_ema = 0.0
        self._no_board = 0
        self._suggestions = []
        self._believed = None
        self._consensus.clear()

    def _update_certainty(self, conf: float) -> None:
        self.certainty_bar.setValue(int(round(conf * 100)))

    def _refresh_turn_label(self) -> None:
        side = "White" if self.tracker.board.turn else "Black"
        self.turn_label.setText(f"To move: {side}" + ("" if self.tracker.turn_known else " (?)"))

    def _on_flip_turn(self) -> None:
        # Manual realignment: flip whose turn it is and re-analyze.
        self.tracker.set_turn(self.tracker.board.turn == chess.BLACK)
        self._after_commit()
        self.status_label.setText(
            f"Side to move set to {'White' if self.tracker.board.turn else 'Black'}.")

    def _on_track_toggled(self, on: bool) -> None:
        if self._loading:
            return
        self.cfg.auto_track = on
        self.cfg.save()
        if on:
            if self._start_tracking():
                self.status_label.setText("Auto-tracking on.")
            else:
                self._set_track_checkbox(False)
                self.cfg.auto_track = False
                self.cfg.save()
        else:
            self._stop_tracking()
            self.status_label.setText("Auto-tracking off.")

    def _dragging_on_board(self) -> bool:
        """True while the left mouse button is held with the cursor over the board
        — i.e. a piece is being moved. Used to freeze recognition/eval so a drag
        doesn't spray transient positions at the engine."""
        if self._cap_region is None or not _left_mouse_down():
            return False
        x, y = _cursor_xy()
        left, top, w, h = self._cap_region
        return left <= x < left + w and top <= y < top + h

    def _on_frame(self, raw: chess.Board, debug: list) -> None:
        try:
            if self.cfg.pause_on_drag and self._dragging_on_board():
                return   # piece held on the board: freeze the eval, keep current arrows,
                         # and don't feed mid-drag junk to the engine
            frame_cert = certainty(debug)
            self._cert_ema = 0.6 * self._cert_ema + 0.4 * frame_cert
            self._update_certainty(self._cert_ema)
            self._consensus.push(debug)
            self._believed, agreement = (self._consensus.consensus()
                                         if self._consensus.ready() else (raw, 0.0))
            self.mini_board.set_board(self._believed, self.cfg.white_bottom)

            confident = (frame_cert >= FRAME_MIN and self._consensus.ready()
                         and agreement >= AGREEMENT_MIN)
            if confident:
                self._no_board = 0
                self._reconcile(self._believed)
            elif frame_cert < NO_BOARD_CERT:
                self._no_board += 1
                if self._no_board >= NO_BOARD_FRAMES:    # clearly no board for a while
                    self._suggestions = []
            # Always re-filter candidate arrows against the LIVE board: a moved
            # piece's stale arrow vanishes at once while unmoved ones persist.
            self._draw_arrows()
        except Exception as exc:                          # never let one frame crash the app
            self.status_label.setText(f"vision frame error: {exc}")

    def _reconcile(self, board: chess.Board) -> None:
        # Always try the legal-move path FIRST: an unchanged position returns []
        # (turn preserved — vital so the opponent's thinking time doesn't reset the
        # side to move), a legal move flips the turn correctly, and only a genuine
        # unreachable jump falls through to a confirmed resync (turn inferred from
        # the diff). This holds whether or not 'allow illegal' is on; that flag now
        # only governs whether a technically-illegal position is sent to the engine.
        moves = self.tracker.update_to(board)
        if moves is None:
            # A jump (fast play / new game / mid-drag transient). Require it to
            # persist before resyncing so a drag-in-progress (which resolves to a
            # legal position on drop) is not adopted. The turn is inferred from the
            # diff, so a resync never goes 'turn unknown'.
            fen = board.board_fen()
            if fen == self._resync_fen:
                self._resync_count += 1
            else:
                self._resync_fen, self._resync_count = fen, 1
            if self._resync_count >= RESYNC_CONFIRM:
                self.tracker.reset(board, previous=self.tracker.board.copy())
                self._resync_fen = None
                self._after_commit()
                self.status_label.setText("Resynced to the current board.")
            return
        # Legal move, or no change: the tracker now matches the board.
        self._resync_fen = None
        self._after_commit()

    @staticmethod
    def _pos_key(board: chess.Board) -> str:
        return board.board_fen() + (" w" if board.turn == chess.WHITE else " b")

    def _after_commit(self) -> None:
        """Keep the analysed position in step with the board. Re-analyses only
        when the position actually changes, so a stable one keeps deepening. The
        turn is always resolved (diff-inferred), so arrows never blank out."""
        board = self.tracker.board
        self._refresh_turn_label()
        if self._pos_key(board) != self._analyzing_key:
            self._refresh_moves()
            self.fen_edit.setText(board.fen())
            self._start_analysis(board)

    def _draw_arrows(self) -> None:
        self.overlay.set_annotations(visible_annotations(self._believed, self._suggestions))

    def _recalibrate_to_live(self, require_start: bool) -> str:
        """Best-effort re-calibration to whatever is on screen now (the existing
        'Calibrate vision' routine — NOT a new CV path). Adopts the fresh model only
        when it cleanly reads a START position, so a new game that flipped your
        colour or changed the board theme is picked up automatically, while a
        half-set board can never corrupt the templates. Restarts the worker so the
        live read uses the new model. Returns a status note ('' if nothing adopted)."""
        if not self.vision.calibrated:
            return ""
        img = self._grab_board()
        if img is None:
            return ""
        trial = VisionModel()
        try:
            warning = trial.calibrate(img, self.cfg.white_bottom)
            observed = trial.recognize(img)
        except Exception:
            return ""
        if require_start and (warning or observed.board_fen() != chess.Board().board_fen()):
            return ""                              # not a clean start — keep the trusted model
        self.vision = trial
        detected = trial.white_bottom
        flipped = detected != self.cfg.white_bottom
        if flipped:
            self.cfg.analyze_for = "auto"
            self._sync_side_combo()
            self.cfg.white_bottom = detected
            self._sync_white_bottom_cb()
            if self._geometry is not None:
                self._geometry.white_bottom = detected
                self.overlay.set_board_geometry(self._geometry)
            self.cfg.save()
        self._refresh_vision_status()
        if self._vision_worker is not None:        # rebuild the worker on the new model
            self._start_tracking()
        return (f" (recalibrated — {'White' if detected else 'Black'} on bottom)"
                if flipped else " (recalibrated)")

    def _on_reset_game(self) -> None:
        # Re-read the live board fresh: recalibrate if a start position is shown,
        # then reseed from whatever is on screen and restart analysis cleanly.
        if self._controller is not None:
            self._controller.clear()
        note = self._recalibrate_to_live(require_start=True)
        self._reset_tracking_state()
        observed = None
        if self.vision.calibrated:
            img = self._grab_board()
            if img is not None:
                observed = self.vision.recognize(img, self.cfg.white_bottom)
        self.tracker.reset(observed)
        self.overlay.clear()
        self.results_list.clear()
        self._after_commit()
        self.status_label.setText("Re-read the board." + note)

    def _on_new_game(self) -> None:
        # Clean reset to a fresh game from move 1. If a start position is on screen,
        # recalibrate to it first so a colour/theme change carries over cleanly.
        if self._controller is not None:
            self._controller.clear()
        note = self._recalibrate_to_live(require_start=True)
        self._reset_tracking_state()
        self.tracker.reset()                       # start position, White to move
        self.overlay.clear()
        self.results_list.clear()
        self._after_commit()
        self.status_label.setText("New game — start position." + note)

    def _refresh_moves(self) -> None:
        self.moves_view.setPlainText(self.tracker.san_line())

    # ---------------------------------------------------------------- lifecycle
    def closeEvent(self, e: QtGui.QCloseEvent) -> None:
        self._stop_tracking()
        if self._controller is not None:
            self._controller.shutdown()
        if self._capture is not None:
            self._capture.close()
        self._app.quit()
        super().closeEvent(e)
