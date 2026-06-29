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
from src.engine_profiles import PROFILES, make_controller
from src.capture import ScreenCapture, save_image
from src.config import Config
from src.consensus import ConsensusBuffer
from src.engine import find_stockfish
from src.openings import identify as identify_opening
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
MAIA_ELO_MIN = 1100      # Maia 2's trained rating range (values outside clamp)
MAIA_ELO_MAX = 1900


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
        try:
            s = min(self.width(), self.height()) / 8.0
            light, dark = QtGui.QColor(240, 217, 181), QtGui.QColor(181, 136, 99)
            for r in range(8):
                for c in range(8):
                    p.fillRect(QtCore.QRectF(c * s, r * s, s, s),
                               light if (r + c) % 2 == 0 else dark)
            font = p.font()
            font.setPixelSize(max(1, int(s * 0.82)))
            p.setFont(font)
            for r in range(8):
                for c in range(8):
                    sq = chess.square(c, 7 - r) if self._white_bottom else chess.square(7 - c, r)
                    pc = self._board.piece_at(sq)
                    if pc is None:
                        continue
                    glyph = _GLYPHS[pc.symbol().lower()]
                    rect = QtCore.QRectF(c * s, r * s, s, s)
                    if pc.color == chess.WHITE:       # outline white pieces for contrast
                        p.setPen(QtGui.QColor(30, 30, 30))
                        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            p.drawText(rect.translated(dx, dy), QtCore.Qt.AlignCenter, glyph)
                        p.setPen(QtGui.QColor(250, 250, 250))
                    else:
                        p.setPen(QtGui.QColor(20, 20, 20))
                    p.drawText(rect, QtCore.Qt.AlignCenter, glyph)
        except Exception:               # a paint must never crash the app
            import traceback
            traceback.print_exc()
        finally:
            p.end()


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
        self._puzzle_side: bool | None = None        # puzzle: forced side to solve (None = auto)

        self._vision_worker: VisionWorker | None = None

        self.setWindowTitle("Chess Overlay")
        self.setMinimumWidth(420)
        self._build_ui()

        self.overlay.show()
        self.overlay.set_overlay_visible(self.cfg.show_arrows)
        self.overlay.set_show_border(self.cfg.show_border)
        self._refresh_board_status()
        self._refresh_vision_status()
        self._apply_engine_profile()
        self._sync_maia_controls()
        self._init_controller()
        self._reconcile_orientation_controls()
        self._apply_play_mode()
        self._sync_strength_controls()
        self._loading = False

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        tabs = QtWidgets.QTabWidget()
        root.addWidget(tabs)
        tabs.addTab(self._tab_vision(), "Board && Vision")
        tabs.addTab(self._tab_engine(), "Engine")
        tabs.addTab(self._tab_play(), "Play")

        self.status_label = QtWidgets.QLabel("Ready. Calibrate the board, then vision.")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)
        self._connect_signals()

    def _tab_vision(self) -> QtWidgets.QWidget:
        """Board calibration + overlay display + piece recognition — all the CV setup."""
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
        v.addWidget(board)

        overlay = QtWidgets.QGroupBox("Overlay display")
        ov = QtWidgets.QVBoxLayout(overlay)
        self.show_arrows_cb = QtWidgets.QCheckBox("Show arrows on board")
        self.show_arrows_cb.setChecked(self.cfg.show_arrows)
        self.gold_moves_cb = QtWidgets.QCheckBox("Highlight a clearly-best / mate move in gold")
        self.gold_moves_cb.setChecked(self.cfg.gold_moves)
        self.show_border_cb = QtWidgets.QCheckBox("Show calibrated board border")
        self.show_border_cb.setChecked(self.cfg.show_border)
        for cb in (self.show_arrows_cb, self.gold_moves_cb, self.show_border_cb):
            ov.addWidget(cb)
        v.addWidget(overlay)

        vision = QtWidgets.QGroupBox("Piece recognition")
        vv = QtWidgets.QVBoxLayout(vision)
        row = QtWidgets.QHBoxLayout()
        self.calib_vision_btn = QtWidgets.QPushButton("Calibrate vision (start position)")
        self.calib_vision_btn.setToolTip(
            "Learn the pieces from the start position AND auto-detect orientation. "
            "Click again any time the overlay/eval looks wrong to resync.")
        self.recognize_btn = QtWidgets.QPushButton("Recognize now")
        row.addWidget(self.calib_vision_btn)
        row.addWidget(self.recognize_btn)
        vv.addLayout(row)
        self.allow_illegal_cb = QtWidgets.QCheckBox(
            "Allow illegal moves (accept any read, skip legality check)")
        self.allow_illegal_cb.setChecked(self.cfg.allow_illegal)
        vv.addWidget(self.allow_illegal_cb)
        self.certainty_bar = QtWidgets.QProgressBar()
        self.certainty_bar.setRange(0, 100)
        self.certainty_bar.setFormat("Read certainty: %p%")
        vv.addWidget(self.certainty_bar)
        self.vision_status = QtWidgets.QLabel()
        self.vision_status.setWordWrap(True)
        vv.addWidget(self.vision_status)
        vv.addWidget(QtWidgets.QLabel("What the CV sees:"))
        self.mini_board = MiniBoard()
        vv.addWidget(self.mini_board, 1)
        v.addWidget(vision, 1)
        return w

    def _tab_engine(self) -> QtWidgets.QWidget:
        """Engine selection + the option profile (per-engine groups) for it."""
        from src.engine_profiles import ENGINE_ORDER, PROFILES, leela_networks
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        # --- engine selector + always-shown controls ---
        sel = QtWidgets.QGroupBox("Engine")
        slf = QtWidgets.QFormLayout(sel)
        self.engine_combo = QtWidgets.QComboBox()
        for key in ENGINE_ORDER:
            self.engine_combo.addItem(PROFILES[key].label, key)
        self.engine_combo.setCurrentIndex(max(0, self.engine_combo.findData(self.cfg.engine)))
        slf.addRow("Engine", self.engine_combo)
        self.engine_blurb = QtWidgets.QLabel()
        self.engine_blurb.setWordWrap(True)
        slf.addRow(self.engine_blurb)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem("Live (instant, refines)", "live")
        self.mode_combo.addItem("Fixed depth (strong)", "fixed")
        self.mode_combo.addItem("Predictive (a reply to each likely move)", "predictive")
        self.mode_combo.setCurrentIndex(max(0, self.mode_combo.findData(self.cfg.engine_mode)))
        slf.addRow("Mode", self.mode_combo)
        self.lines_spin = self._spin(1, 5, self.cfg.multipv)
        slf.addRow("Lines", self.lines_spin)
        self.engine_status = QtWidgets.QLabel("Engine: starting…")
        self.engine_status.setWordWrap(True)
        slf.addRow(self.engine_status)
        v.addWidget(sel)

        # --- search resources (Stockfish / Leela) ---
        self.search_group = QtWidgets.QGroupBox("Search")
        ef = QtWidgets.QFormLayout(self.search_group)
        self.depth_spin = self._spin(1, 60, self.cfg.engine_depth)
        self.threads_spin = self._spin(1, max(1, os.cpu_count() or 1), self.cfg.engine_threads)
        self.hash_spin = self._spin(16, 8192, self.cfg.engine_hash_mb, step=64)
        ef.addRow("Depth (fixed)", self.depth_spin)
        self.opp_live_cb = QtWidgets.QCheckBox("Refine opponent look-ahead live (deepen over time)")
        self.opp_live_cb.setChecked(self.cfg.opp_lookahead_live)
        self.opp_depth_spin = self._spin(2, 40, self.cfg.opp_lookahead_depth)
        self.opp_max_spin = self._spin(2, 60, self.cfg.opp_lookahead_max)
        ef.addRow("Opponent look-ahead", self.opp_live_cb)
        ef.addRow("Opp preview depth", self.opp_depth_spin)
        ef.addRow("Opp refine ceiling", self.opp_max_spin)
        ef.addRow("Threads", self.threads_spin)
        ef.addRow("Hash (MB)", self.hash_spin)
        v.addWidget(self.search_group)

        # --- Stockfish strength limiter (simulated Elo) ---
        self.strength_group = QtWidgets.QGroupBox("Player eval strength")
        sf = QtWidgets.QFormLayout(self.strength_group)
        self.strength_preset = QtWidgets.QComboBox()
        for text, data in (("Maximum (full strength)", "full"), ("Expert (~2600)", 2600),
                           ("Advanced (~2200)", 2200), ("Intermediate (~1800)", 1800),
                           ("Casual (~1500)", 1500), ("Beginner (~1320)", 1320), ("Custom", "custom")):
            self.strength_preset.addItem(text, data)
        self.strength_preset.setToolTip(
            "Caps Stockfish's strength for YOUR suggestions via UCI_LimitStrength. "
            "Maximum = unchanged full strength.")
        sf.addRow("Preset", self.strength_preset)
        self.strength_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.strength_slider.setRange(1320, 3190)
        self.strength_slider.setSingleStep(20)
        self.strength_slider.setPageStep(100)
        self.strength_slider.setValue(self.cfg.player_elo)
        self.strength_slider.setToolTip("Stockfish's UCI_Elo range is 1320–3190.")
        self.strength_value = QtWidgets.QLabel()
        sf.addRow("Simulated Elo", self.strength_slider)
        sf.addRow("", self.strength_value)
        sf_note = QtWidgets.QLabel(
            "Caps only YOUR suggestions (greens); the opponent prediction (reds) always "
            "stays full strength. Range 1320–3190.")
        sf_note.setWordWrap(True)
        sf.addRow(sf_note)
        v.addWidget(self.strength_group)

        # --- Leela network (its strength/style control) ---
        self.leela_group = QtWidgets.QGroupBox("Leela network")
        lf = QtWidgets.QFormLayout(self.leela_group)
        self.leela_net_combo = QtWidgets.QComboBox()
        for label, path in leela_networks():
            self.leela_net_combo.addItem(label, path)
        if self.cfg.leela_network:
            i = self.leela_net_combo.findData(self.cfg.leela_network)
            if i >= 0:
                self.leela_net_combo.setCurrentIndex(i)
        lf.addRow("Network", self.leela_net_combo)
        self.leela_net_combo.setToolTip(
            "Leela's strength/style IS its network. The strong general net plays at a "
            "high level; a Maia rating net plays human-like at that Elo.")
        l_note = QtWidgets.QLabel("Strength = the network. Strong general net, or a Maia rating net for human play.")
        l_note.setWordWrap(True)
        lf.addRow(l_note)
        v.addWidget(self.leela_group)

        # --- Maia 2 (human Elo). Maia is trained on 1100-1900; above/below clamps. ---
        self.maia_group = QtWidgets.QGroupBox("Maia 2 — human strength")
        mf = QtWidgets.QFormLayout(self.maia_group)
        self.player_elo_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.player_elo_slider.setRange(MAIA_ELO_MIN, MAIA_ELO_MAX)
        self.player_elo_slider.setSingleStep(100)
        self.player_elo_slider.setPageStep(100)
        self.player_elo_slider.setValue(min(MAIA_ELO_MAX, max(MAIA_ELO_MIN, self.cfg.maia_player_elo)))
        self.player_elo_slider.setToolTip("The rating Maia emulates for YOUR moves (the greens).")
        self.player_elo_value = QtWidgets.QLabel()
        self.opp_elo_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.opp_elo_slider.setRange(MAIA_ELO_MIN, MAIA_ELO_MAX)
        self.opp_elo_slider.setSingleStep(100)
        self.opp_elo_slider.setPageStep(100)
        self.opp_elo_slider.setValue(min(MAIA_ELO_MAX, max(MAIA_ELO_MIN, self.cfg.maia_opp_elo)))
        self.opp_elo_slider.setToolTip("The rating Maia emulates for the OPPONENT's moves (the reds).")
        self.opp_elo_value = QtWidgets.QLabel()
        self.maia_model_combo = QtWidgets.QComboBox()
        self.maia_model_combo.addItem("Rapid", "rapid")
        self.maia_model_combo.addItem("Blitz", "blitz")
        self.maia_model_combo.setCurrentIndex(max(0, self.maia_model_combo.findData(self.cfg.maia_model)))
        mf.addRow("Your Elo", self.player_elo_slider)
        mf.addRow("", self.player_elo_value)
        mf.addRow("Opponent Elo", self.opp_elo_slider)
        mf.addRow("", self.opp_elo_value)
        mf.addRow("Model", self.maia_model_combo)
        m_note = QtWidgets.QLabel(
            f"Predicts the moves a human of that rating would play (Maia is trained on "
            f"{MAIA_ELO_MIN}–{MAIA_ELO_MAX}; values outside clamp).")
        m_note.setWordWrap(True)
        mf.addRow(m_note)
        v.addWidget(self.maia_group)

        v.addStretch(1)
        return w

    def _tab_play(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        # Play mode — a top-level switch between normal play and isolated puzzles.
        mode = QtWidgets.QGroupBox("Play mode")
        mrow = QtWidgets.QHBoxLayout(mode)
        self.mode_live_radio = QtWidgets.QRadioButton("Live play")
        self.mode_puzzle_radio = QtWidgets.QRadioButton("Puzzle")
        self.mode_live_radio.setToolTip(
            "Track the live game: your moves (green) and the opponent's likely reply (red).")
        self.mode_puzzle_radio.setToolTip(
            "Treat the on-screen position as an isolated puzzle: find the decisive side "
            "and show its single best move. Eval engines only (Stockfish / Leela).")
        (self.mode_puzzle_radio if self.cfg.play_mode == "puzzle"
         else self.mode_live_radio).setChecked(True)
        mrow.addWidget(self.mode_live_radio)
        mrow.addWidget(self.mode_puzzle_radio)
        mrow.addStretch(1)
        v.addWidget(mode)

        # Player colour and board orientation are INDEPENDENT. Colour sets the
        # green (mine) / red (opponent) split; orientation is purely how the board is
        # read and drawn. Neither control touches the other.
        colour = QtWidgets.QGroupBox("Player colour && board")
        cf = QtWidgets.QFormLayout(colour)
        self.player_colour_combo = QtWidgets.QComboBox()
        self.player_colour_combo.addItem("Auto (whoever's on the bottom)", "auto")
        self.player_colour_combo.addItem("I'm White", "white")
        self.player_colour_combo.addItem("I'm Black", "black")
        self.player_colour_combo.setCurrentIndex(
            max(0, self.player_colour_combo.findData(self.cfg.player_colour_mode)))
        self.player_colour_combo.setToolTip(
            "Who YOU play — sets which moves are green (yours) vs red (opponent's). "
            "This never rotates the board.")
        cf.addRow("Playing as", self.player_colour_combo)
        self.orientation_label = QtWidgets.QLabel()
        self.orientation_label.setWordWrap(True)
        cf.addRow(self.orientation_label)
        self.flip_orient_btn = QtWidgets.QPushButton("Flip board orientation (if vision read it upside down)")
        self.flip_orient_btn.setToolTip(
            "Rotates how the board is read and drawn 180° — use if the arrows/pieces look "
            "upside down. Does NOT change who you're playing as. Recalibrating vision is "
            "the cleaner fix.")
        cf.addRow(self.flip_orient_btn)
        v.addWidget(colour)

        # Whose turn — the weakest signal, so make it explicit and overridable.
        turn = QtWidgets.QGroupBox("Whose turn")
        tf = QtWidgets.QVBoxLayout(turn)
        self.turn_label = QtWidgets.QLabel("To move: —")
        tf.addWidget(self.turn_label)
        trow = QtWidgets.QHBoxLayout()
        self.turn_white_btn = QtWidgets.QPushButton("White to move")
        self.turn_black_btn = QtWidgets.QPushButton("Black to move")
        self.turn_mine_btn = QtWidgets.QPushButton("My move")
        self.turn_opp_btn = QtWidgets.QPushButton("Opponent's move")
        for b in (self.turn_white_btn, self.turn_black_btn, self.turn_mine_btn, self.turn_opp_btn):
            trow.addWidget(b)
        tf.addLayout(trow)
        turn_note = QtWidgets.QLabel(
            "Live: correct whose turn it is if the tracker drifts ('My/Opponent's move' "
            "are relative to your colour). Puzzle: these force which side to solve for, "
            "overriding the auto-picked side.")
        turn_note.setWordWrap(True)
        tf.addWidget(turn_note)
        v.addWidget(turn)

        opts = QtWidgets.QVBoxLayout()
        self.track_cb = QtWidgets.QCheckBox("Auto-track game (live board → engine)")
        self.track_cb.setChecked(self.cfg.auto_track)
        opts.addWidget(self.track_cb)
        self.predict_cb = QtWidgets.QCheckBox("Show opponent's likely moves (red) on their turn")
        self.predict_cb.setChecked(self.cfg.show_predicted)
        opts.addWidget(self.predict_cb)
        self.pause_drag_cb = QtWidgets.QCheckBox("Pause eval while dragging a piece")
        self.pause_drag_cb.setChecked(self.cfg.pause_on_drag)
        opts.addWidget(self.pause_drag_cb)
        v.addLayout(opts)

        # Puzzle options — only meaningful in Puzzle mode (disabled in Live play).
        self.puzzle_group = QtWidgets.QGroupBox("Puzzle options")
        pf = QtWidgets.QVBoxLayout(self.puzzle_group)
        self.puzzle_winning_cb = QtWidgets.QCheckBox(
            "Show only the winning side's move (hide the other side's)")
        self.puzzle_winning_cb.setChecked(self.cfg.puzzle_winning_only)
        pf.addWidget(self.puzzle_winning_cb)
        self.puzzle_auto_btn = QtWidgets.QPushButton("Auto-pick the side to solve")
        self.puzzle_auto_btn.setToolTip(
            "Let the engine pick the decisive side. The 'Whose turn' buttons override "
            "this (force White/Black to move); this clears that override.")
        pf.addWidget(self.puzzle_auto_btn)
        self.puzzle_note = QtWidgets.QLabel(
            "Analyses the position as both sides and shows the decisive side's single "
            "best move. Eval engines only (Stockfish / Leela).")
        self.puzzle_note.setWordWrap(True)
        pf.addWidget(self.puzzle_note)
        v.addWidget(self.puzzle_group)

        self.fen_edit = QtWidgets.QLineEdit(chess.STARTING_FEN)
        v.addWidget(self.fen_edit)
        row = QtWidgets.QHBoxLayout()
        self.analyze_btn = QtWidgets.QPushButton("Analyze FEN")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.newgame_btn = QtWidgets.QPushButton("New game")
        self.newgame_btn.setToolTip(
            "Reset to the starting position and re-detect your colour from the board "
            "(handles starting a new game as the other colour).")
        self.reset_game_btn = QtWidgets.QPushButton("Re-read board")
        self.reset_game_btn.setToolTip(
            "Resync to whatever is on the board right now (mid-game), keeping the "
            "current orientation.")
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
        # Player colour (green/red split only — never rotates) + orientation fallback.
        self.player_colour_combo.currentIndexChanged.connect(self._on_player_colour_changed)
        self.flip_orient_btn.clicked.connect(self._on_flip_orientation)
        # Explicit turn controls (live: pin whose turn; puzzle: force the side to solve).
        self.turn_white_btn.clicked.connect(lambda: self._set_turn(True))
        self.turn_black_btn.clicked.connect(lambda: self._set_turn(False))
        self.turn_mine_btn.clicked.connect(lambda: self._set_turn(self._player_color() == chess.WHITE))
        self.turn_opp_btn.clicked.connect(lambda: self._set_turn(self._player_color() != chess.WHITE))
        # Play mode (Live / Puzzle) + puzzle options.
        self.mode_puzzle_radio.toggled.connect(self._on_play_mode_changed)
        self.puzzle_auto_btn.clicked.connect(self._on_puzzle_auto)
        self.puzzle_winning_cb.toggled.connect(self._on_puzzle_winning_toggled)
        self.strength_preset.currentIndexChanged.connect(self._on_strength_preset)
        self.strength_slider.valueChanged.connect(self._on_strength_slider)
        # Engine selection + per-engine option controls.
        self.engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        self.leela_net_combo.currentIndexChanged.connect(self._on_leela_net_changed)
        self.maia_model_combo.currentIndexChanged.connect(self._on_maia_model_changed)
        self.player_elo_slider.valueChanged.connect(self._on_maia_elo_changed)
        self.opp_elo_slider.valueChanged.connect(self._on_maia_elo_changed)
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
        self._start_engine()

    def _start_engine(self) -> None:
        """(Re)build the controller for the configured engine. All engines expose the
        same updated/failed/ready + request interface, so the rest of the app is
        engine-agnostic. Tears down any existing controller first."""
        if self._controller is not None:
            try:
                self._controller.clear()
                self._controller.shutdown()
            except Exception:
                pass
            self._controller = None
        label = PROFILES.get(self.cfg.engine, PROFILES["stockfish"]).label
        ctrl, err = make_controller(self.cfg)
        if ctrl is None:
            self.engine_status.setText(f"{label}: {err}")
            return
        ctrl.updated.connect(self._on_analysis_updated)
        ctrl.failed.connect(self._on_analysis_failed)
        ctrl.ready.connect(lambda lbl=label: self.engine_status.setText(f"{lbl}: ready."))
        ctrl.start()
        self._controller = ctrl
        self._eng_sig = (self.cfg.engine_threads, self.cfg.engine_hash_mb)
        self.engine_status.setText(f"{label}: starting…")

    def _apply_engine_profile(self) -> None:
        """Show only the option groups the active engine uses (its menu profile)."""
        prof = PROFILES.get(self.cfg.engine, PROFILES["stockfish"])
        f = prof.features
        self.engine_blurb.setText(prof.blurb)
        self.search_group.setVisible("depth" in f)
        self.strength_group.setVisible("strength_elo" in f)
        self.leela_group.setVisible("leela_network" in f)
        self.maia_group.setVisible("player_elo" in f)

    @staticmethod
    def _maia_band(elo: int) -> str:
        if elo <= MAIA_ELO_MIN:
            return "beginner"
        if elo < 1400:
            return "casual"
        if elo < 1700:
            return "intermediate"
        return "advanced club"

    def _sync_maia_controls(self) -> None:
        # Clamp persisted values to Maia's trained range so the UI reflects reality.
        self.cfg.maia_player_elo = min(MAIA_ELO_MAX, max(MAIA_ELO_MIN, self.cfg.maia_player_elo))
        self.cfg.maia_opp_elo = min(MAIA_ELO_MAX, max(MAIA_ELO_MIN, self.cfg.maia_opp_elo))
        for slider, value, elo in ((self.player_elo_slider, self.player_elo_value, self.cfg.maia_player_elo),
                                   (self.opp_elo_slider, self.opp_elo_value, self.cfg.maia_opp_elo)):
            slider.blockSignals(True)
            slider.setValue(elo)
            slider.blockSignals(False)
            value.setText(f"Elo {elo} · {self._maia_band(elo)}")

    def _is_policy_engine(self) -> bool:
        return PROFILES.get(self.cfg.engine, PROFILES["stockfish"]).display == "policy"

    # ----- engine-selection slots -----
    def _on_engine_changed(self, *_) -> None:
        if self._loading:
            return
        self.cfg.engine = self.engine_combo.currentData()
        self.cfg.save()
        self._apply_engine_profile()
        self._start_engine()
        self._reanalyze_current()

    def _on_leela_net_changed(self, *_) -> None:
        if self._loading:
            return
        self.cfg.leela_network = self.leela_net_combo.currentData() or ""
        self.cfg.save()
        if self.cfg.engine == "leela":          # the net is baked into the lc0 process
            self._start_engine()
            self._reanalyze_current()

    def _on_maia_model_changed(self, *_) -> None:
        if self._loading:
            return
        self.cfg.maia_model = self.maia_model_combo.currentData()
        self.cfg.save()
        if self.cfg.engine == "maia2":          # the model is baked into the worker
            self._start_engine()
            self._reanalyze_current()

    def _on_maia_elo_changed(self, *_) -> None:
        if self._loading:
            return
        self.cfg.maia_player_elo = self.player_elo_slider.value()
        self.cfg.maia_opp_elo = self.opp_elo_slider.value()
        self._sync_maia_controls()
        self.cfg.save()
        self._reanalyze_current()               # Elos are per-request; no rebuild needed

    def _start_analysis(self, board: chess.Board) -> None:
        puzzle = self.cfg.play_mode == "puzzle" and not self._is_policy_engine()  # eval engines only
        # Puzzle mode picks the side itself (it analyses each colour), so the
        # tracker's assumed turn can make the position 'invalid' (the side not to
        # move is in check) while it is still a perfectly good puzzle — skip the
        # live-play legality gate for it.
        if not puzzle and not board.is_valid():
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
        # Strength inputs differ per engine: Stockfish uses the simulated-Elo limiter
        # on player_elo; Maia 2 uses your/opponent rating (no limiter). Leela ignores
        # both. All go through the one request() interface.
        if self.cfg.engine == "maia2":
            limit, p_elo, o_elo = False, self.cfg.maia_player_elo, self.cfg.maia_opp_elo
        else:
            limit, p_elo, o_elo = self.cfg.limit_player_strength, self.cfg.player_elo, self.cfg.maia_opp_elo
        multipv = 1 if puzzle else self.cfg.multipv    # a puzzle has a single answer
        self._controller.request(board, multipv, self.cfg.engine_mode,
                                 self.cfg.engine_depth, self._player_color(), self._req_id,
                                 self.cfg.opp_lookahead_live, self.cfg.opp_lookahead_depth,
                                 self.cfg.opp_lookahead_max, limit, p_elo, o_elo, puzzle,
                                 self._puzzle_side if puzzle else None)
        self.stop_btn.setEnabled(True)

    def _player_color(self) -> bool:
        """Which colour the human plays — for the green (mine) / red (opponent) split
        ONLY. Independent of board orientation: ``"auto"`` means you're whoever is on
        the bottom of the screen (the usual case); ``"white"``/``"black"`` force it.
        Changing this never rotates the board, and flipping the board never changes a
        forced colour (on ``"auto"`` the 'bottom army' naturally follows the flip)."""
        mode = self.cfg.player_colour_mode
        if mode == "white":
            return chess.WHITE
        if mode == "black":
            return chess.BLACK
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

    # ----------------------------------------------- orientation / player colour
    def _sync_player_colour_combo(self) -> None:
        self.player_colour_combo.blockSignals(True)
        self.player_colour_combo.setCurrentIndex(
            max(0, self.player_colour_combo.findData(self.cfg.player_colour_mode)))
        self.player_colour_combo.blockSignals(False)

    def _refresh_orientation_label(self) -> None:
        """Spell out the two independent facts: how the board is read, and who you play."""
        orient = "White on bottom" if self.cfg.white_bottom else "Black on bottom"
        colour = "White" if self._player_color() == chess.WHITE else "Black"
        how = "auto" if self.cfg.player_colour_mode == "auto" else "you set"
        self.orientation_label.setText(
            f"Board read {orient} (vision). You play {colour} ({how}) → your moves are green.")

    def _apply_orientation(self, white_bottom: bool) -> None:
        """Adopt a board ORIENTATION (which army is on the bottom). This is vision's
        job; the only manual entry point is the 'Flip board orientation' fallback for
        a wrong read. It rotates the drawing AND re-derives your colour, then
        re-analyses so the green/red split is right immediately."""
        self.cfg.white_bottom = white_bottom
        self.cfg.save()
        if self._geometry is not None:
            self._geometry.white_bottom = white_bottom
            self.overlay.set_board_geometry(self._geometry)
        if self._believed is not None:
            self.mini_board.set_board(self._believed, white_bottom)
        self._refresh_orientation_label()
        self._refresh_turn_label()
        self._reanalyze_current()

    def _on_player_colour_changed(self, *_) -> None:
        """Your COLOUR (auto / white / black) — sets the green/red split and re-runs
        analysis. NEVER touches orientation, so the board never flips when you change
        who you're playing as."""
        if self._loading:
            return
        self.cfg.player_colour_mode = self.player_colour_combo.currentData()
        self.cfg.save()
        self._refresh_orientation_label()
        self._refresh_turn_label()
        self._reanalyze_current()

    def _on_flip_orientation(self) -> None:
        """Manual fallback when vision read the orientation upside down."""
        self._apply_orientation(not self.cfg.white_bottom)
        self.status_label.setText(
            f"Board orientation flipped — {'White' if self.cfg.white_bottom else 'Black'} on bottom.")

    def _adopt_detected_orientation(self, detected: bool) -> None:
        """Adopt a vision-detected orientation (from calibration / a new game). Sets
        only the orientation; your seat is unchanged. Callers re-analyse."""
        self.cfg.white_bottom = detected
        if self._geometry is not None:
            self._geometry.white_bottom = detected
            self.overlay.set_board_geometry(self._geometry)
        self.cfg.save()
        self._refresh_orientation_label()

    def _reconcile_orientation_controls(self) -> None:
        """Sync the colour / orientation controls to the loaded config at startup."""
        self._sync_player_colour_combo()
        self._refresh_orientation_label()

    # ----------------------------------------------------- player-eval strength
    @staticmethod
    def _elo_band(elo: int) -> str:
        if elo < 1500:
            return "Beginner"
        if elo < 1900:
            return "Intermediate"
        if elo < 2300:
            return "Advanced"
        return "Expert"

    def _strength_text(self) -> str:
        if not self.cfg.limit_player_strength:
            return "Full strength (no limit)"
        return f"Elo {self.cfg.player_elo} · {self._elo_band(self.cfg.player_elo)}"

    def _sync_strength_controls(self) -> None:
        if not self.cfg.limit_player_strength:
            idx = self.strength_preset.findData("full")
        else:
            idx = self.strength_preset.findData(self.cfg.player_elo)   # exact preset match?
            if idx < 0:
                idx = self.strength_preset.findData("custom")
        self.strength_preset.blockSignals(True)
        self.strength_preset.setCurrentIndex(max(0, idx))
        self.strength_preset.blockSignals(False)
        self.strength_slider.blockSignals(True)
        self.strength_slider.setValue(self.cfg.player_elo)
        self.strength_slider.blockSignals(False)
        self.strength_slider.setEnabled(self.cfg.limit_player_strength)
        self.strength_value.setText(self._strength_text())

    def _apply_strength(self, limit: bool, elo: int) -> None:
        self.cfg.limit_player_strength = bool(limit)
        self.cfg.player_elo = int(elo)
        self._sync_strength_controls()
        self.cfg.save()
        self._reanalyze_current()

    def _on_strength_preset(self, *_) -> None:
        if self._loading:
            return
        data = self.strength_preset.currentData()
        if data == "full":
            self._apply_strength(False, self.cfg.player_elo)
        elif data == "custom":
            self._apply_strength(True, self.cfg.player_elo)
        else:
            self._apply_strength(True, int(data))

    def _on_strength_slider(self, value: int) -> None:
        if self._loading:
            return
        self._apply_strength(True, value)   # the slider is only enabled while limiting

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
            policy = self._is_policy_engine()    # Maia: show human-move probabilities
            puzzle = self.cfg.play_mode == "puzzle" and not policy
            # In puzzle mode the greens are the (engine-determined) side-to-move's best
            # and the reds are the other side; 'winning side only' hides the reds.
            show_opp = (not self.cfg.puzzle_winning_only) if puzzle else self.cfg.show_predicted
            self.results_list.clear()
            flip = board.turn != chess.WHITE     # show eval ABSOLUTE (white +, black -)
            for s in suggestions:
                try:
                    san = board.san(s.move)
                except Exception:
                    san = s.uci
                value = f"{(s.policy or 0.0) * 100:.0f}%" if policy else s.eval_text_pov(flip)
                self.results_list.addItem(f"#{s.rank}  {san:7s} {value}")
            self._suggestions = build_annotations(suggestions, opp_suggestions, show_opp,
                                                  board.turn == chess.WHITE, self.cfg.gold_moves,
                                                  policy_mode=policy)
            self._draw_arrows()
            if not suggestions:
                self.status_label.setText(self._terminal_status(board, opp_suggestions))
            elif puzzle:
                side = "White" if board.turn == chess.WHITE else "Black"
                how = "you forced" if self._puzzle_side is not None else "decisive side"
                extra = "" if self.cfg.puzzle_winning_only else " — other side's best in red"
                self.status_label.setText(
                    f"Puzzle — {side} to move ({how}). Best move for {side}{extra}.")
            else:
                note = ""
                if opp_suggestions:
                    if self.cfg.engine_mode == "predictive" or policy:
                        note = f"  (a reply to each of their {len(opp_suggestions)} likely moves)"
                    else:
                        try:
                            opp_san = self.tracker.board.san(opp_suggestions[0].move)
                        except Exception:
                            opp_san = opp_suggestions[0].move.uci()
                        note = f"  (prep vs their best: {opp_san})"
                head = (f"Maia 2 — {len(suggestions)} human move(s)" if policy
                        else f"Depth {depth}, {len(suggestions)} line(s)")
                self.status_label.setText(f"{head}.{note}")
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
        # Orientation / player colour and strength are owned by dedicated handlers
        # (_on_player_colour_changed / _on_strength_*), not read here.
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
        self._adopt_detected_orientation(detected)
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
        self._puzzle_side = None        # a fresh position re-auto-picks the puzzle side

    def _update_certainty(self, conf: float) -> None:
        self.certainty_bar.setValue(int(round(conf * 100)))

    def _refresh_turn_label(self) -> None:
        if self.cfg.play_mode == "puzzle":
            if self._puzzle_side is None:
                self.turn_label.setText("Solving: auto (the decisive side)")
            else:
                self.turn_label.setText(
                    f"Solving: {'White' if self._puzzle_side else 'Black'} to move (you forced)")
            return
        side = "White" if self.tracker.board.turn else "Black"
        mine = "you" if self.tracker.board.turn == self._player_color() else "opponent"
        self.turn_label.setText(
            f"To move: {side} ({mine})" + ("" if self.tracker.turn_known else "  — uncertain"))

    def _set_turn(self, white_to_move: bool) -> None:
        """The 'whose turn' buttons. Live: pin whose turn it is on the tracker. Puzzle:
        force which side we solve for (overriding the auto-picked decisive side)."""
        if self.cfg.play_mode == "puzzle":
            self._puzzle_side = bool(white_to_move)
            self._refresh_turn_label()
            self._reanalyze_current()
            self.status_label.setText(
                f"Puzzle — solving for {'White' if white_to_move else 'Black'}.")
            return
        self.tracker.set_turn(bool(white_to_move))
        self._after_commit()
        self.status_label.setText(f"Set: {'White' if white_to_move else 'Black'} to move.")

    def _apply_play_mode(self) -> None:
        """Reflect the active play mode in the UI: the puzzle options are live only in
        puzzle mode, and the opponent-reds toggle is a live-play concept."""
        puzzle = self.cfg.play_mode == "puzzle"
        self.puzzle_group.setEnabled(puzzle)
        self.predict_cb.setEnabled(not puzzle)
        self._refresh_turn_label()

    def _on_play_mode_changed(self, *_) -> None:
        if self._loading:
            return
        self.cfg.play_mode = "puzzle" if self.mode_puzzle_radio.isChecked() else "live"
        self.cfg.save()
        self._puzzle_side = None                     # a fresh mode auto-picks the side
        self._apply_play_mode()
        if self.cfg.play_mode == "puzzle" and self._is_policy_engine():
            self.status_label.setText(
                "Puzzle mode needs an eval engine (Stockfish or Leela) — Maia plays normally.")
        elif self.cfg.play_mode == "puzzle":
            self.status_label.setText("Puzzle mode — solving the on-screen position.")
        else:
            self.status_label.setText("Live play.")
        self._reanalyze_current()

    def _on_puzzle_auto(self) -> None:
        self._puzzle_side = None
        self._refresh_turn_label()
        self._reanalyze_current()
        self.status_label.setText("Puzzle — auto-picking the decisive side.")

    def _on_puzzle_winning_toggled(self, on: bool) -> None:
        if self._loading:
            return
        self.cfg.puzzle_winning_only = on
        self.cfg.save()
        self._reanalyze_current()

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
            self._adopt_detected_orientation(detected)
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
        # A new game = re-detect orientation from the fresh start position and let your
        # colour follow whoever is on the bottom. So: colour -> auto, recalibrate vision
        # (which sets white_bottom), then reseed from move 1.
        if self._controller is not None:
            self._controller.clear()
        self.cfg.player_colour_mode = "auto"
        self._sync_player_colour_combo()
        self.cfg.save()
        note = self._recalibrate_to_live(require_start=True)
        self._reset_tracking_state()
        self.tracker.reset()                       # start position, White to move
        self.overlay.clear()
        self.results_list.clear()
        self._after_commit()
        self._refresh_orientation_label()
        self.status_label.setText("New game — colour auto-set from the board." + note)

    def _refresh_moves(self) -> None:
        self.moves_view.setPlainText(self.tracker.san_line())
        self._refresh_opening()

    def _refresh_opening(self) -> None:
        name = identify_opening(self.tracker.board)
        self.opening_label.setText(f"Opening: {name}" if name else "Opening: —")

    # ---------------------------------------------------------------- lifecycle
    def closeEvent(self, e: QtGui.QCloseEvent) -> None:
        self._stop_tracking()
        if self._controller is not None:
            self._controller.shutdown()
        if self._capture is not None:
            self._capture.close()
        self._app.quit()
        super().closeEvent(e)
