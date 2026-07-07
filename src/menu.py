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
from src.engine_profiles import (COMBINED_ENGINES, PROFILES, availability,
                                 make_controller)
from src.capture import ScreenCapture, save_image
from src.config import Config
from src.consensus import ConsensusBuffer
from src.engine import find_stockfish
from src.openings import identify as identify_opening
from src.orientation import detect_orientation
from src.overlay import (Annotation, BoardGeometry, ENGINE_COLORS, OverlayManager,
                         build_annotations, build_combined_annotations,
                         build_puzzle_line, visible_annotations)
from src import themes
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
# Maia 2 accepts any Elo but BUCKETS it internally (maia2's create_elo_dict): everything
# under 1100 is ONE band, everything 2000+ is ONE band, and 100-wide bands sit between.
# Confirmed against the model — 600..1099 give identical play, as do 2000..2600. So the
# full useful input range is 600–2600 (11 distinct levels); _maia_band spells out which.
MAIA_ELO_MIN = 600
MAIA_ELO_MAX = 2600
MAIA_ELO_STEP = 100
ORIENT_FLIP_P = 0.04     # CV must be >=96% sure the orientation is wrong to auto-flip
ORIENT_FLIP_FRAMES = 4   # confident disagreeing frames required before auto-flipping


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
        self.setMinimumSize(150, 150)

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
        self._puzzle_side: bool | None = None        # puzzle: side to solve for now (None = pick it)
        self._puzzle_side_forced = False             # did YOU pin the side (vs auto/parity)?
        self._puzzle_side_source = "auto"            # where the side came from (status text)
        self._puzzle_anchor: str | None = None       # placement the current puzzle side applies to
        self._last_highlight = None                  # latest move-highlight read (VisionWorker)
        self._combined_state: dict = {}              # combined mode: engine_key -> (suggestions, board)
        self._filter_arrows = True                   # hide arrows off the live board (off for solution lines)
        self._orient_belief: tuple = (None, 0.5)     # (CV agrees with active orientation? , confidence)
        self._orient_votes = 0                       # consecutive confident-disagree frames
        self._orient_locked_key: str | None = None   # placement we last re-oriented on (no ping-pong)

        self._vision_worker: VisionWorker | None = None

        self.setWindowTitle("Chess Overlay")
        self.setMinimumWidth(680)       # 2-column rows need a little width to breathe
        self.resize(820, 600)           # wider but shorter than before (was 640x720)
        self._build_ui()

        self.overlay.show()
        self.overlay.set_overlay_visible(self.cfg.show_arrows)
        self.overlay.set_show_border(self.cfg.show_border)
        self._refresh_orientation_indicator()
        self._refresh_board_status()
        self._refresh_vision_status()
        self._refresh_theme_combo()
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
        self.reposition_btn = QtWidgets.QPushButton("Reposition board only (keep pieces)…")
        self.reposition_btn.setToolTip(
            "Re-drag the board box WITHOUT relearning the pieces — for when the board "
            "moved or changed size on screen. Templates are size-independent, so this "
            "just re-aims the capture. Recalibrate vision only when the THEME changes.")
        bf.addRow(self.reposition_btn)
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
        self.show_orient_cb = QtWidgets.QCheckBox("Show board-direction indicator (W/B side arrow)")
        self.show_orient_cb.setChecked(self.cfg.show_orientation)
        self.show_orient_cb.setToolTip(
            "Draws which way the board faces beside it: cyan = the CV agrees with this "
            "orientation, amber = it thinks the board is flipped.")
        self.auto_orient_cb = QtWidgets.QCheckBox("Auto-correct orientation when the CV is confident")
        self.auto_orient_cb.setChecked(self.cfg.auto_orient)
        self.auto_orient_cb.setToolTip(
            "Flip the board automatically when the CV is confident (>=96%, sustained) "
            "that the current orientation is upside down.")
        for cb in (self.show_arrows_cb, self.gold_moves_cb, self.show_border_cb,
                   self.show_orient_cb, self.auto_orient_cb):
            ov.addWidget(cb)
        ov.addStretch(1)

        # Board and Overlay side by side — trade width for height (see _row).
        v.addLayout(self._row(board, overlay))

        # Piece recognition: the controls on the left, the live mini-board on the right.
        vision = QtWidgets.QGroupBox("Piece recognition")
        vg = QtWidgets.QHBoxLayout(vision)
        left = QtWidgets.QVBoxLayout()
        row = QtWidgets.QHBoxLayout()
        self.calib_vision_btn = QtWidgets.QPushButton("Calibrate vision (start position)")
        self.calib_vision_btn.setToolTip(
            "Learn the pieces from the start position AND auto-detect orientation. "
            "Click again any time the overlay/eval looks wrong to resync.")
        self.recognize_btn = QtWidgets.QPushButton("Recognize now")
        row.addWidget(self.calib_vision_btn)
        row.addWidget(self.recognize_btn)
        left.addLayout(row)
        trow = QtWidgets.QHBoxLayout()
        trow.addWidget(QtWidgets.QLabel("Theme:"))
        self.theme_combo = QtWidgets.QComboBox()
        self.theme_combo.setToolTip(
            "Saved piece sets. Selecting one loads it instantly (no recalibration); "
            "themes work at any board size.")
        self.theme_save_btn = QtWidgets.QPushButton("Save current…")
        self.theme_delete_btn = QtWidgets.QPushButton("Delete")
        trow.addWidget(self.theme_combo, 1)
        trow.addWidget(self.theme_save_btn)
        trow.addWidget(self.theme_delete_btn)
        left.addLayout(trow)
        self.allow_illegal_cb = QtWidgets.QCheckBox(
            "Allow illegal moves (accept any read, skip legality check)")
        self.allow_illegal_cb.setChecked(self.cfg.allow_illegal)
        left.addWidget(self.allow_illegal_cb)
        self.certainty_bar = QtWidgets.QProgressBar()
        self.certainty_bar.setRange(0, 100)
        self.certainty_bar.setFormat("Read certainty: %p%")
        left.addWidget(self.certainty_bar)
        self.vision_status = QtWidgets.QLabel()
        self.vision_status.setWordWrap(True)
        left.addWidget(self.vision_status)
        left.addStretch(1)
        vg.addLayout(left, 1)
        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel("What the CV sees:"))
        self.mini_board = MiniBoard()
        right.addWidget(self.mini_board)
        right.addStretch(1)
        vg.addLayout(right)
        v.addWidget(vision)
        v.addStretch(1)
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
        self._engine_form = slf         # so the Lines row can be hidden per engine profile
        self.engine_status = QtWidgets.QLabel("Engine: starting…")
        self.engine_status.setWordWrap(True)
        slf.addRow(self.engine_status)

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
        v.addLayout(self._row(sel, self.search_group))

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
        l_note = QtWidgets.QLabel(
            "Strength = the network. The strong net plays at a high level; a 'Maia-1' net "
            "emulates ONE fixed rating (human-style, inside lc0). For tunable, opponent-aware "
            "human play, use the separate Maia 2 engine instead.")
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
            "Predicts the move a human would play; BOTH ratings condition the model. Works "
            "in bands (600–2600): under-1100 and 2000+ each behave as one level, with "
            "100-point bands between. No search — it shows candidate moves ('Lines'), not a "
            "deep line.")
        m_note.setWordWrap(True)
        mf.addRow(m_note)
        # Strength / Leela / Maia are mutually exclusive (see _apply_engine_profile hides
        # the inactive ones), so stacking them costs no height — only the active one shows.
        v.addWidget(self.maia_group)

        # --- Combined: which engines to overlay, and how many arrows each ---
        self.combined_group = QtWidgets.QGroupBox("Combined engines — overlay several at once")
        cg = QtWidgets.QGridLayout(self.combined_group)
        self.combined_visible_cbs: dict = {}
        self.combined_lines_spins: dict = {}
        for row, key in enumerate(COMBINED_ENGINES):
            ok, reason = availability(key)
            col = ENGINE_COLORS.get(key, (200, 200, 200))
            swatch = QtWidgets.QLabel()
            swatch.setFixedSize(14, 14)
            swatch.setStyleSheet(f"background: rgb{tuple(col)}; border-radius: 3px;")
            cb = QtWidgets.QCheckBox(PROFILES[key].label.split(" (")[0])
            cb.setChecked(bool(self.cfg.combined_visible.get(key)) and ok)
            cb.setEnabled(ok)
            spin = QtWidgets.QSpinBox()
            spin.setRange(1, 5)
            spin.setValue(int(self.cfg.combined_lines.get(key, 1)))
            spin.setEnabled(ok)
            spin.setToolTip(reason if not ok else f"How many of {cb.text()}'s top moves to draw.")
            cb.setToolTip(reason if not ok else "Show this engine's pick in the overlay.")
            cb.toggled.connect(lambda on, k=key: self._on_combined_visible_toggled(k, on))
            spin.valueChanged.connect(lambda val, k=key: self._on_combined_lines_changed(k, val))
            cg.addWidget(swatch, row, 0)
            cg.addWidget(cb, row, 1)
            cg.addWidget(QtWidgets.QLabel("arrows"), row, 2)
            cg.addWidget(spin, row, 3)
            self.combined_visible_cbs[key] = cb
            self.combined_lines_spins[key] = spin
        cg.setColumnStretch(1, 1)
        c_note = QtWidgets.QLabel(
            "Runs the ticked engines together — each draws its best move(s) in its colour "
            "(solid = your move, dashed = its prediction on the opponent's turn). Unticked "
            "engines don't run, so hiding one frees its compute. Each engine keeps its own "
            "network / Elo / threads from above.")
        c_note.setWordWrap(True)
        cg.addWidget(c_note, len(COMBINED_ENGINES), 0, 1, 4)
        v.addWidget(self.combined_group)

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

        # Player colour and board orientation are INDEPENDENT. Colour sets the green
        # (mine) / red (opponent) split; orientation is purely how the board is read
        # and drawn. Neither control touches the other.
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
        self.flip_orient_btn = QtWidgets.QPushButton("Flip board orientation")
        self.flip_orient_btn.setToolTip(
            "Rotates how the board is read and drawn 180° — use if the arrows/pieces look "
            "upside down. Does NOT change who you're playing as. Recalibrating vision is "
            "the cleaner fix.")
        cf.addRow(self.flip_orient_btn)

        # Whose turn — the weakest signal, so make it explicit and overridable.
        turn = QtWidgets.QGroupBox("Whose turn")
        turn.setToolTip("Live: fix whose turn it is if the tracker drifts. Puzzle: force "
                        "which side to solve for (overrides the auto-pick).")
        tf = QtWidgets.QVBoxLayout(turn)
        self.turn_label = QtWidgets.QLabel("To move: —")
        tf.addWidget(self.turn_label)
        tgrid = QtWidgets.QGridLayout()
        self.turn_white_btn = QtWidgets.QPushButton("White to move")
        self.turn_black_btn = QtWidgets.QPushButton("Black to move")
        self.turn_mine_btn = QtWidgets.QPushButton("My move")
        self.turn_opp_btn = QtWidgets.QPushButton("Opponent's move")
        tgrid.addWidget(self.turn_white_btn, 0, 0)
        tgrid.addWidget(self.turn_black_btn, 0, 1)
        tgrid.addWidget(self.turn_mine_btn, 1, 0)
        tgrid.addWidget(self.turn_opp_btn, 1, 1)
        tf.addLayout(tgrid)

        # Puzzle options — only meaningful in Puzzle mode (disabled in Live play).
        self.puzzle_group = QtWidgets.QGroupBox("Puzzle options")
        self.puzzle_group.setToolTip(
            "Solves the on-screen position and streams the forced solution (like live). The "
            "side is found once and tracked as you play. Eval engines only.")
        pf = QtWidgets.QVBoxLayout(self.puzzle_group)
        arow = QtWidgets.QHBoxLayout()
        arow.addWidget(QtWidgets.QLabel("Show moves ahead:"))
        self.puzzle_lookahead_spin = self._spin(0, 12, max(0, self.cfg.puzzle_lookahead))
        self.puzzle_lookahead_spin.setToolTip(
            "How many half-moves PAST the move to play now to also draw, as a fading line "
            "(your moves green, current gold, opponent replies red). 0 = just the next "
            "move. This is display-only — it does not slow the engine.")
        arow.addWidget(self.puzzle_lookahead_spin)
        arow.addStretch(1)
        pf.addLayout(arow)
        self.puzzle_numbers_cb = QtWidgets.QCheckBox("Number the moves (1, 2, 3…) instead of the eval")
        self.puzzle_numbers_cb.setChecked(self.cfg.puzzle_move_numbers)
        self.puzzle_numbers_cb.setToolTip(
            "Label each solution arrow with its place in the line (1 = the move to play now) "
            "instead of the evaluation. Moves that land on the same square share one label.")
        self.puzzle_winning_cb = QtWidgets.QCheckBox("Show only your side's moves (hide opponent replies)")
        self.puzzle_winning_cb.setChecked(self.cfg.puzzle_winning_only)
        self.puzzle_mover_cb = QtWidgets.QCheckBox("Side to move = the side on the bottom (mover's view)")
        self.puzzle_mover_cb.setChecked(self.cfg.puzzle_mover_on_bottom)
        self.puzzle_mover_cb.setToolTip(
            "Puzzles are shown from the side-to-move's perspective, so the BOTTOM army is the "
            "side to move. Derives the side from the (very accurate) board orientation — the "
            "strongest signal, so it takes priority over the highlight and the engine's guess. "
            "The 'Flip board orientation' and 'Whose turn' overrides still win.")
        self.puzzle_highlight_cb = QtWidgets.QCheckBox("Use last-move highlight to fix the side (all themes)")
        self.puzzle_highlight_cb.setChecked(self.cfg.puzzle_use_highlight)
        self.puzzle_highlight_cb.setToolTip(
            "Optional, gated layer: read the site's last-move square highlight to pin whose "
            "move it is with near-certainty (fixes the ~10% side misses). Self-calibrates to "
            "any theme, runs off the main thread, and NEVER touches piece recognition — if "
            "it can't read a highlight it simply abstains and the engine picks the side.")
        # Two columns keep the group short: side-detection on top, display below.
        cbgrid = QtWidgets.QGridLayout()
        cbgrid.addWidget(self.puzzle_mover_cb, 0, 0)
        cbgrid.addWidget(self.puzzle_highlight_cb, 0, 1)
        cbgrid.addWidget(self.puzzle_numbers_cb, 1, 0)
        cbgrid.addWidget(self.puzzle_winning_cb, 1, 1)
        pf.addLayout(cbgrid)
        self.puzzle_auto_btn = QtWidgets.QPushButton("Auto-pick the side to solve")
        self.puzzle_auto_btn.setToolTip(
            "Let the engine pick the side. The 'Whose turn' buttons override this "
            "(force White/Black to move); this clears that override.")
        pf.addWidget(self.puzzle_auto_btn)

        # Live-play options.
        live = QtWidgets.QGroupBox("Live play options")
        lo = QtWidgets.QVBoxLayout(live)
        self.track_cb = QtWidgets.QCheckBox("Auto-track game (live board → engine)")
        self.track_cb.setChecked(self.cfg.auto_track)
        self.predict_cb = QtWidgets.QCheckBox("Show opponent's likely moves (red) on their turn")
        self.predict_cb.setChecked(self.cfg.show_predicted)
        self.pause_drag_cb = QtWidgets.QCheckBox("Pause eval while dragging a piece")
        self.pause_drag_cb.setChecked(self.cfg.pause_on_drag)
        for cb in (self.track_cb, self.predict_cb, self.pause_drag_cb):
            lo.addWidget(cb)

        # Side-by-side groups (see _row) so the panel stays short. Puzzle options span the
        # full width (its checkboxes are a 2-column grid), so 'Whose turn' pairs with 'Live'.
        v.addLayout(self._row(mode, colour))
        v.addLayout(self._row(turn, live))
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
        self.opening_label = QtWidgets.QLabel("Opening: —")
        v.addWidget(self.opening_label)
        # The two read-out panes (candidate moves + the game line) side by side.
        self.results_list = QtWidgets.QListWidget()
        self.results_list.setFixedHeight(72)
        self.moves_view = QtWidgets.QPlainTextEdit()
        self.moves_view.setReadOnly(True)
        self.moves_view.setFixedHeight(72)
        v.addLayout(self._row(self.results_list, self.moves_view))
        return w

    @staticmethod
    def _row(*widgets) -> QtWidgets.QHBoxLayout:
        """Lay widgets out side by side (equal width, top-aligned) — trade width for
        height so the panel stays short."""
        h = QtWidgets.QHBoxLayout()
        for wdg in widgets:
            h.addWidget(wdg, 1, QtCore.Qt.AlignTop)
        return h

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
        self.reposition_btn.clicked.connect(self._on_recalibrate_position)
        self.theme_combo.currentIndexChanged.connect(self._on_theme_selected)
        self.theme_save_btn.clicked.connect(self._on_save_theme)
        self.theme_delete_btn.clicked.connect(self._on_delete_theme)
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
        self.puzzle_lookahead_spin.valueChanged.connect(self._on_puzzle_lookahead_changed)
        self.puzzle_numbers_cb.toggled.connect(self._on_puzzle_numbers_toggled)
        self.puzzle_highlight_cb.toggled.connect(self._on_puzzle_highlight_toggled)
        self.puzzle_mover_cb.toggled.connect(self._on_puzzle_mover_toggled)
        self.strength_preset.currentIndexChanged.connect(self._on_strength_preset)
        self.strength_slider.valueChanged.connect(self._on_strength_slider)
        # Engine selection + per-engine option controls.
        self.engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        self.leela_net_combo.currentIndexChanged.connect(self._on_leela_net_changed)
        self.maia_model_combo.currentIndexChanged.connect(self._on_maia_model_changed)
        self.player_elo_slider.valueChanged.connect(self._on_maia_elo_changed)
        self.opp_elo_slider.valueChanged.connect(self._on_maia_elo_changed)
        for wdg in (self.show_arrows_cb, self.gold_moves_cb, self.show_border_cb,
                    self.show_orient_cb, self.auto_orient_cb,
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
        if hasattr(ctrl, "combined_updated"):     # the combined controller relays per engine
            ctrl.combined_updated.connect(self._on_combined_updated)
        else:
            ctrl.updated.connect(self._on_analysis_updated)
        ctrl.failed.connect(self._on_analysis_failed)
        ctrl.ready.connect(lambda lbl=label: self.engine_status.setText(f"{lbl}: ready."))
        self._combined_state = {}                 # start each engine's arrows fresh
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
        self.combined_group.setVisible("combined" in f)
        # The global 'Lines' (multipv) is per-engine in combined mode, so hide the shared
        # one there (combined's profile omits the 'multipv' feature).
        lines_used = "multipv" in f
        self.lines_spin.setVisible(lines_used)
        lbl = self._engine_form.labelForField(self.lines_spin)
        if lbl is not None:
            lbl.setVisible(lines_used)
        self._sync_mode_combo()

    def _sync_mode_combo(self) -> None:
        """Mode choices depend on the engine. A searcher refines over depth (Live vs Fixed);
        Maia 2 is a SINGLE forward pass, so it offers only Standard vs Predictive — a
        Fixed-depth search mode is meaningless there and isn't shown (don't shoehorn search
        onto Maia). The data values stay shared, so switching engines keeps the mode sane."""
        self.mode_combo.blockSignals(True)
        self.mode_combo.clear()
        if self.cfg.engine == "maia2":
            self.mode_combo.addItem("Standard (one pass)", "live")
            self.mode_combo.addItem("Predictive (a reply to each likely move)", "predictive")
        elif self.cfg.engine == "combined":
            # Combined compares each engine's move for the current position — the searchers
            # refine (Live) or run to depth (Fixed); look-ahead/predictive don't apply here.
            self.mode_combo.addItem("Live (instant, refines)", "live")
            self.mode_combo.addItem("Fixed depth (strong)", "fixed")
        else:
            self.mode_combo.addItem("Live (instant, refines)", "live")
            self.mode_combo.addItem("Fixed depth (strong)", "fixed")
            self.mode_combo.addItem("Predictive (a reply to each likely move)", "predictive")
        i = self.mode_combo.findData(self.cfg.engine_mode)
        if i < 0:                       # saved mode not offered here (e.g. Fixed while on Maia)
            i = 0
            self.cfg.engine_mode = self.mode_combo.itemData(0)
        self.mode_combo.setCurrentIndex(i)
        self.mode_combo.blockSignals(False)

    @staticmethod
    def _maia_band(elo: int) -> str:
        """The EFFECTIVE Maia band for an Elo (see MAIA_ELO_MIN note): sub-1100 and 2000+
        are each a single band, so the label states the real granularity rather than
        implying a continuous scale."""
        if elo < 1100:
            return "under-1100 band (all sub-1100 alike)"
        if elo >= 2000:
            return "2000+ band (all 2000+ alike)"
        lo = (elo // 100) * 100
        return f"{lo}–{lo + 99} band"

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
        def snap(v):
            return int(round(v / MAIA_ELO_STEP) * MAIA_ELO_STEP)
        self.cfg.maia_player_elo = snap(self.player_elo_slider.value())
        self.cfg.maia_opp_elo = snap(self.opp_elo_slider.value())
        self._sync_maia_controls()          # writes the snapped value back + refreshes the band
        self.cfg.save()
        self._reanalyze_current()               # Elos are per-request; no rebuild needed

    def _on_combined_visible_toggled(self, key: str, on: bool) -> None:
        """Tick/untick an engine in combined mode: spawn or tear down just that child, and
        refresh. Off frees the engine's compute; on (re)starts it and re-analyses."""
        if self._loading:
            return
        self.cfg.combined_visible[key] = bool(on)
        self.cfg.save()
        if self.cfg.engine != "combined":
            return                                   # takes effect next time combined is active
        if self._controller is not None and hasattr(self._controller, "set_visible"):
            self._controller.set_visible(key, bool(on))
        if not on:
            self._combined_state.pop(key, None)      # drop its arrows at once
        self._render_combined()
        if on:
            self._reanalyze_current()                # feed the just-spawned engine the position

    def _on_combined_lines_changed(self, key: str, val: int) -> None:
        if self._loading:
            return
        self.cfg.combined_lines[key] = max(1, min(5, int(val)))
        self.cfg.save()
        if self.cfg.engine == "combined":
            self._reanalyze_current()                # per-engine arrow count is per-request

    def _start_analysis(self, board: chess.Board) -> None:
        puzzle = self._puzzle_active()                       # eval engines only
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
        # Puzzle dedup keys on placement + the side we solve (not the tracker's turn),
        # so tracker-turn noise can't retrigger; live keys on placement + turn.
        self._analyzing_key = self._puzzle_key(board) if puzzle else self._pos_key(board)
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
        self._orient_belief = (None, 0.5)     # re-evaluated on the next confident frame
        self._orient_votes = 0
        self._refresh_orientation_indicator()
        # Mover-on-bottom: the side to move IS the bottom army, so flipping the board (manual
        # flip or auto-orient) flips the puzzle side too — keep them consistent before the
        # re-analysis. A manually-pinned side (the turn buttons) still wins.
        if (self._puzzle_active() and self.cfg.puzzle_mover_on_bottom
                and not self._puzzle_side_forced):
            self._puzzle_side = chess.WHITE if white_bottom else chess.BLACK
            self._puzzle_side_source = "mover-on-bottom"
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
        """Manual fallback when vision read the orientation upside down. Locks this
        placement so auto-orient won't immediately undo your choice."""
        self._orient_locked_key = self._orient_canonical_key()
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
        self._orient_belief = (None, 0.5)
        self._orient_votes = 0
        self._refresh_orientation_indicator()

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
            if self._puzzle_active():
                self._on_puzzle_updated(suggestions, board)
                return
            policy = self._is_policy_engine()    # Maia: show human-move probabilities
            self.results_list.clear()
            flip = board.turn != chess.WHITE     # show eval ABSOLUTE (white +, black -)
            for s in suggestions:
                try:
                    san = board.san(s.move)
                except Exception:
                    san = s.uci
                value = f"{(s.policy or 0.0) * 100:.0f}%" if policy else s.eval_text_pov(flip)
                self.results_list.addItem(f"#{s.rank}  {san:7s} {value}")
            self._suggestions = build_annotations(suggestions, opp_suggestions,
                                                  self.cfg.show_predicted,
                                                  board.turn == chess.WHITE, self.cfg.gold_moves,
                                                  policy_mode=policy)
            self._filter_arrows = True
            self._draw_arrows()
            if not suggestions:
                self.status_label.setText(self._terminal_status(board, opp_suggestions))
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

    # ------------------------------------------------ combined (multi-engine) mode
    _ENGINE_TAG = {"stockfish": "SF", "leela": "LC0", "maia2": "Maia"}

    def _on_combined_updated(self, engine_key: str, suggestions: list, depth: int,
                             board: chess.Board, opp_suggestions: list, token: int) -> None:
        """One engine reported its picks for the current position (combined mode). Cache
        them per engine and redraw the union of all visible engines' arrows."""
        if token != self._req_id:
            return
        try:
            self._combined_state[engine_key] = (suggestions, board)
            self._render_combined()
        except Exception as exc:
            self.status_label.setText(f"combined render error: {exc}")

    def _render_combined(self) -> None:
        """Merge every visible engine's cached picks into engine-coloured arrows (solid on
        your turn, dashed on the opponent's) and refresh the overlay + results list."""
        per_engine, results, opp_turn = [], [], False
        for key in COMBINED_ENGINES:
            if not self.cfg.combined_visible.get(key):
                continue
            state = self._combined_state.get(key)
            if not state:
                continue
            sugg, eboard = state
            if eboard is not None:
                opp_turn = eboard.turn != self._player_color()   # all engines share the position
            per_engine.append((key, sugg, eboard))
            if sugg:
                results.append((key, sugg[0], eboard))
        self._suggestions = build_combined_annotations(per_engine, opp_turn)
        self._filter_arrows = True
        self._draw_arrows()
        self._fill_combined_results(results, opp_turn)

    def _fill_combined_results(self, results: list, opp_turn: bool) -> None:
        self.results_list.clear()
        for key, s, eboard in results:
            try:
                san = eboard.san(s.move)
            except Exception:
                san = s.uci
            flip = (eboard is not None and eboard.turn != chess.WHITE)
            val = f"{(s.policy or 0.0) * 100:.0f}%" if s.policy is not None else s.eval_text_pov(flip)
            self.results_list.addItem(f"{self._ENGINE_TAG.get(key, key):5s} {san:7s} {val}")
        active = [k for k in COMBINED_ENGINES if self.cfg.combined_visible.get(k)]
        if not active:
            self.status_label.setText("Combined — no engines ticked (choose at least one).")
        else:
            who = "opponent (dashed)" if opp_turn else "you"
            self.status_label.setText(
                f"Combined — {len(results)}/{len(active)} engine(s) reporting; move for {who}.")

    @staticmethod
    def _stm_winning(top) -> bool:
        """Is the side to move the WINNING (hero) side? After you play the solution it
        becomes the opponent's turn and their 'best' is a losing forced reply — that's
        how we tell a hero-to-move position from an opponent-to-reply one."""
        if top.mate_in is not None:
            return top.mate_in > 0
        return top.score_cp is not None and top.score_cp >= 0

    def _on_puzzle_updated(self, suggestions: list, board: chess.Board) -> None:
        """Render a solved puzzle: the whole forced line as fading arrows (your moves
        green, the move to play now gold, the opponent's forced replies red), or — with
        'show full solution' off — just the single best move. ``board.turn`` is the side
        the engine solved for."""
        # Cache the engine's one-time side pick; parity tracks it from here. The tracker
        # stays turn-unknown until you move, so a cold-wrong pick self-corrects then.
        if self._puzzle_side is None and suggestions:
            self._puzzle_side = board.turn
            self._puzzle_side_source = "engine"      # highlight abstained → the cold pick
            self._analyzing_key = self._puzzle_key(board)
        self.results_list.clear()
        if not suggestions:
            self._suggestions = []
            self._filter_arrows = True
            self._draw_arrows()
            self.status_label.setText(self._terminal_status(board, []))
            return
        top = suggestions[0]
        if self._puzzle_side_forced:
            stm_winning = True                       # you chose this side — show it as yours
            hero_white = (board.turn == chess.WHITE)
        else:
            stm_winning = self._stm_winning(top)
            hero_white = (board.turn == chess.WHITE) if stm_winning else (board.turn != chess.WHITE)
        show_opp = not self.cfg.puzzle_winning_only
        side = "White" if board.turn == chess.WHITE else "Black"
        how = self._puzzle_side_source
        self._fill_puzzle_results(top, board, hero_white)
        # Winning-only + it's the opponent's forced (losing) reply: suppress and wait for
        # your move, rather than flash their move as if it were the solution.
        if self.cfg.puzzle_winning_only and not stm_winning:
            self._suggestions = []
            self._filter_arrows = True
            self._draw_arrows()
            self.status_label.setText(
                f"Puzzle — {side} (opponent) to reply; waiting for your move (winning side only).")
            return
        ahead = max(0, self.cfg.puzzle_lookahead)
        max_plies = 1 + ahead                    # the move to play now + N half-moves ahead
        self._suggestions = build_puzzle_line(top, board, hero_white, show_opp,
                                              max_plies=max_plies,
                                              move_numbers=self.cfg.puzzle_move_numbers)
        # A lone current move can use the live off-board filter (it clears the instant the
        # piece moves); a multi-ply line can't (its later source squares aren't occupied
        # yet), so it draws unfiltered and is refreshed wholesale each move.
        self._filter_arrows = (len(self._suggestions) <= 1)
        self._draw_arrows()
        tail = "" if show_opp else ", your side only"
        if ahead == 0:
            self.status_label.setText(f"Puzzle — {side} to move ({how}). Best move{tail}.")
        else:
            self.status_label.setText(
                f"Puzzle solution ({how}) — {side} to move, {ahead} ahead{tail}.")

    def _fill_puzzle_results(self, top, board: chess.Board, hero_white: bool) -> None:
        """List the solution line as SAN in the results box (you/opponent tagged)."""
        walk = board.copy()
        n = 1 + max(0, self.cfg.puzzle_lookahead)
        for i, mv in enumerate(top.pv[:n]):
            try:
                san = walk.san(mv)
            except Exception:
                break
            who = "you" if ((walk.turn == chess.WHITE) == hero_white) else "opp"
            self.results_list.addItem(f"{'►' if i == 0 else ' '} {san:8s} ({who})")
            walk.push(mv)

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
        self.cfg.show_orientation = self.show_orient_cb.isChecked()
        self.cfg.auto_orient = self.auto_orient_cb.isChecked()
        self.cfg.allow_illegal = self.allow_illegal_cb.isChecked()
        self.cfg.show_predicted = self.predict_cb.isChecked()
        self.cfg.pause_on_drag = self.pause_drag_cb.isChecked()
        self.cfg.opp_lookahead_live = self.opp_live_cb.isChecked()
        self.cfg.opp_lookahead_depth = self.opp_depth_spin.value()
        self.cfg.opp_lookahead_max = self.opp_max_spin.value()
        self.cfg.save()
        self.overlay.set_overlay_visible(self.cfg.show_arrows)
        self.overlay.set_show_border(self.cfg.show_border)
        self._refresh_orientation_indicator()
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

    def _on_recalibrate_position(self) -> None:
        """Re-aim the capture box WITHOUT relearning the pieces (board moved/resized)."""
        if not self.vision.calibrated:
            self.status_label.setText(
                "Calibrate vision once first — then you can reposition without redoing it.")
            return
        self.hide()
        QtWidgets.QApplication.processEvents()
        result = calibrate(self._app, self._monitor_index(), self.cfg.white_bottom)
        self.show()
        self.raise_()
        if result is None:
            self.status_label.setText("Reposition cancelled.")
            return
        self._apply_calibration(result, "manual", keep_vision=True)

    # ----- piece themes -----
    def _refresh_theme_combo(self, select: str | None = None) -> None:
        self.theme_combo.blockSignals(True)
        self.theme_combo.clear()
        self.theme_combo.addItem("— current (unsaved) —", "")
        for name in themes.list_themes():
            self.theme_combo.addItem(name, name)
        if select:
            i = self.theme_combo.findData(select)
            if i >= 0:
                self.theme_combo.setCurrentIndex(i)
        self.theme_combo.blockSignals(False)
        self.theme_delete_btn.setEnabled(bool(self.theme_combo.currentData()))

    def _on_theme_selected(self, *_) -> None:
        if self._loading:
            return
        name = self.theme_combo.currentData()
        self.theme_delete_btn.setEnabled(bool(name))
        if not name:                         # the "current (unsaved)" entry — do nothing
            return
        try:
            model = themes.load_theme(name)
        except Exception as exc:
            self.status_label.setText(f"Couldn't load theme '{name}': {exc}")
            return
        self._stop_tracking()
        self.vision = model
        self._reset_tracking_state()
        self._refresh_vision_status()
        if self._cap_region is not None and self.cfg.auto_track:
            self._set_track_checkbox(True)
            self._start_tracking()
        self.status_label.setText(
            f"Loaded theme '{name}'." + ("" if self._cap_region else " Calibrate the board to use it."))

    def _on_save_theme(self) -> None:
        if not self.vision.calibrated:
            self.status_label.setText("Calibrate vision first, then save it as a theme.")
            return
        name, ok = QtWidgets.QInputDialog.getText(self, "Save piece theme", "Theme name:")
        if not ok or not name.strip():
            return
        try:
            saved = themes.save_theme(name, self.vision)
        except Exception as exc:
            self.status_label.setText(f"Couldn't save theme: {exc}")
            return
        self._refresh_theme_combo(select=saved)
        self.status_label.setText(f"Saved piece theme '{saved}'.")

    def _on_delete_theme(self) -> None:
        name = self.theme_combo.currentData()
        if not name:
            return
        themes.delete_theme(name)
        self._refresh_theme_combo()
        self.status_label.setText(f"Deleted theme '{name}'.")

    def _apply_calibration(self, result, how: str, keep_vision: bool = False) -> None:
        """Adopt a calibration result. ``keep_vision`` repositions the capture box but
        KEEPS the learned pieces (templates are size-normalised, so a different board
        size still matches) — for when the board just moved/resized; otherwise it's a
        clean slate (a possibly-different board/theme) and the pieces are dropped."""
        self.cfg.board_monitor = result.screen_index
        self.cfg.save()
        self._geometry = result.geometry
        self._cap_region = (result.phys_left, result.phys_top, result.phys_side, result.phys_side)
        self.overlay.set_board_geometry(self._geometry)
        self._orient_belief = (None, 0.5)
        self._refresh_orientation_indicator()
        self._refresh_board_status()
        if keep_vision and self.vision.calibrated:
            # Reposition only: same pieces, new box. Re-read at the new region; tracking
            # resyncs to the live position.
            self._reset_tracking_state()
            if self.cfg.auto_track:
                self._set_track_checkbox(True)
                self._start_tracking()
            self._on_snapshot()
            self.status_label.setText(
                f"Board repositioned ({how}, {result.phys_side}px) — pieces kept.")
            return
        # Full (re)calibration is a clean slate: drop the old templates, tracking, arrows.
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
                              lambda: self.cfg.white_bottom, interval_ms=TICK_MS,
                              highlight_fn=lambda: self.cfg.puzzle_use_highlight and self._puzzle_active())
        worker.frame.connect(self._on_frame)     # queued to the GUI thread
        self._vision_worker = worker
        worker.start()
        return True

    def _stop_tracking(self) -> None:
        # Clear the field FIRST, then join: a frame signal already queued to the GUI thread
        # can fire during the join, and it must not see a half-stopped worker.
        worker, self._vision_worker = self._vision_worker, None
        if worker is not None:
            worker.stop()

    def _reset_tracking_state(self) -> None:
        self._resync_fen = self._analyzing_key = None
        self._resync_count = 0
        self._cert_ema = 0.0
        self._no_board = 0
        self._suggestions = []
        self._believed = None
        self._consensus.clear()
        self._puzzle_side = None        # a fresh position re-auto-picks the puzzle side
        self._puzzle_side_forced = False
        self._puzzle_side_source = "auto"
        self._puzzle_anchor = None
        self._last_highlight = None
        self._combined_state = {}       # drop stale per-engine combined arrows
        self._orient_locked_key = None  # ...and re-enables one auto-orient on it

    def _update_certainty(self, conf: float) -> None:
        self.certainty_bar.setValue(int(round(conf * 100)))

    def _refresh_turn_label(self) -> None:
        if self.cfg.play_mode == "puzzle":
            if self._puzzle_side is None:
                self.turn_label.setText("Solving: auto (engine picks the side)")
            else:
                self.turn_label.setText(
                    f"Solving: {'White' if self._puzzle_side else 'Black'} to move "
                    f"({self._puzzle_side_source})")
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
            self._puzzle_side_forced = True
            self._puzzle_side_source = "you set"
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
        self._puzzle_side_forced = False
        self._puzzle_side_source = "auto"
        self._puzzle_anchor = None
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
        self._puzzle_side, self._puzzle_side_source = self._auto_puzzle_side()
        self._puzzle_side_forced = False
        self._refresh_turn_label()
        self._reanalyze_current()
        self.status_label.setText("Puzzle — auto-picking the side.")

    def _on_puzzle_winning_toggled(self, on: bool) -> None:
        if self._loading:
            return
        self.cfg.puzzle_winning_only = on
        self.cfg.save()
        self._reanalyze_current()

    def _on_puzzle_lookahead_changed(self, value: int) -> None:
        if self._loading:
            return
        self.cfg.puzzle_lookahead = int(value)
        self.cfg.save()
        self._reanalyze_current()

    def _on_puzzle_numbers_toggled(self, on: bool) -> None:
        if self._loading:
            return
        self.cfg.puzzle_move_numbers = on
        self.cfg.save()
        self._reanalyze_current()          # relabel the current solution arrows

    def _on_puzzle_highlight_toggled(self, on: bool) -> None:
        if self._loading:
            return
        self.cfg.puzzle_use_highlight = on
        self.cfg.save()
        self._rederive_puzzle_side()

    def _on_puzzle_mover_toggled(self, on: bool) -> None:
        if self._loading:
            return
        self.cfg.puzzle_mover_on_bottom = on
        self.cfg.save()
        self._rederive_puzzle_side()

    def _rederive_puzzle_side(self) -> None:
        """Re-pick the current puzzle's side from the auto source (after a side-source
        toggle changed). A manual pin (the turn buttons) still wins; then re-analyse."""
        if self._puzzle_active() and not self._puzzle_side_forced:
            self._puzzle_side, self._puzzle_side_source = self._auto_puzzle_side()
            self._refresh_turn_label()
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

    def _on_frame(self, raw: chess.Board, debug: list, highlight=None) -> None:
        try:
            if self.cfg.pause_on_drag and self._dragging_on_board():
                return   # piece held on the board: freeze the eval, keep current arrows,
                         # and don't feed mid-drag junk to the engine
            self._last_highlight = highlight     # optional last-move read (None when off/unclear)
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
                if self._puzzle_active():
                    self._reconcile_puzzle(self._believed)
                else:
                    self._reconcile(self._believed)
                self._update_orientation_belief()
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

    def _puzzle_active(self) -> bool:
        """Puzzle mode AND a single eval engine. Maia plays normally even in puzzle mode,
        and Combined is a live-play comparison view — both fall through to the live path."""
        return (self.cfg.play_mode == "puzzle" and not self._is_policy_engine()
                and self.cfg.engine != "combined")

    def _highlight_side(self) -> bool | None:
        """The side to move read from the last-move highlight, if enabled and confident —
        the near-certain override for the engine's ~91% cold pick. None => abstain."""
        h = self._last_highlight
        if (self.cfg.puzzle_use_highlight and h is not None
                and getattr(h, "side_to_move", None) is not None):
            return h.side_to_move
        return None

    def _auto_puzzle_side(self) -> tuple[bool | None, str]:
        """The AUTO (non-forced) side to move for a fresh puzzle + its source, in priority:
          1. mover-on-bottom — the board is shown from the mover's view, so the BOTTOM army
             is the side to move; derived from the ~99% orientation, it beats every other
             signal and is the authority when enabled (highlight/standout defer to it);
          2. the last-move HIGHLIGHT (if enabled and confident);
          3. else None -> the engine cold-picks the side during analysis.
        Only for the INITIAL position — after a move, parity (not this) sets the side."""
        if self.cfg.puzzle_mover_on_bottom:
            return (chess.WHITE if self.cfg.white_bottom else chess.BLACK), "mover-on-bottom"
        hl = self._highlight_side()
        return (hl, "move-highlight") if hl is not None else (None, "engine")

    def _reconcile_puzzle(self, board: chess.Board) -> None:
        """Puzzle reconcile — anchored on PLACEMENT, with the side tracked by PARITY.

        The side to move is determined ONCE per puzzle (the engine's pick, or your
        override) and then strictly alternates as moves are played — it never re-guesses
        per frame, which is what made it flip mid-puzzle (the 'two moves in a row for the
        same side' bug) and ran the both-sides search constantly (the slowness).

        ``tracker.update_to`` follows up to TWO plies, so a site that auto-plays the
        opponent's reply still lands on the right side, and it infers the mover's colour
        from the pieces — so even a COLD-WRONG initial pick self-corrects the instant you
        play the real move (the tracker is left turn-unknown until then, so the observed
        move, not the guess, fixes the parity). Only a genuine board jump (a new puzzle)
        re-opens the one-time pick."""
        fen = board.board_fen()
        if fen == self._puzzle_anchor:
            return                                   # same position — analysis already handled it
        moves = self.tracker.update_to(board)
        if moves:                                    # one or two real moves: parity is now exact
            self._puzzle_side = self.tracker.board.turn
            self._puzzle_side_forced = False         # a played move supersedes any manual pin
            self._puzzle_side_source = "tracked"
            self._resync_fen = None
            self._suggestions = []                   # the move changed the position — drop the stale line
        elif moves is None:                          # a jump: new puzzle (or >2 ply / churn)
            if fen == self._resync_fen:
                self._resync_count += 1
            else:
                self._resync_fen, self._resync_count = fen, 1
            if self._resync_count < RESYNC_CONFIRM:   # let a mid-drag transient settle first
                return
            self._resync_fen = None
            self.tracker.reset(board)                # placement only; turn UNKNOWN until a move
            self._puzzle_side = None                 # re-pick the side once for the new puzzle
            self._puzzle_side_forced = False
            self._suggestions = []
        # Side still unknown (new puzzle, or just entered puzzle mode): pick it once from
        # the best available auto source (mover-on-bottom > highlight > engine — see
        # _auto_puzzle_side). A played move (above) already set the side by parity, so this
        # is skipped then — parity, being ground truth, always wins.
        if self._puzzle_side is None and not self._puzzle_side_forced:
            self._puzzle_side, self._puzzle_side_source = self._auto_puzzle_side()
        # moves == [] : the tracker was already at this placement — only (re)establish the
        # anchor; keep any arrows on screen.
        self._puzzle_anchor = fen
        self._refresh_turn_label()
        self._after_commit_puzzle(board)

    def _after_commit_puzzle(self, board: chess.Board) -> None:
        """Re-analyse the puzzle only when the placement or the side to solve changes
        (keyed on both, so vision noise on the tracker's turn can't retrigger it)."""
        if self._puzzle_key(board) == self._analyzing_key:
            return
        self._refresh_moves()
        self.fen_edit.setText(board.fen())
        self._start_analysis(board)

    def _puzzle_key(self, board: chess.Board) -> str:
        side = "?" if self._puzzle_side is None else ("w" if self._puzzle_side else "b")
        return board.board_fen() + "|" + side

    @staticmethod
    def _pos_key(board: chess.Board) -> str:
        return board.board_fen() + (" w" if board.turn == chess.WHITE else " b")

    def _after_commit(self) -> None:
        """Keep the analysed position in step with the board. Re-analyses only
        when the position actually changes, so a stable one keeps deepening."""
        board = self.tracker.board
        self._refresh_turn_label()
        # Live play, whose-turn still unknown (a cold mid-game reseed with no observed
        # move): assume it's YOUR move — you consult the overlay about your own move —
        # until a real move resolves it. This never marks the turn 'known', so an
        # observed move still corrects it; puzzle mode resolves the side itself.
        analyse = board
        if (self.cfg.play_mode != "puzzle" and not self.tracker.turn_known
                and board.turn != self._player_color()):
            analyse = board.copy()
            analyse.turn = self._player_color()
        if self._pos_key(analyse) != self._analyzing_key:
            self._refresh_moves()
            self.fen_edit.setText(analyse.fen())
            self._start_analysis(analyse)

    def _draw_arrows(self) -> None:
        # Live / single-move: hide arrows whose source square is now empty (a moved
        # piece's stale arrow vanishes). Solution lines show several plies ahead, whose
        # source squares aren't occupied yet, so they bypass the filter and are instead
        # refreshed wholesale when the position changes (see _reconcile_puzzle).
        anns = (self._suggestions if not self._filter_arrows
                else visible_annotations(self._believed, self._suggestions))
        self.overlay.set_annotations(anns)

    def _refresh_orientation_indicator(self) -> None:
        show = self.cfg.show_orientation and self._geometry is not None
        agree, conf = self._orient_belief
        self.overlay.set_orientation(show, agree, conf)

    @staticmethod
    def _mirror(board: chess.Board) -> chess.Board:
        """180° rotation (colours preserved) — the same pieces in the other orientation."""
        return board.transform(chess.flip_vertical).transform(chess.flip_horizontal)

    def _orient_canonical_key(self, board: chess.Board | None = None) -> str | None:
        """A FLIP-INVARIANT identity of the placement: ``min(fen, mirror_fen)`` is
        identical for a board AND its 180° mirror, so flipping the orientation never
        changes it (only a genuinely different puzzle/board does). This is what locks
        auto-orient to one flip per position — even if the believed board momentarily
        lags the setting, the key is the same, so the lock still holds and it can't
        ping-pong. (A plain fen would alternate P/mirror(P) across a flip and defeat
        the lock — that was the residual oscillation.)"""
        b = self._believed if board is None else board
        if b is None:
            return None
        return min(b.board_fen(), self._mirror(b).board_fen())

    def _update_orientation_belief(self) -> None:
        """Drive the overlay indicator and (with auto_orient) correct the orientation.

        The detector runs on a FIXED-frame reference (the believed pieces always mapped
        as if white_bottom=True), so its verdict does NOT change when we flip — auto-
        orient converges in ONE step instead of oscillating (flipping the board used to
        change the detector's own input, which let it ping-pong, mirroring the CV
        preview and the analysis). A per-position lock additionally guarantees at most
        one auto-flip per placement (so it can't ping-pong and won't fight a manual
        flip); a new puzzle/board has a different reference and re-enables it."""
        board = self._believed
        if (board is None or board.king(chess.WHITE) is None
                or board.king(chess.BLACK) is None):
            return
        ref = board if self.cfg.white_bottom else self._mirror(board)
        lock_key = self._orient_canonical_key(board)
        correct_wb, p = detect_orientation(ref)       # the value white_bottom SHOULD have
        agree = (self.cfg.white_bottom == correct_wb)
        conf = p if correct_wb else (1.0 - p)          # confidence that correct_wb is right
        self._orient_belief = (agree, conf)
        if (self.cfg.auto_orient and not agree and conf >= (1.0 - ORIENT_FLIP_P)
                and lock_key != self._orient_locked_key):
            self._orient_votes += 1
            if self._orient_votes >= ORIENT_FLIP_FRAMES:
                self._orient_votes = 0
                self._orient_locked_key = lock_key     # at most one auto-flip per placement
                self._apply_orientation(correct_wb)    # idempotent: ref is unchanged after
                self.status_label.setText(
                    f"Auto-corrected orientation — "
                    f"{'White' if self.cfg.white_bottom else 'Black'} on bottom (CV).")
                return
        else:
            self._orient_votes = 0
        self._refresh_orientation_indicator()

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
        # Stop every worker BEFORE the process tears down. With automatic GC disabled
        # (main._install_gc_guard), a thread still touching Qt / python-chess objects during
        # interpreter shutdown can crash — so join them here, each step isolated so one
        # hang or error can't skip the rest.
        for cleanup in (
                self._stop_tracking,                                       # vision capture QThread
                lambda: self._controller and self._controller.shutdown(),  # engine QThread (+ its asyncio thread)
                lambda: self._capture and self._capture.close()):          # main-thread screen capture
            try:
                cleanup()
            except Exception:
                pass
        self._controller = None
        self._app.quit()
        super().closeEvent(e)
