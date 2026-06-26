"""Chess Overlay — a teaching tool that overlays engine analysis on a chess board.

Package layout
--------------
config      runtime settings (engine path, board geometry, colors, ...)
engine      Stockfish (UCI) wrapper -> top-N moves with evaluations
overlay     transparent, click-through overlay window (draws circles + arrows)

Coming in later phases:
capture     fast screen capture
detect      locate the board on screen
recognize   self-calibrating piece recognition -> FEN
tracker     rules + temporal validation, PGN game log
openings    opening-name lookup
"""

__version__ = "0.1.0"
