"""Opening identification — name the opening from the moves played so far.

A curated table of common openings (name -> SAN line). At import the lines are
replayed once into a position index (EPD -> name); ``identify`` then walks the
game's moves from the start and returns the DEEPEST table position it passes
through — i.e. the most specific opening reached (e.g. it upgrades 'Ruy Lopez' to
'Ruy Lopez: Morphy Defense' once ...a6 appears, and keeps the last known name
after the game leaves book).

Matching is by EPD (placement + side + castling + en-passant), so transpositions
are handled and move-order doesn't matter. The table is intentionally a focused,
verified set of the openings a player actually meets, not an exhaustive ECO dump.
"""
from __future__ import annotations

import chess

# name -> space-separated SAN. Ordered roughly general -> specific; only the EPDs
# matter for lookup (distinct positions), so ordering is just for readability.
_OPENINGS: dict[str, str] = {
    # --- 1.e4 e5 -------------------------------------------------------------
    "Open Game": "e4 e5",
    "Ruy Lopez": "e4 e5 Nf3 Nc6 Bb5",
    "Ruy Lopez: Berlin Defense": "e4 e5 Nf3 Nc6 Bb5 Nf6",
    "Ruy Lopez: Morphy Defense": "e4 e5 Nf3 Nc6 Bb5 a6",
    "Ruy Lopez: Exchange": "e4 e5 Nf3 Nc6 Bb5 a6 Bxc6",
    "Ruy Lopez: Closed": "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7",
    "Italian Game": "e4 e5 Nf3 Nc6 Bc4",
    "Italian Game: Giuoco Piano": "e4 e5 Nf3 Nc6 Bc4 Bc5",
    "Italian Game: Two Knights Defense": "e4 e5 Nf3 Nc6 Bc4 Nf6",
    "Evans Gambit": "e4 e5 Nf3 Nc6 Bc4 Bc5 b4",
    "Hungarian Defense": "e4 e5 Nf3 Nc6 Bc4 Be7",
    "Scotch Game": "e4 e5 Nf3 Nc6 d4",
    "Scotch Gambit": "e4 e5 Nf3 Nc6 d4 exd4 Bc4",
    "Four Knights Game": "e4 e5 Nf3 Nc6 Nc3 Nf6",
    "Italian Four Knights": "e4 e5 Nf3 Nc6 Nc3 Nf6 Bc4",
    "Three Knights Game": "e4 e5 Nf3 Nc6 Nc3",
    "Petrov's Defense": "e4 e5 Nf3 Nf6",
    "Philidor Defense": "e4 e5 Nf3 d6",
    "Ponziani Opening": "e4 e5 Nf3 Nc6 c3",
    "King's Gambit": "e4 e5 f4",
    "King's Gambit Accepted": "e4 e5 f4 exf4",
    "King's Gambit Declined": "e4 e5 f4 Bc5",
    "Vienna Game": "e4 e5 Nc3",
    "Bishop's Opening": "e4 e5 Bc4",
    "Center Game": "e4 e5 d4 exd4",
    "Danish Gambit": "e4 e5 d4 exd4 c3",
    # --- 1.e4 c5 (Sicilian) --------------------------------------------------
    "Sicilian Defense": "e4 c5",
    "Sicilian: Alapin": "e4 c5 c3",
    "Sicilian: Closed": "e4 c5 Nc3 Nc6 g3",
    "Sicilian: Grand Prix Attack": "e4 c5 Nc3 Nc6 f4",
    "Sicilian: Smith-Morra Gambit": "e4 c5 d4 cxd4 c3",
    "Sicilian: Rossolimo": "e4 c5 Nf3 Nc6 Bb5",
    "Sicilian: Moscow Variation": "e4 c5 Nf3 d6 Bb5+",
    "Sicilian: Najdorf": "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6",
    "Sicilian: Dragon": "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 g6",
    "Sicilian: Scheveningen": "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 e6",
    "Sicilian: Classical": "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 Nc6",
    "Sicilian: Sveshnikov": "e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 e5",
    "Sicilian: Accelerated Dragon": "e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 g6",
    "Sicilian: Taimanov": "e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6",
    "Sicilian: Kan": "e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6",
    # --- 1.e4 e6 / c6 / d5 / d6 / g6 / Nf6 -----------------------------------
    "French Defense": "e4 e6",
    "French: Advance": "e4 e6 d4 d5 e5",
    "French: Exchange": "e4 e6 d4 d5 exd5",
    "French: Tarrasch": "e4 e6 d4 d5 Nd2",
    "French: Winawer": "e4 e6 d4 d5 Nc3 Bb4",
    "French: Classical": "e4 e6 d4 d5 Nc3 Nf6",
    "French: Rubinstein": "e4 e6 d4 d5 Nc3 dxe4",
    "Caro-Kann Defense": "e4 c6",
    "Caro-Kann: Advance": "e4 c6 d4 d5 e5",
    "Caro-Kann: Exchange": "e4 c6 d4 d5 exd5",
    "Caro-Kann: Panov Attack": "e4 c6 d4 d5 exd5 cxd5 c4",
    "Caro-Kann: Classical": "e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5",
    "Caro-Kann: Two Knights": "e4 c6 Nc3 d5 Nf3",
    "Scandinavian Defense": "e4 d5",
    "Scandinavian: Main Line": "e4 d5 exd5 Qxd5 Nc3 Qa5",
    "Scandinavian: Modern": "e4 d5 exd5 Nf6",
    "Pirc Defense": "e4 d6 d4 Nf6 Nc3 g6",
    "Modern Defense": "e4 g6",
    "Alekhine's Defense": "e4 Nf6",
    "Nimzowitsch Defense": "e4 Nc6",
    "Owen Defense": "e4 b6",
    # --- 1.d4 d5 -------------------------------------------------------------
    "Queen's Pawn Game": "d4 d5",
    "Queen's Gambit": "d4 d5 c4",
    "Queen's Gambit Accepted": "d4 d5 c4 dxc4",
    "Queen's Gambit Declined": "d4 d5 c4 e6",
    "QGD: Exchange Variation": "d4 d5 c4 e6 Nc3 Nf6 cxd5",
    "QGD: Tarrasch Defense": "d4 d5 c4 e6 Nc3 c5",
    "Slav Defense": "d4 d5 c4 c6",
    "Semi-Slav Defense": "d4 d5 c4 e6 Nc3 Nf6 Nf3 c6",
    "Albin Countergambit": "d4 d5 c4 e5",
    "Chigorin Defense": "d4 d5 c4 Nc6",
    "London System": "d4 d5 Nf3 Nf6 Bf4",
    "Colle System": "d4 d5 Nf3 Nf6 e3",
    # --- 1.d4 Nf6 (Indian) ---------------------------------------------------
    "Indian Defense": "d4 Nf6",
    "Trompowsky Attack": "d4 Nf6 Bg5",
    "London System (Indian)": "d4 Nf6 Nf3 e6 Bf4",
    "Torre Attack": "d4 Nf6 Nf3 e6 Bg5",
    "Nimzo-Indian Defense": "d4 Nf6 c4 e6 Nc3 Bb4",
    "Queen's Indian Defense": "d4 Nf6 c4 e6 Nf3 b6",
    "Bogo-Indian Defense": "d4 Nf6 c4 e6 Nf3 Bb4+",
    "Catalan Opening": "d4 Nf6 c4 e6 g3",
    "King's Indian Defense": "d4 Nf6 c4 g6 Nc3 Bg7",
    "Grünfeld Defense": "d4 Nf6 c4 g6 Nc3 d5",
    "Benoni Defense": "d4 Nf6 c4 c5",
    "Modern Benoni": "d4 Nf6 c4 c5 d5 e6",
    "Benko Gambit": "d4 Nf6 c4 c5 d5 b5",
    "Old Indian Defense": "d4 Nf6 c4 d6",
    # --- 1.d4 others ---------------------------------------------------------
    "Dutch Defense": "d4 f5",
    "Dutch: Leningrad": "d4 f5 g3 Nf6 Bg2 g6",
    "Englund Gambit": "d4 e5",
    # --- 1.c4 / 1.Nf3 / flank ------------------------------------------------
    "English Opening": "c4",
    "English: Symmetrical": "c4 c5",
    "English: Reversed Sicilian": "c4 e5",
    "English: Anglo-Indian": "c4 Nf6",
    "English: Agincourt": "c4 e6",
    "Réti Opening": "Nf3 d5 c4",
    "Zukertort Opening": "Nf3",
    "King's Indian Attack": "Nf3 d5 g3",
    "Bird's Opening": "f4",
    "Nimzo-Larsen Attack": "b3",
    "Sokolsky Opening": "b4",
    "Grob Opening": "g4",
}

_INDEX: dict[str, str] | None = None


def _index() -> dict[str, str]:
    global _INDEX
    if _INDEX is None:
        idx: dict[str, str] = {}
        for name, sans in _OPENINGS.items():
            board = chess.Board()
            try:
                for token in sans.split():
                    board.push_san(token)
            except (ValueError, AssertionError):
                continue                     # skip a malformed line rather than crash
            idx.setdefault(board.epd(), name)
        _INDEX = idx
    return _INDEX


def identify(board: chess.Board) -> str | None:
    """Name the opening for ``board``. Walks the game from the standard start and
    returns the deepest table position passed through (the most specific opening,
    retained after leaving book). Returns None if the game isn't rooted at the
    start (e.g. a mid-game reseed) and the current position isn't itself in book."""
    idx = _index()
    start = chess.Board()
    if board.root().board_fen() != start.board_fen():
        return idx.get(board.epd())          # not from the start; best-effort direct lookup
    replay = chess.Board()
    name = None
    for move in board.move_stack:
        replay.push(move)
        hit = idx.get(replay.epd())
        if hit:
            name = hit
    return name
