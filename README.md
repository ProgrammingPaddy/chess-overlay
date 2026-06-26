# Chess Overlay

A chess *teaching* overlay: it watches a chess board on your screen, runs the
position through a strong local engine (Stockfish), and draws the best moves —
circling the piece and its destination square with an arrow between them — in a
transparent, click-through window floating over the board.

> Intended for offline analysis, study, and play against bots/the computer.
> Using engine assistance in live rated games against people violates fair-play
> rules on chess.com, lichess, and most other sites.

## Status

| Phase | Feature | State |
|------:|---------|-------|
| 1 | Engine wrapper, click-through overlay, control panel, fixed + live engine modes | ✅ done |
| 2 | Screen capture + drag-calibrate ✅; self-calibrating piece recognition → FEN | 🔨 first cut (tune on real board) |
| 3 | Rules/temporal validation + game tracker + move list ✅; auto-track loop wired | 🔨 depends on vision |
| 4 | Opening names, click-into-a-move depth lines, board auto-detect | ⏳ |
| 5 | Packaging, global hotkeys, performance | ⏳ |

## How the "near-0% error" goal is met

Per-square computer vision is never perfectly reliable alone, so the engine
doubles as an error-corrector. The app holds an authoritative `chess.Board()`
and only **commits** a new camera reading when it (a) is stable across several
frames and (b) differs from the known position by exactly one **legal move**.
Animations, half-finished drags, premove highlights, and drawn arrows don't form
legal positions, so they're rejected automatically.

## Setup (Windows, Python 3.10+)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Get a Stockfish binary from <https://stockfishchess.org/download/>, and either:

- drop it in `engines\stockfish.exe`, **or**
- put it on your `PATH`, **or**
- set `engine_path` in `config.json`.

## Run

```powershell
python main.py
```

Opens the **control panel** (and the transparent overlay). From there:

1. **Monitor** — pick the screen your board is on (defaults to monitor 0).
2. **Calibrate board** — drag a box corner-to-corner over the board. Sets the
   overlay geometry + capture region and saves `debug/board.png`.
3. **Calibrate vision** — with the **starting position** on the board, click it
   once to learn your piece set (persisted). Then **Recognize now** reads the
   board into the FEN box.
4. **Engine mode** — *Fixed depth* (one strong search) or *Live* (streams and
   refines the candidate moves/arrows as it thinks deeper; **Stop** to halt).
5. **Analyze** — runs the FEN box through the engine and draws the top moves,
   color-coded, with evals. The FEN box stays editable.
6. **Auto-track game** — captures + recognizes on a timer, validates each frame
   against the rules of chess, and logs the moves (`1. e4 e5 …`). **Reset game**
   clears it.
7. **Show arrows** toggles the board UI; **White on bottom** sets orientation;
   engine depth / lines / threads / hash are adjustable and saved.

## Multi-monitor & DPI handling

Mixed per-monitor scaling is handled deliberately so overlays stay pixel-aligned:

* **One overlay window per monitor**, each covering exactly its screen — so a
  window never spans two DPI contexts and Windows never bitmap-stretches it.
* **Single coordinate contract:** everything internal is *global logical
  (device-independent) pixels*. The capture/CV layer is the only place that
  converts physical↔logical (`physical ÷ devicePixelRatio + screen origin`).
* The process is marked **per-monitor DPI aware** at startup so screen capture
  reports true physical pixels.

## Project layout

```
main.py            Phase 1 demo entry point
requirements.txt   dependencies
engines/           drop stockfish.exe here
src/
  config.py        persisted settings (config.json)
  engine.py        Stockfish (UCI) wrapper → ranked MoveSuggestions
  overlay.py       transparent click-through overlay (circles + arrows)
```
