"""Saved piece 'themes' — persisted vision templates you can switch between without
recalibrating from the start position every time.

A theme is just a serialized ``VisionModel`` (the learned empty squares + the 12
piece templates) under ``themes/<name>.npz``. Because recognition normalises every
square to a fixed size before matching, a theme is portable across board PIXEL SIZES
— so the same theme works whether the board is large or small on screen.
"""
from __future__ import annotations

import re
from pathlib import Path

from src.vision import VisionModel

THEMES_DIR = Path(__file__).resolve().parent.parent / "themes"


def _safe(name: str) -> str:
    """A filesystem-safe theme name (and the on-disk stem)."""
    s = re.sub(r"[^A-Za-z0-9 _-]", "", name).strip()
    return s[:48] or "theme"


def theme_path(name: str) -> Path:
    return THEMES_DIR / f"{_safe(name)}.npz"


def list_themes() -> list[str]:
    if not THEMES_DIR.is_dir():
        return []
    return sorted((p.stem for p in THEMES_DIR.glob("*.npz")), key=str.lower)


def save_theme(name: str, model: VisionModel) -> str:
    """Save ``model`` as a theme; returns the sanitized name actually used."""
    THEMES_DIR.mkdir(exist_ok=True)
    safe = _safe(name)
    model.save(THEMES_DIR / f"{safe}.npz")
    return safe


def load_theme(name: str) -> VisionModel:
    return VisionModel.load(theme_path(name))


def delete_theme(name: str) -> None:
    p = theme_path(name)
    if p.exists():
        p.unlink()
