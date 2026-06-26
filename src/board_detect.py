"""Automatic board alignment — a second CALIBRATION mode.

Classical OpenCV/numpy only (no ML). Finds the exact 8x8 board rectangle inside a
roughly-selected region, so a hand-drawn box can be snapped to the board to the
pixel. This refines only the board *region*; the piece recognition is untouched.

How: a chessboard's square boundaries form an evenly-spaced grid. A square-to-
square boundary (the colours alternate) shows BOTH a positive and a negative
gradient stacked along the line, so multiplying the per-line totals of the two
signs makes real grid lines spike while lone edges (pieces, page/UI) cancel. We
then fit the best 8-cell grid (period + phase) to that 1-D line signal on each
axis and return the board's tight bounds.
"""
from __future__ import annotations

import numpy as np


def _line_signal(gpos: np.ndarray, gneg: np.ndarray, axis: int) -> np.ndarray:
    """Per-position grid-line strength: product of the summed +grad and -grad."""
    return gpos.sum(axis=axis) * gneg.sum(axis=axis)


def _parabolic(y0: float, y1: float, y2: float) -> float:
    """Sub-sample peak offset (in samples) from three points around a maximum."""
    d = y0 - 2 * y1 + y2
    return 0.5 * (y0 - y2) / d if abs(d) > 1e-12 else 0.0


def _fit_grid(strength: np.ndarray) -> tuple[int, int] | None:
    """Fit the 8-cell grid to a 1-D line signal; return (lo, hi) board edges.

    Period from autocorrelation (averages over all 8 squares, so it is not biased
    by any one mislocated line, e.g. piece edges), phase from a 9-tooth comb."""
    n = len(strength)
    if n < 32:
        return None
    s = strength.astype(np.float64)
    if s.max() < 1e-9:
        return None
    s = np.convolve(s / s.max(), np.array([0.25, 0.5, 0.25]), mode="same")

    lo, hi = max(4, int(n / 12.5)), int(n / 7.5)      # square ≈ 1/8 of the region
    if hi <= lo + 1:
        return None
    z = s - s.mean()
    ac = np.correlate(z, z, mode="full")[n - 1:]      # ac[lag], lag >= 0
    k = lo + int(np.argmax(ac[lo:hi]))
    step = k + (_parabolic(ac[k - 1], ac[k], ac[k + 1]) if 0 < k < n - 1 else 0.0)
    if not (n / 12.5 <= step <= n / 7.5):
        return None

    xs = np.arange(n)
    offs = np.arange(-1.5, max(0.0, n - 8 * step) + 1.5 + 1e-6, 0.25)
    scores = np.array([np.interp(off + step * np.arange(9), xs, s, left=0.0, right=0.0).sum()
                       for off in offs])
    bi = int(np.argmax(scores))
    off = offs[bi]
    if 0 < bi < len(offs) - 1:
        off += 0.25 * _parabolic(scores[bi - 1], scores[bi], scores[bi + 1])

    e0 = int(round(max(0, off)))
    e1 = int(round(min(n, off + 8 * step)))
    return (e0, e1) if e1 - e0 >= n * 0.4 else None


def find_board(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    """Find the board rectangle (x, y, w, h) in a grayscale image, or None."""
    g = gray.astype(np.float32)
    gy, gx = np.gradient(g)
    gxp, gxn = np.clip(gx, 0, None), np.clip(-gx, 0, None)
    gyp, gyn = np.clip(gy, 0, None), np.clip(-gy, 0, None)
    xs = _fit_grid(_line_signal(gxp, gxn, axis=0))     # vertical lines -> x extent
    ys = _fit_grid(_line_signal(gyp, gyn, axis=1))     # horizontal lines -> y extent
    if xs is None or ys is None:
        return None
    x0, x1 = xs
    y0, y1 = ys
    return (x0, y0, x1 - x0, y1 - y0)
