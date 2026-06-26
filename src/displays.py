"""Win32 monitor geometry in PHYSICAL pixels, keyed by GDI device name.

Qt positions windows in logical pixels and handles per-monitor scaling for the
overlay, but screen capture works in physical pixels. This module is the bridge:
it maps a monitor's GDI device name (which equals ``QScreen.name()`` on Windows)
to its physical rectangle and scale, so capture regions are computed exactly.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass


@dataclass
class MonitorInfo:
    name: str          # GDI device name, e.g. r"\\.\DISPLAY1"
    left: int          # physical px
    top: int
    width: int
    height: int
    scale: float       # 1.0, 1.25, 1.5, ...


class _MONITORINFOEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT),
        ("dwFlags", wintypes.DWORD),
        ("szDevice", ctypes.c_wchar * 32),
    ]


_MONITORENUMPROC = ctypes.WINFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(wintypes.RECT), ctypes.c_void_p)


def enumerate_monitors() -> dict[str, MonitorInfo]:
    """Return {device_name: MonitorInfo} for all monitors, in physical pixels."""
    user32 = ctypes.windll.user32
    try:
        shcore = ctypes.windll.shcore
    except OSError:
        shcore = None

    results: dict[str, MonitorInfo] = {}

    def _callback(hmon, _hdc, _lprc, _lparam):
        mi = _MONITORINFOEXW()
        mi.cbSize = ctypes.sizeof(_MONITORINFOEXW)
        if not user32.GetMonitorInfoW(hmon, ctypes.byref(mi)):
            return 1
        r = mi.rcMonitor
        scale = 1.0
        if shcore is not None:
            dpi_x, dpi_y = wintypes.UINT(), wintypes.UINT()
            try:  # GetDpiForMonitor(hmon, MDT_EFFECTIVE_DPI=0, &x, &y)
                if shcore.GetDpiForMonitor(hmon, 0, ctypes.byref(dpi_x),
                                           ctypes.byref(dpi_y)) == 0:
                    scale = dpi_x.value / 96.0
            except Exception:
                pass
        results[mi.szDevice] = MonitorInfo(
            name=mi.szDevice, left=r.left, top=r.top,
            width=r.right - r.left, height=r.bottom - r.top, scale=scale)
        return 1

    user32.EnumDisplayMonitors(None, None, _MONITORENUMPROC(_callback), 0)
    return results


def _read_monitor(hmon) -> MonitorInfo:
    user32 = ctypes.windll.user32
    try:
        shcore = ctypes.windll.shcore
    except OSError:
        shcore = None
    mi = _MONITORINFOEXW()
    mi.cbSize = ctypes.sizeof(_MONITORINFOEXW)
    user32.GetMonitorInfoW(hmon, ctypes.byref(mi))
    r = mi.rcMonitor
    scale = 1.0
    if shcore is not None:
        dpi_x, dpi_y = wintypes.UINT(), wintypes.UINT()
        try:
            if shcore.GetDpiForMonitor(hmon, 0, ctypes.byref(dpi_x),
                                       ctypes.byref(dpi_y)) == 0:
                scale = dpi_x.value / 96.0
        except Exception:
            pass
    return MonitorInfo(mi.szDevice, r.left, r.top,
                       r.right - r.left, r.bottom - r.top, scale)


def monitor_from_hwnd(hwnd: int) -> MonitorInfo:
    """Physical geometry + scale of the monitor a window is on.

    Robust because the window itself identifies the monitor — no name or
    position matching (Qt's QScreen.name() is the model name on Windows, not
    the GDI device name, so name matching is unreliable)."""
    user32 = ctypes.windll.user32
    user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
    user32.MonitorFromWindow.restype = ctypes.c_void_p
    MONITOR_DEFAULTTONEAREST = 2
    hmon = user32.MonitorFromWindow(wintypes.HWND(hwnd), MONITOR_DEFAULTTONEAREST)
    return _read_monitor(hmon)
