"""
Utilities to export Plotly figures reliably on platforms where Kaleido/Chrome may be missing.
- Try to ensure a Chrome/Chromium binary using plotly_get_chrome if available.
- Try static PNG export via Plotly; return None if it fails so callers can fallback.
"""
from __future__ import annotations

import io
import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Optional

try:
    import plotly.io as pio
except Exception:  # pragma: no cover
    pio = None  # type: ignore


def _default_chrome_path() -> str:
    home = Path.home()
    if sys.platform.startswith("linux"):
        return str(home / ".plotly" / "chrome" / "chrome")
    if sys.platform == "win32":
        return str(home / ".plotly" / "chrome" / "chrome.exe")
    if sys.platform == "darwin":
        return str(home / ".plotly" / "chrome" / "Chromium.app" / "Contents" / "MacOS" / "Chromium")
    return ""


def ensure_plotly_chrome() -> bool:
    """Ensure a Chrome/Chromium binary is available for Plotly Kaleido.
    Returns True if a usable Chrome path is set or discovered.
    """
    # Already set and exists
    env_path = os.getenv("PLOTLY_CHROME_PATH", "")
    if env_path and Path(env_path).exists():
        return True

    # Look in PATH
    for exe in ("google-chrome", "chrome", "chromium", "chromium-browser"):
        found = shutil.which(exe)
        if found:
            os.environ["PLOTLY_CHROME_PATH"] = found
            return True

    # Try installing a portable Chrome via module, if present (best-effort)
    try:
        subprocess.run([sys.executable, "-m", "plotly_get_chrome", "-y"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        chrome_path = _default_chrome_path()
        if chrome_path and Path(chrome_path).exists():
            os.environ["PLOTLY_CHROME_PATH"] = chrome_path
            return True
    except Exception:
        pass

    return False


def fig_to_png_bytes_plotly(fig, *, width: int = 1200, height: int = 1800, scale: int = 2) -> Optional[bytes]:
    """Attempt to export a Plotly figure to PNG bytes.
    Returns bytes on success, or None on failure (so callers can fallback) without printing warnings to the UI/logs.
    """
    if pio is None:
        return None
    try:
        # Ensure Chrome first (best effort)
        ensure_plotly_chrome()
        return pio.to_image(fig, format="png", width=width, height=height, scale=scale)
    except Exception:
        # Do not print noisy Kaleido/Chrome warnings; simply signal failure
        return None
