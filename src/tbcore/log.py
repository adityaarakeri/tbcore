"""Opt-in console logging with level colors.

tbcore logs under the ``tbcore`` logger and installs only a ``NullHandler``, so importing the
library never prints. Call :func:`configure_logging` to see the output. Records carry model
names, token counts and latency; never prompts, headers or API keys.
"""

from __future__ import annotations

import copy
import logging
import os
import sys
from typing import TextIO

LOGGER_NAME = "tbcore"
_RESET = "\x1b[0m"
_LEVEL_COLORS = {
    logging.DEBUG: "\x1b[36m",  # cyan
    logging.INFO: "\x1b[32m",  # green
    logging.WARNING: "\x1b[33m",  # yellow
    logging.ERROR: "\x1b[31m",  # red
    logging.CRITICAL: "\x1b[1;41m",  # bold on red
}
_DIM = "\x1b[2m"
_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

logging.getLogger(LOGGER_NAME).addHandler(logging.NullHandler())
_installed: logging.Handler | None = None


class ColorFormatter(logging.Formatter):
    """Colors the timestamp (dim) and level name; the message text is left untouched."""

    def __init__(self, *, color: bool) -> None:
        super().__init__(_FORMAT)
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        if not self.color:
            return super().format(record)
        # Copy so other handlers still see the plain level name.
        styled = copy.copy(record)
        paint = _LEVEL_COLORS.get(record.levelno, "")
        styled.levelname = f"{paint}{record.levelname}{_RESET}"
        text = super().format(styled)
        stamp = self.formatTime(record, self.datefmt)
        return text.replace(stamp, f"{_DIM}{stamp}{_RESET}", 1)


def configure_logging(
    level: int | str = "INFO", *, color: bool | None = None, stream: TextIO | None = None
) -> logging.Handler:
    """Send ``tbcore`` logs to ``stream`` (default stderr); safe to call again to reconfigure.

    ``color=None`` decides automatically: ``NO_COLOR`` disables, ``FORCE_COLOR`` enables, and
    otherwise color is used only on an interactive terminal that supports ANSI codes. The
    ``tbcore`` logger stops propagating to the root logger so lines are not printed twice.
    """
    global _installed
    target = stream if stream is not None else sys.stderr
    logger = logging.getLogger(LOGGER_NAME)
    if _installed is not None:
        logger.removeHandler(_installed)
    handler = logging.StreamHandler(target)
    handler.setFormatter(ColorFormatter(color=_use_color(target, color)))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    _installed = handler
    return handler


def _use_color(stream: TextIO, requested: bool | None) -> bool:
    if requested is not None:
        return requested
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR", "0") not in ("", "0"):
        return True
    if os.environ.get("TERM") == "dumb" or not stream.isatty():
        return False
    return _ansi_supported(stream)


def _ansi_supported(stream: TextIO) -> bool:
    supported = True
    # No early return: mypy's warn_unreachable would flag the Windows-only code on other
    # platforms (and the fallthrough on Windows).
    if sys.platform == "win32":
        # Legacy Windows consoles need virtual-terminal processing switched on.
        import ctypes
        import msvcrt

        kernel32 = ctypes.windll.kernel32
        handle = msvcrt.get_osfhandle(stream.fileno())
        mode = ctypes.c_uint32()
        supported = bool(kernel32.GetConsoleMode(handle, ctypes.byref(mode))) and bool(
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        )
    return supported
