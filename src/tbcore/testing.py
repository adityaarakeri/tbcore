"""Helpers for gating runtime-dependent integration tests."""

from __future__ import annotations

import shutil
import subprocess


def docker_available() -> bool:
    """True when a container runtime CLI exists AND its daemon responds."""
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "ok"], capture_output=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0
