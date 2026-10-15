"""Helpers for gating runtime-dependent integration tests."""

from __future__ import annotations

import shutil
import subprocess


def docker_available() -> bool:
    """True when a container runtime CLI exists AND its daemon runs Linux containers.

    The bundled images are Linux-only, so a daemon in Windows-container mode (the default
    on GitHub's Windows runners) counts as unavailable.
    """
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.OSType}}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and result.stdout.strip() == "linux"
