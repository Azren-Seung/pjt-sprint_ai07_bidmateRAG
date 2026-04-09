"""Capture git commit/branch info for experiment metadata."""

from __future__ import annotations

import logging
import subprocess
from typing import Any

logger = logging.getLogger(__name__)


def capture_git_info() -> dict[str, Any]:
    """Return current git commit, branch, and dirty flag.

    Returns ``{"commit": "unknown", "branch": "unknown", "dirty": False}`` if
    we are not in a git repo or if git is not installed.
    """
    try:
        commit = _git("rev-parse", "HEAD")
        branch = _git("rev-parse", "--abbrev-ref", "HEAD")
        status = _git("status", "--porcelain")
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning("Could not capture git info: %s", exc)
        return {"commit": "unknown", "branch": "unknown", "dirty": False}
    return {
        "commit": commit,
        "commit_short": commit[:7],
        "branch": branch,
        "dirty": bool(status),
    }


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()
