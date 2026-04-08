"""kordoc subprocess helper."""

from __future__ import annotations

import subprocess
from pathlib import Path

DEFAULT_KORDOC_BIN = Path.home() / ".npm-global" / "bin" / "kordoc"


def parse_with_kordoc(
    file_path: str | Path,
    kordoc_bin: str | Path | None = None,
    timeout: int = 120,
) -> dict:
    path = Path(file_path)
    binary = Path(kordoc_bin) if kordoc_bin else DEFAULT_KORDOC_BIN
    try:
        result = subprocess.run(
            [str(binary), str(path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        text = result.stdout.strip()
        return {
            "파일명": path.name,
            "파서": "kordoc",
            "텍스트": text,
            "글자수": len(text),
            "성공": bool(text),
            "에러": None if text else result.stderr.strip() or "empty-output",
        }
    except subprocess.TimeoutExpired:
        return {
            "파일명": path.name,
            "파서": "kordoc",
            "텍스트": "",
            "글자수": 0,
            "성공": False,
            "에러": "timeout",
        }
    except Exception as exc:  # pragma: no cover - system dependent
        return {
            "파일명": path.name,
            "파서": "kordoc",
            "텍스트": "",
            "글자수": 0,
            "성공": False,
            "에러": str(exc),
        }
