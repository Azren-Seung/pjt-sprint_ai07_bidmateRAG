"""HWP loading logic."""

from __future__ import annotations

from pathlib import Path

from bidmate_rag.loaders.kordoc_loader import parse_with_kordoc


def parse_hwp(file_path: str | Path, kordoc_bin: str | Path | None = None) -> dict:
    result = parse_with_kordoc(file_path, kordoc_bin=kordoc_bin)
    if result["성공"]:
        return result
    path = Path(file_path)
    try:
        from hwp_hwpx_parser import HWP5Reader

        reader = HWP5Reader(str(path))
        text = reader.extract_text()
        reader.close()
        return {
            "파일명": path.name,
            "파서": "hwp-hwpx-parser",
            "텍스트": text,
            "글자수": len(text),
            "성공": bool(text),
            "에러": None,
        }
    except Exception as exc:  # pragma: no cover - fallback path
        return {
            "파일명": path.name,
            "파서": "hwp-hwpx-parser",
            "텍스트": "",
            "글자수": 0,
            "성공": False,
            "에러": str(exc),
        }
