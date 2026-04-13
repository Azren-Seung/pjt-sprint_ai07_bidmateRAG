"""Shared helpers for rendering retrieved chunks into LLM context blocks."""

from __future__ import annotations

import math
import re
from numbers import Real

from bidmate_rag.schema import RetrievedChunk

_MISSING_STRINGS = {
    "",
    "nan",
    "none",
    "null",
    "undefined",
    "<na>",
    "n/a",
    "na",
    "-",
}

_SOURCE_KEYS = ("사업명", "발주 기관", "파일명")
_DETAIL_KEYS = ("사업 금액", "공개연도", "기관유형", "사업도메인")


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _MISSING_STRINGS
    if isinstance(value, bool):
        return False
    if isinstance(value, Real):
        try:
            return math.isnan(float(value))
        except (TypeError, ValueError):
            return False
    value_str = str(value).strip().lower()
    return value_str in _MISSING_STRINGS


def _clean_text(value: object) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    return text if text else None


def _format_won(value: object) -> str | None:
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return f"{value:,}원"
    if isinstance(value, float):
        number = int(value)
        return f"{number:,}원"
    raw_text = str(value).strip()
    if not raw_text:
        return None
    normalized = raw_text.replace(",", "").replace("원", "")
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", normalized):
        return raw_text
    number = int(float(normalized))
    return f"{number:,}원"


def _format_metadata_line(key: str, value: object) -> str | None:
    if key == "사업 금액":
        formatted = _format_won(value)
    else:
        formatted = _clean_text(value)
    if formatted is None:
        return None
    return f"{key}={formatted}"


def _build_chunk_header(metadata: dict[str, object]) -> str:
    source_values: list[str] = []
    for key in _SOURCE_KEYS:
        value = _clean_text(metadata.get(key))
        if value is not None:
            source_values.append(value)

    lines: list[str] = []
    if source_values:
        lines.append(f"[출처: {' | '.join(source_values)}]")

    for key in _DETAIL_KEYS:
        line = _format_metadata_line(key, metadata.get(key))
        if line is not None:
            lines.append(line)

    return "\n".join(lines)


def _build_chunk_block(chunk: RetrievedChunk) -> str:
    header = _build_chunk_header(chunk.chunk.metadata)
    text = chunk.chunk.text
    if header and text:
        return f"{header}\n{text}"
    if header:
        return header
    return text


def build_context_block(chunks: list[RetrievedChunk], max_chars: int = 8000) -> str:
    """Render retrieved chunks into a metadata-aware context block."""

    if max_chars <= 0:
        return ""

    parts: list[str] = []
    total_chars = 0
    separator = "\n\n---\n\n"

    for chunk in chunks:
        block = _build_chunk_block(chunk)
        if not block:
            continue
        candidate = block if not parts else f"{separator}{block}"
        if total_chars + len(candidate) > max_chars:
            break
        parts.append(block)
        total_chars += len(candidate)

    return separator.join(parts)
