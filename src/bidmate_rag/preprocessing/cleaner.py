"""Text cleaning helpers reproduced from the notebook baseline."""

from __future__ import annotations

import re


def clean_br_tags(text: str) -> str:
    """HTML <br> 태그를 공백으로 변환

    Args:
        text: 원본 텍스트.

    Returns:
        <br> 태그가 공백으로 치환된 텍스트.
    """
    return text.replace("<br>", " ")


def clean_duplicate_table_cells(text: str) -> str:
    """테이블에서 동일 값이 반복되는 셀을 하나로 축소

    Args:
        text: 원본 텍스트.

    Returns:
        중복 셀이 제거된 텍스트.
    """
    lines = text.split("\n")
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and not re.match(r"^\|[\s\-:|]+\|$", stripped):
            cells = [cell.strip() for cell in stripped.split("|") if cell.strip()]
            if len(cells) >= 2 and len(set(cells)) == 1 and cells[0]:
                result.append(f"| {cells[0]} |")
                continue
        result.append(line)
    return "\n".join(result)


def clean_whitespace(text: str) -> str:
    """연속 빈 줄과 중복 공백을 정리

    Args:
        text: 원본 텍스트.

    Returns:
        공백이 정규화된 텍스트.
    """
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = text.split("\n")
    normalized: list[str] = []
    for line in lines:
        if line.strip().startswith("|"):
            normalized.append(line)
        else:
            normalized.append(re.sub(r" {2,}", " ", line))
    return "\n".join(normalized)


def clean_broken_chars(text: str) -> str:
    """깨진 유니코드 문자와 특수 공백을 제거

    Args:
        text: 원본 텍스트.

    Returns:
        깨진 문자가 제거된 텍스트.
    """
    text = re.sub(r"[\ue000-\uf8ff]", "", text)
    return text.replace("\u00a0", " ")


def clean_kordoc_warnings(text: str) -> str:
    """kordoc 파싱 시 생성된 Warning 줄을 제거

    Args:
        text: 원본 텍스트.

    Returns:
        Warning 줄이 제거된 텍스트.
    """
    return "\n".join(line for line in text.split("\n") if not line.startswith("Warning: "))


def clean_text(text: str) -> str:
    """전체 정제 파이프라인을 순서대로 실행

    Args:
        text: 원본 텍스트.

    Returns:
        정제된 텍스트.
    """
    if not text or not isinstance(text, str):
        return ""
    text = clean_kordoc_warnings(text)
    text = clean_br_tags(text)
    text = clean_duplicate_table_cells(text)
    text = clean_broken_chars(text)
    text = clean_whitespace(text)
    return text.strip()
