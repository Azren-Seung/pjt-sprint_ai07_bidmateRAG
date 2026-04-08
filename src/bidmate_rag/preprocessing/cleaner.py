"""Text cleaning helpers reproduced from the notebook baseline."""

from __future__ import annotations

import re


def clean_br_tags(text: str) -> str:
    return text.replace("<br>", "\n")


def clean_duplicate_table_cells(text: str) -> str:
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
    text = re.sub(r"[\ue000-\uf8ff]", "", text)
    return text.replace("\u00a0", " ")


def clean_kordoc_warnings(text: str) -> str:
    return "\n".join(line for line in text.split("\n") if not line.startswith("Warning: "))


def clean_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = clean_kordoc_warnings(text)
    text = clean_br_tags(text)
    text = clean_duplicate_table_cells(text)
    text = clean_broken_chars(text)
    text = clean_whitespace(text)
    return text.strip()
