"""Structured metadata helpers used by retrieval and UI."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


class MetadataStore:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.fillna("")
        self.agency_list = (
            sorted(self.frame["발주 기관"].astype(str).unique().tolist())
            if "발주 기관" in self.frame.columns
            else []
        )

    @classmethod
    def from_parquet(cls, path: str | Path) -> "MetadataStore":
        return cls(pd.read_parquet(path))

    def find_relevant_docs(self, query: str, top_n: int = 3) -> list[str]:
        tokens = [token for token in re.split(r"\s+", query) if len(token) >= 2]
        if not tokens:
            return []
        scored = self.frame.copy()
        haystack = (
            scored.get("사업 요약", "").astype(str) + " " + scored.get("사업명", "").astype(str)
        )
        scored["_score"] = [sum(token in text for token in tokens) for text in haystack]
        top = scored.sort_values("_score", ascending=False).head(top_n)
        return [filename for filename in top.get("파일명", []).tolist() if filename]
