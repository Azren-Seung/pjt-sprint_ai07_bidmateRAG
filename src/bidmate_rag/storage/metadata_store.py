"""Structured metadata helpers used by retrieval and UI.

파싱된 문서의 메타데이터(사업명, 발주기관 등)를 parquet에서 읽어
검색 및 UI에서 활용할 수 있도록 구조화한다.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


class MetadataStore:
    """문서 메타데이터 저장소. parquet 기반으로 발주기관 목록 조회, 키워드 검색을 제공한다."""

    def __init__(self, frame: pd.DataFrame) -> None:
        """MetadataStore를 초기화한다.

        Args:
            frame: 문서 메타데이터가 담긴 DataFrame.
        """
        # NaN을 빈 문자열로 치환
        self.frame = frame.fillna("")
        # 발주 기관 목록 추출 (UI 필터 드롭다운 등에 사용)
        self.agency_list = (
            sorted(self.frame["발주 기관"].astype(str).unique().tolist())
            if "발주 기관" in self.frame.columns
            else []
        )

    @classmethod
    def from_parquet(cls, path: str | Path) -> "MetadataStore":
        """parquet 파일에서 MetadataStore를 생성한다.

        Args:
            path: parquet 파일 경로.

        Returns:
            MetadataStore 인스턴스.
        """
        return cls(pd.read_parquet(path))

    def find_relevant_docs(self, query: str, top_n: int = 3) -> list[str]:
        """질문과 관련된 문서 파일명을 키워드 매칭으로 찾는다.

        Args:
            query: 사용자 질문 문자열.
            top_n: 반환할 최대 문서 수.

        Returns:
            관련도 높은 순으로 정렬된 파일명 리스트.
        """
        # 질문을 공백으로 분리, 2글자 이상 토큰만 사용
        tokens = [token for token in re.split(r"\s+", query) if len(token) >= 2]
        if not tokens:
            return []

        scored = self.frame.copy()
        # 사업 요약 + 사업명을 합쳐서 검색 대상 텍스트 생성
        haystack = (
            scored.get("사업 요약", "").astype(str) + " " + scored.get("사업명", "").astype(str)
        )
        # 각 문서에 대해 매칭되는 토큰 수를 점수로 계산
        scored["_score"] = [sum(token in text for token in tokens) for text in haystack]
        # 점수 높은 순으로 정렬 후 상위 top_n개 반환
        top = scored.sort_values("_score", ascending=False).head(top_n)
        return [filename for filename in top.get("파일명", []).tolist() if filename]
