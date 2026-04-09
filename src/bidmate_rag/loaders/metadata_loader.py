"""Metadata loading helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_metadata_frame(metadata_path: str | Path) -> pd.DataFrame:
    """메타데이터 CSV를 DataFrame으로 로드

    Args:
        metadata_path: 메타데이터 CSV 파일 경로.

    Returns:
        공고 번호가 문자열로 변환된 DataFrame.
    """
    frame = pd.read_csv(metadata_path)
    frame["공고 번호"] = frame["공고 번호"].astype(str)
    return frame
