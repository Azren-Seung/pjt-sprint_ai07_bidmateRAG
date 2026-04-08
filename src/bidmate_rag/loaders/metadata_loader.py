"""Metadata loading helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_metadata_frame(metadata_path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(metadata_path)
    frame["공고 번호"] = frame["공고 번호"].astype(str)
    return frame
