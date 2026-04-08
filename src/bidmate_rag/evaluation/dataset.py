"""Evaluation dataset loading helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bidmate_rag.schema import EvalSample


def load_eval_samples(path: str | Path) -> list[EvalSample]:
    source = Path(path)
    if source.suffix == ".json":
        rows = json.loads(source.read_text(encoding="utf-8"))
        return [EvalSample.model_validate(row) for row in rows]
    if source.suffix == ".jsonl":
        return [
            EvalSample.model_validate(json.loads(line))
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if source.suffix == ".csv":
        frame = pd.read_csv(source)
    else:
        frame = pd.read_parquet(source)
    rows = frame.to_dict(orient="records")
    return [EvalSample.model_validate(row) for row in rows]
