"""Evaluation dataset loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from bidmate_rag.schema import EvalSample

# Columns from data/eval/eval_batch_*.csv that aren't part of EvalSample's
# top-level fields but should be preserved in `metadata` so downstream filters
# (e.g. --filter-type) can use them.
_METADATA_PASSTHROUGH_COLUMNS = (
    "type",
    "difficulty",
    "ground_truth_answer",
    "metadata_filter",
    "history",
)


def _coerce_json_field(value: Any) -> Any:
    """Best-effort JSON parse for CSV cells; passthrough if already parsed."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text in ("[]", "{}"):
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Map a raw eval-set row to EvalSample's schema.

    Accepts either the canonical EvalSample shape or the
    `data/eval/eval_batch_*.csv` shape (`id, type, difficulty, question,
    ground_truth_answer, ground_truth_docs, metadata_filter, history`).
    """
    # Already canonical — pass through unchanged.
    if "question_id" in row and "question" in row:
        return row

    normalized: dict[str, Any] = {}
    if "id" in row:
        normalized["question_id"] = str(row["id"])
    normalized["question"] = row.get("question", "")

    # ground_truth_docs is a JSON list of file titles in the project's CSVs.
    docs = _coerce_json_field(row.get("ground_truth_docs"))
    if isinstance(docs, list):
        normalized["expected_doc_titles"] = [str(d) for d in docs]

    metadata: dict[str, Any] = {}
    for col in _METADATA_PASSTHROUGH_COLUMNS:
        if col not in row:
            continue
        raw = row[col]
        if col in ("metadata_filter", "history"):
            parsed = _coerce_json_field(raw)
            if parsed is not None:
                metadata[col] = parsed
        else:
            try:
                if pd.isna(raw):
                    continue
            except (TypeError, ValueError):
                pass
            if raw is None or raw == "":
                continue
            metadata[col] = raw
    if metadata:
        normalized["metadata"] = metadata
    return normalized


def load_eval_samples(path: str | Path) -> list[EvalSample]:
    source = Path(path)
    if source.suffix == ".json":
        rows = json.loads(source.read_text(encoding="utf-8"))
    elif source.suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        if source.suffix == ".csv":
            frame = pd.read_csv(source, encoding="utf-8-sig")
        else:
            frame = pd.read_parquet(source)
        rows = frame.to_dict(orient="records")
    return [EvalSample.model_validate(_normalize_row(row)) for row in rows]
