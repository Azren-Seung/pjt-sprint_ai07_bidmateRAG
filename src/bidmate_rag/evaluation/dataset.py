"""Evaluation dataset loading helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from bidmate_rag.schema import EvalSample

# Eval set version directories live under ``data/eval/`` (e.g. ``eval_v1``,
# ``eval_v2``). The CLI / Streamlit / scripts default to the highest-numbered
# version so that adding ``eval_v2/`` later "just works" without code edits.
EVAL_ROOT = Path("data/eval")
_EVAL_VERSION_PATTERN = re.compile(r"^eval_v(\d+)$")


def find_latest_eval_dir(root: Path | str = EVAL_ROOT) -> Path:
    """Return the highest-numbered ``eval_v*`` directory under ``root``.

    Falls back to ``root`` itself if no versioned directory exists yet
    (preserves legacy behavior for old checkouts that still keep CSVs at the
    top level).
    """
    root_path = Path(root)
    versions: list[tuple[int, Path]] = []
    if root_path.exists():
        for child in root_path.iterdir():
            if not child.is_dir():
                continue
            match = _EVAL_VERSION_PATTERN.match(child.name)
            if match:
                versions.append((int(match.group(1)), child))
    if versions:
        return max(versions, key=lambda item: item[0])[1]
    return root_path


def list_eval_csvs(root: Path | str = EVAL_ROOT) -> list[Path]:
    """List ``eval_batch_*.csv`` files inside the latest eval version dir."""
    return sorted(find_latest_eval_dir(root).glob("eval_batch_*.csv"))


# Columns from data/eval/eval_v*/eval_batch_*.csv that aren't part of EvalSample's
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
    `data/eval/eval_v*/eval_batch_*.csv` shape (`id, type, difficulty, question,
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
