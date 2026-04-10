"""Evaluation dataset loading helpers."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from bidmate_rag.schema import EvalSample

logger = logging.getLogger(__name__)


# 평가셋 CSV의 metadata_filter 컬럼은 영문 키를 사용하지만, ChromaDB에 저장된
# 청크 메타데이터는 한국어 키(공백 포함)를 사용합니다. 평가 시 retrieval에
# 적용하려면 영문 → 한국어 매핑이 필요합니다.
EVAL_FILTER_KEY_MAP: dict[str, str] = {
    "agency": "발주 기관",
    "institution": "발주 기관",
    "project": "사업명",
    "domain": "사업도메인",
    "agency_type": "기관유형",
    "tech_stack": "기술스택",
    "year": "공개연도",
    "budget": "사업 금액",
}

# `공개연도`는 ChromaDB에 int로 저장되므로 문자열을 변환해야 매칭됩니다.
_NUMERIC_FILTER_KEYS = {"공개연도", "사업 금액"}


def normalize_metadata_filter(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    """평가셋 metadata_filter 영문 키를 ChromaDB metadata 키로 정규화.

    예::

        {"agency": "한국가스공사", "year": "2024"}
        → {"발주 기관": "한국가스공사", "공개연도": 2024}

    매핑에 없는 키는 그대로 통과시키고 한 번만 warning 로그.
    """
    if not raw or not isinstance(raw, dict):
        return None
    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        target_key = EVAL_FILTER_KEY_MAP.get(key, key)
        if target_key not in EVAL_FILTER_KEY_MAP.values() and target_key == key:
            logger.warning(
                "Unknown metadata_filter key %r in eval sample — passing through as-is",
                key,
            )
        # 숫자형 필드는 ChromaDB가 int를 기대하므로 변환
        if target_key in _NUMERIC_FILTER_KEYS and isinstance(value, str):
            try:
                value = int(value)
            except (TypeError, ValueError):
                pass
        normalized[target_key] = value
    return normalized or None

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


PROCESSED_ROOT = Path("data/processed")


def find_latest_metadata_path(
    root: Path | str = PROCESSED_ROOT,
) -> Path:
    """가장 최신 ``cleaned_documents.parquet`` 경로를 반환.

    우선순위:
      1. ``data/processed/{exp_name}/cleaned_documents.parquet`` 중 mtime이 가장
         최근인 것 (실험별 ingest 결과)
      2. ``data/processed/cleaned_documents.parquet`` (top-level legacy)

    Streamlit과 다른 caller가 실험별 metadata를 자동으로 사용하도록 하기 위해
    추가. CLI는 이미 ``_resolve_metadata_path`` (runtime 기반)를 사용하지만,
    Streamlit은 runtime 컨텍스트가 부족해서 mtime 기반 휴리스틱이 필요.
    """
    root_path = Path(root)
    if not root_path.exists():
        return root_path / "cleaned_documents.parquet"

    candidates: list[tuple[float, Path]] = []
    for sub in root_path.iterdir():
        if not sub.is_dir():
            continue
        candidate = sub / "cleaned_documents.parquet"
        if candidate.exists():
            candidates.append((candidate.stat().st_mtime, candidate))

    if candidates:
        return max(candidates, key=lambda item: item[0])[1]

    return root_path / "cleaned_documents.parquet"


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
        if col == "metadata_filter":
            parsed = _coerce_json_field(raw)
            normalized_filter = normalize_metadata_filter(
                parsed if isinstance(parsed, dict) else None
            )
            if normalized_filter:
                metadata["metadata_filter"] = normalized_filter
        elif col == "history":
            parsed = _coerce_json_field(raw)
            if isinstance(parsed, list) and parsed:
                metadata["history"] = parsed
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
