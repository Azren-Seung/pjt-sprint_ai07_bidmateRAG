"""평가 데이터셋 로딩 헬퍼."""

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


def _extract_agencies_from_question(
    question: str,
    agency_list: list[str],
) -> list[str]:
    """질문 텍스트에서 발주 기관명을 추출한다.

    Args:
        question: 평가 질문 문자열.
        agency_list: 전체 발주기관 목록.

    Returns:
        매칭된 기관명 리스트.
    """
    matched: list[str] = []
    for agency in agency_list:
        # "(주)", "㈜" 제거 후 전체/앞6자/앞4자로 질문과 매칭 시도
        short = agency.replace("(주)", "").replace("㈜", "").strip()
        for part in [short, short[:6], short[:4]]:
            if len(part) >= 3 and part in question:
                if agency not in matched:
                    matched.append(agency)
                break
    return matched


def normalize_metadata_filter(
    raw: dict[str, Any] | None,
    question: str = "",
    agency_list: list[str] | None = None,
) -> dict[str, Any] | None:
    """평가셋 metadata_filter의 영문 키를 ChromaDB 한국어 키로 정규화한다.

    Args:
        raw: 평가셋의 metadata_filter 딕셔너리 (영문 키).
        question: 평가 질문 문자열 ("다중" 처리 시 기관명 추출에 사용).
        agency_list: 전체 발주기관 목록 ("다중" 처리 시 매칭에 사용).

    Returns:
        ChromaDB where 절에 사용할 정규화된 필터 딕셔너리, 또는 None.
    """
    if not raw or not isinstance(raw, dict):
        return None
    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        # 영문 키 → 한국어 키 변환 (예: "agency" → "발주 기관")
        target_key = EVAL_FILTER_KEY_MAP.get(key, key)
        if target_key not in EVAL_FILTER_KEY_MAP.values() and target_key == key:
            logger.warning(
                "Unknown metadata_filter key %r in eval sample — passing through as-is",
                key,
            )
        # "다중" → 질문에서 기관명 추출 후 $in 필터로 변환
        # 예: {"agency": "다중"} → {"발주 기관": {"$in": ["한국가스공사", "고려대학교"]}}
        if value == "다중" and target_key == "발주 기관" and agency_list:
            agencies = _extract_agencies_from_question(question, agency_list)
            if agencies:
                normalized[target_key] = {"$in": agencies}
            # 기관을 못 찾으면 필터 없이 전체 검색
            continue
        # 숫자형 필드는 ChromaDB가 int를 기대하므로 변환
        if target_key in _NUMERIC_FILTER_KEYS and isinstance(value, str):
            try:
                value = int(value)
            except (TypeError, ValueError):
                pass
        normalized[target_key] = value
    return normalized or None


# 평가셋 버전 디렉터리는 data/eval/ 아래에 eval_v1, eval_v2 등으로 존재.
# CLI / Streamlit은 가장 높은 버전 번호를 자동으로 사용한다.
EVAL_ROOT = Path("data/eval")
_EVAL_VERSION_PATTERN = re.compile(r"^eval_v(\d+)$")


def find_latest_eval_dir(root: Path | str = EVAL_ROOT) -> Path:
    """가장 최신 eval_v* 디렉터리 경로를 반환한다.

    Args:
        root: 평가셋 루트 디렉터리.

    Returns:
        가장 높은 버전 번호의 eval_v* 디렉터리 경로.
    """
    root_path = Path(root)
    versions: list[tuple[int, Path]] = []
    if root_path.exists():
        for child in root_path.iterdir():
            if not child.is_dir():
                continue
            # eval_v1, eval_v2 등에서 버전 번호 추출
            match = _EVAL_VERSION_PATTERN.match(child.name)
            if match:
                versions.append((int(match.group(1)), child))
    # 가장 높은 버전 반환, 없으면 루트 자체 반환 (레거시 호환)
    if versions:
        return max(versions, key=lambda item: item[0])[1]
    return root_path


def list_eval_csvs(root: Path | str = EVAL_ROOT) -> list[Path]:
    """최신 eval 디렉터리 내의 eval_batch_*.csv 파일 목록을 반환한다.

    Args:
        root: 평가셋 루트 디렉터리.

    Returns:
        정렬된 CSV 파일 경로 리스트.
    """
    return sorted(find_latest_eval_dir(root).glob("eval_batch_*.csv"))


PROCESSED_ROOT = Path("data/processed")


def find_latest_metadata_path(
    root: Path | str = PROCESSED_ROOT,
) -> Path:
    """가장 최신 cleaned_documents.parquet 경로를 반환한다.

    Args:
        root: 처리된 데이터 루트 디렉터리.

    Returns:
        가장 최신 cleaned_documents.parquet 파일 경로.
    """
    root_path = Path(root)
    if not root_path.exists():
        return root_path / "cleaned_documents.parquet"

    # 실험별 sub-dir에서 mtime이 가장 최근인 parquet 탐색
    candidates: list[tuple[float, Path]] = []
    for sub in root_path.iterdir():
        if not sub.is_dir():
            continue
        candidate = sub / "cleaned_documents.parquet"
        if candidate.exists():
            candidates.append((candidate.stat().st_mtime, candidate))

    # 실험별 파일이 있으면 최신 것 반환, 없으면 공용 경로 반환 (레거시 폴백)
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]

    return root_path / "cleaned_documents.parquet"


# EvalSample의 top-level 필드가 아니지만 metadata에 보존해야 하는 CSV 컬럼들.
# 다운스트림 필터(예: --filter-type)에서 사용된다.
_METADATA_PASSTHROUGH_COLUMNS = (
    "type",
    "difficulty",
    "ground_truth_answer",
    "metadata_filter",
    "history",
)


def _coerce_json_field(value: Any) -> Any:
    """CSV 셀 값을 JSON으로 파싱한다. 이미 파싱된 값은 그대로 반환.

    Args:
        value: CSV 셀 값 (문자열, dict, list, None 등).

    Returns:
        파싱된 Python 객체 또는 원본 값.
    """
    if value is None:
        return None
    # 이미 파싱된 dict/list는 그대로 반환
    if isinstance(value, (dict, list)):
        return value
    # NaN 체크
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    # 빈 문자열이나 빈 JSON 구조는 None 반환
    if not text or text in ("[]", "{}"):
        return None
    # JSON 파싱 시도, 실패 시 원본 문자열 반환
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text


def _normalize_row(
    row: dict[str, Any],
    agency_list: list[str] | None = None,
) -> dict[str, Any]:
    """평가셋 CSV 행을 EvalSample 스키마로 변환한다.

    Args:
        row: 평가셋 CSV의 한 행 (dict).
        agency_list: 전체 발주기관 목록 ("다중" 필터 변환에 사용).

    Returns:
        EvalSample.model_validate()에 전달할 정규화된 딕셔너리.
    """
    # 이미 정규화된 형식이면 그대로 반환
    if "question_id" in row and "question" in row:
        return row

    normalized: dict[str, Any] = {}
    # CSV의 "id" 컬럼 → EvalSample의 "question_id"로 매핑
    if "id" in row:
        normalized["question_id"] = str(row["id"])
    normalized["question"] = row.get("question", "")

    # ground_truth_docs: JSON 문자열 → 파일명 리스트로 변환
    docs = _coerce_json_field(row.get("ground_truth_docs"))
    if isinstance(docs, list):
        normalized["expected_doc_titles"] = [str(d) for d in docs]

    # 메타데이터 컬럼들을 파싱하여 metadata 딕셔너리로 수집
    metadata: dict[str, Any] = {}
    for col in _METADATA_PASSTHROUGH_COLUMNS:
        if col not in row:
            continue
        raw = row[col]
        if col == "metadata_filter":
            # metadata_filter: JSON 파싱 → 영문 키 정규화 → "다중" 처리
            parsed = _coerce_json_field(raw)
            normalized_filter = normalize_metadata_filter(
                parsed if isinstance(parsed, dict) else None,
                question=normalized.get("question", ""),
                agency_list=agency_list,
            )
            if normalized_filter:
                metadata["metadata_filter"] = normalized_filter
        elif col == "history":
            # history: JSON 파싱 → 비어있지 않은 리스트만 저장
            parsed = _coerce_json_field(raw)
            if isinstance(parsed, list) and parsed:
                metadata["history"] = parsed
        else:
            # 나머지 컬럼: NaN/빈 값 제외 후 그대로 저장
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


def load_eval_samples(
    path: str | Path,
    agency_list: list[str] | None = None,
) -> list[EvalSample]:
    """평가셋 파일을 로딩하여 EvalSample 리스트로 반환한다.

    Args:
        path: 평가셋 파일 경로 (CSV, JSON, JSONL, Parquet 지원).
        agency_list: 전체 발주기관 목록 ("다중" 필터 변환에 사용).

    Returns:
        EvalSample 객체 리스트.
    """
    source = Path(path)
    # 파일 형식에 따라 행(row) 리스트로 로딩
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
    # 각 행을 정규화 후 EvalSample 객체로 변환
    return [EvalSample.model_validate(_normalize_row(row, agency_list)) for row in rows]
