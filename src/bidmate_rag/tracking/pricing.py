"""Token pricing helpers — model price lookup and cost calculation.

모델별 토큰 단가를 pricing.yaml에서 읽어 LLM/임베딩 비용을 계산한다.
실험 리포트에 비용 정보를 표시할 때 사용.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def _find_project_root() -> Path:
    """pyproject.toml을 기준으로 프로젝트 루트 경로를 찾는다.

    Returns:
        프로젝트 루트 Path. 못 찾으면 현재 작업 디렉터리.
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


DEFAULT_PRICING_PATH = _find_project_root() / "configs" / "pricing.yaml"


def load_pricing(path: str | Path = DEFAULT_PRICING_PATH) -> dict[str, Any]:
    """YAML에서 모델 단가 테이블을 로딩한다.

    Args:
        path: pricing.yaml 파일 경로.

    Returns:
        llm과 embedding 키를 가진 단가 딕셔너리.
    """
    pricing_path = Path(path)
    if not pricing_path.exists():
        logger.warning("Pricing file not found at %s — all costs will be 0.0", pricing_path)
        return {"llm": {}, "embedding": {}}
    data = yaml.safe_load(pricing_path.read_text(encoding="utf-8")) or {}
    data.setdefault("llm", {})
    data.setdefault("embedding", {})
    return data


def calc_llm_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    pricing: dict[str, Any],
    *,
    cached_tokens: int = 0,
) -> float:
    """LLM 호출 1회의 USD 비용을 계산한다.

    Args:
        model: 모델 이름 (예: "gpt-5-mini").
        prompt_tokens: 입력 토큰 수.
        completion_tokens: 출력 토큰 수.
        pricing: 단가 딕셔너리 (load_pricing 반환값).
        cached_tokens: 캐시된 입력 토큰 수 (할인 적용).

    Returns:
        USD 비용. 단가 미등록 모델은 0.0 반환.
    """
    table = (pricing or {}).get("llm", {})
    entry = table.get(model)
    if not entry:
        _warn_unknown_model("llm", model)
        return 0.0
    # 단가 로딩 (100만 토큰당 USD)
    input_rate = float(entry.get("input_per_1m", 0.0))
    output_rate = float(entry.get("output_per_1m", 0.0))
    cached_rate = float(entry.get("cached_input_per_1m", input_rate))  # 캐시 할인 단가
    # 캐시되지 않은 입력 토큰만 정가 적용
    uncached_prompt = max(prompt_tokens - cached_tokens, 0)
    cost = (
        uncached_prompt * input_rate
        + cached_tokens * cached_rate
        + completion_tokens * output_rate
    ) / 1_000_000
    return round(cost, 6)


def calc_embedding_cost(
    model: str,
    total_tokens: int,
    pricing: dict[str, Any],
) -> float:
    """임베딩 토큰의 USD 비용을 계산한다.

    Args:
        model: 임베딩 모델 이름 (예: "text-embedding-3-small").
        total_tokens: 총 토큰 수.
        pricing: 단가 딕셔너리.

    Returns:
        USD 비용.
    """
    table = (pricing or {}).get("embedding", {})
    entry = table.get(model)
    if not entry:
        _warn_unknown_model("embedding", model)
        return 0.0
    rate = float(entry.get("per_1m", 0.0))
    cost = total_tokens * rate / 1_000_000
    return round(cost, 6)


def is_model_priced(kind: str, model: str, pricing: dict[str, Any]) -> bool:
    """모델의 단가가 등록돼 있는지 확인한다.

    Args:
        kind: "llm" 또는 "embedding".
        model: 모델 이름.
        pricing: 단가 딕셔너리.

    Returns:
        단가가 있으면 True.
    """
    return bool((pricing or {}).get(kind, {}).get(model))


_warned_models: set[tuple[str, str]] = set()


def _warn_unknown_model(kind: str, model: str) -> None:
    """단가 미등록 모델에 대해 경고를 한 번만 출력한다."""
    key = (kind, model)
    if key in _warned_models:  # 같은 모델은 한 번만 경고
        return
    _warned_models.add(key)
    logger.warning(
        "No pricing entry for %s model %r — cost will be 0.0. Update configs/pricing.yaml.",
        kind,
        model,
    )
