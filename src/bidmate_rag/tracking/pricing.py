"""Token pricing helpers — model price lookup and cost calculation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def _find_project_root() -> Path:
    """Walk up from this file looking for ``pyproject.toml``.

    Allows ``load_pricing()`` to work regardless of the caller's current
    working directory. Falls back to ``Path.cwd()`` if no marker is found
    (e.g. when the package is installed into site-packages without a checkout).
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


DEFAULT_PRICING_PATH = _find_project_root() / "configs" / "pricing.yaml"


def load_pricing(path: str | Path = DEFAULT_PRICING_PATH) -> dict[str, Any]:
    """Load model pricing table from YAML.

    Returns an empty pricing dict if the file does not exist (so callers can
    still operate; cost will simply be 0 with a warning per unknown model).
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
    """Compute USD cost for a single LLM call given token counts.

    If ``cached_tokens > 0``, those tokens are billed at ``cached_input_per_1m``
    (typically 10–50% of input rate). Models without ``cached_input_per_1m``
    fall back to the regular ``input_per_1m`` rate so existing behavior is
    preserved.

    Returns 0.0 (and logs a warning once per unknown model) if the model has no
    entry in the pricing table.
    """
    table = (pricing or {}).get("llm", {})
    entry = table.get(model)
    if not entry:
        _warn_unknown_model("llm", model)
        return 0.0
    input_rate = float(entry.get("input_per_1m", 0.0))
    output_rate = float(entry.get("output_per_1m", 0.0))
    cached_rate = float(entry.get("cached_input_per_1m", input_rate))
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
    """Compute USD cost for embedding `total_tokens` tokens with `model`."""
    table = (pricing or {}).get("embedding", {})
    entry = table.get(model)
    if not entry:
        _warn_unknown_model("embedding", model)
        return 0.0
    rate = float(entry.get("per_1m", 0.0))
    cost = total_tokens * rate / 1_000_000
    return round(cost, 6)


def is_model_priced(kind: str, model: str, pricing: dict[str, Any]) -> bool:
    """Check whether a model has a pricing entry. Used for report warnings."""
    return bool((pricing or {}).get(kind, {}).get(model))


_warned_models: set[tuple[str, str]] = set()


def _warn_unknown_model(kind: str, model: str) -> None:
    key = (kind, model)
    if key in _warned_models:
        return
    _warned_models.add(key)
    logger.warning(
        "No pricing entry for %s model %r — cost will be 0.0. Update configs/pricing.yaml.",
        kind,
        model,
    )
