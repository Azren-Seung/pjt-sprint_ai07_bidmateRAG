"""Markdown experiment report generator.

Reads ``artifacts/logs/runs/{run_id}.jsonl``,
``artifacts/logs/benchmarks/{exp_name}.parquet``,
``artifacts/logs/runs/{run_id}.meta.json``, and
``artifacts/logs/embeddings/{collection_name}.json`` and produces a single
markdown file at ``artifacts/reports/{exp_name}_{run_id}.md`` that the team
can copy-paste into Notion.
"""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from bidmate_rag.tracking.pricing import is_model_priced, load_pricing
from bidmate_rag.tracking.templates import REPORT_TEMPLATE

logger = logging.getLogger(__name__)


def _fmt_num(value: Any, digits: int = 4, default: str = "N/A") -> str:
    if value is None:
        return default
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return default


def _fmt_int(value: Any, default: str = "N/A") -> str:
    if value is None:
        return default
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return default


@dataclass
class ReportData:
    run_id: str
    experiment_name: str
    meta: dict[str, Any]
    summary_row: dict[str, Any]
    results: list[dict[str, Any]]
    embedding_meta: dict[str, Any] | None
    pricing: dict[str, Any]
    runs_dir: Path
    benchmarks_dir: Path
    embeddings_dir: Path
    extras: dict[str, Any] = field(default_factory=dict)


def load_report_data(
    run_id: str,
    runs_dir: str | Path = "artifacts/logs/runs",
    benchmarks_dir: str | Path = "artifacts/logs/benchmarks",
    embeddings_dir: str | Path = "artifacts/logs/embeddings",
    experiment_name: str | None = None,
) -> ReportData:
    runs_path = Path(runs_dir)
    benchmarks_path = Path(benchmarks_dir)
    embeddings_path = Path(embeddings_dir)

    # 1. meta.json (있으면)
    meta_file = runs_path / f"{run_id}.meta.json"
    meta: dict[str, Any] = {}
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))

    # 2. experiment_name 결정 (인자 우선, 그다음 meta, 그다음 parquet 스캔)
    exp_name = experiment_name or meta.get("experiment_name")
    if not exp_name:
        exp_name = _scan_benchmarks_for_run(benchmarks_path, run_id)
    if not exp_name:
        raise FileNotFoundError(
            f"Could not determine experiment_name for run_id={run_id}. "
            "Pass --experiment-name or ensure meta.json exists."
        )

    # 3. results jsonl
    jsonl_file = runs_path / f"{run_id}.jsonl"
    if not jsonl_file.exists():
        raise FileNotFoundError(f"Run results not found: {jsonl_file}")
    results = [
        json.loads(line)
        for line in jsonl_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # 4. parquet → 해당 run_id 행
    parquet_file = benchmarks_path / f"{exp_name}.parquet"
    summary_row: dict[str, Any] = {}
    if parquet_file.exists():
        frame = pd.read_parquet(parquet_file)
        matching = frame[frame["run_id"] == run_id]
        if not matching.empty:
            summary_row = matching.iloc[-1].to_dict()
    else:
        logger.warning("Benchmark parquet not found: %s", parquet_file)

    # 5. embedding meta (collection_name으로 매칭)
    collection_name = meta.get("collection_name")
    embedding_meta: dict[str, Any] | None = None
    if collection_name:
        emb_file = embeddings_path / f"{collection_name}.json"
        if emb_file.exists():
            embedding_meta = json.loads(emb_file.read_text(encoding="utf-8"))

    return ReportData(
        run_id=run_id,
        experiment_name=exp_name,
        meta=meta,
        summary_row=summary_row,
        results=results,
        embedding_meta=embedding_meta,
        pricing=load_pricing(),
        runs_dir=runs_path,
        benchmarks_dir=benchmarks_path,
        embeddings_dir=embeddings_path,
    )


def _scan_benchmarks_for_run(benchmarks_path: Path, run_id: str) -> str | None:
    if not benchmarks_path.exists():
        return None
    for parquet_file in benchmarks_path.glob("*.parquet"):
        try:
            frame = pd.read_parquet(parquet_file)
        except Exception:  # noqa: BLE001
            continue
        if "run_id" in frame.columns and (frame["run_id"] == run_id).any():
            return parquet_file.stem
    return None


def render_markdown(data: ReportData) -> str:
    ctx = _build_context(data)
    return REPORT_TEMPLATE.format(**ctx)


def write_report(
    data: ReportData,
    output_dir: str | Path = "artifacts/reports",
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{data.experiment_name}_{data.run_id}.md"
    out_path.write_text(render_markdown(data), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# context builders
# ---------------------------------------------------------------------------

def _build_context(data: ReportData) -> dict[str, Any]:
    summary = data.summary_row
    meta = data.meta
    runtime_cfg = meta.get("config_snapshot", {}) or {}
    provider_cfg = runtime_cfg.get("provider", {}) or {}
    experiment_cfg = runtime_cfg.get("experiment", {}) or {}
    project_cfg = runtime_cfg.get("project", {}) or {}
    git = meta.get("git", {}) or {}

    # Costs
    generation_cost = sum(
        float(r.get("cost_usd") or 0.0) for r in data.results
    )
    embedding_cost = float((data.embedding_meta or {}).get("total_cost_usd", 0.0) or 0.0)
    judge_cost = float(meta.get("judge_total_cost_usd", 0.0) or 0.0)
    grand_total = generation_cost + embedding_cost + judge_cost

    # Tokens
    prompt_tokens_sum = sum(
        int((r.get("token_usage") or {}).get("prompt", 0) or 0) for r in data.results
    )
    completion_tokens_sum = sum(
        int((r.get("token_usage") or {}).get("completion", 0) or 0) for r in data.results
    )
    total_tokens = prompt_tokens_sum + completion_tokens_sum

    # Latency
    latencies_ms = [
        float(r.get("latency_ms") or 0.0) for r in data.results if r.get("latency_ms")
    ]
    latency_avg_s = (sum(latencies_ms) / len(latencies_ms) / 1000) if latencies_ms else None
    latency_p95_s = _percentile(latencies_ms, 95) / 1000 if latencies_ms else None

    # Models / config
    embedding_model = (
        provider_cfg.get("embedding_model")
        or (data.results[0].get("embedding_model") if data.results else None)
        or "unknown"
    )
    llm_model = (
        provider_cfg.get("model")
        or (data.results[0].get("llm_model") if data.results else None)
        or "unknown"
    )
    chunk_size = experiment_cfg.get("chunk_size") or project_cfg.get("default_chunk_size", "?")
    chunk_overlap = experiment_cfg.get("chunk_overlap") or project_cfg.get(
        "default_chunk_overlap", "?"
    )
    top_k = experiment_cfg.get("retrieval_top_k") or project_cfg.get(
        "default_retrieval_top_k", "?"
    )
    collection_name = meta.get("collection_name") or provider_cfg.get("collection_name") or "?"
    scenario = summary.get("scenario") or provider_cfg.get("scenario") or "?"
    provider_label = summary.get("provider_label", "?")

    # Eval path
    eval_path = meta.get("eval_path", "?")
    eval_basename = Path(str(eval_path)).name if eval_path != "?" else "?"
    num_samples = int(summary.get("num_samples") or len(data.results) or 0)

    # Cost warnings (priced?)
    warnings = []
    if not is_model_priced("llm", llm_model, data.pricing):
        warnings.append(f"⚠️ 생성 모델 `{llm_model}` 단가 미등록 — `configs/pricing.yaml` 갱신 필요")
    if data.embedding_meta and not is_model_priced(
        "embedding", embedding_model, data.pricing
    ):
        warnings.append(
            f"⚠️ 임베딩 모델 `{embedding_model}` 단가 미등록 — `configs/pricing.yaml` 갱신 필요"
        )
    if not data.embedding_meta:
        warnings.append("⚠️ 임베딩 비용 미수집 (build_index를 새 트래킹 코드로 다시 실행 필요)")
    cost_warning = "\n".join(warnings) if warnings else ""

    # Config links
    configs = meta.get("configs", {}) or {}
    config_links = "\n".join(
        f"  - `{configs[k]}`" for k in ("base", "provider", "experiment") if configs.get(k)
    ) or "  - (미기록)"

    # Judge metrics from summary first, fall back to results aggregate
    def _get_metric(key: str) -> Any:
        return summary.get(key) if summary else None

    return {
        "experiment_name": data.experiment_name,
        "run_id": data.run_id,
        "timestamp_kst": meta.get("timestamp_kst", "N/A"),
        "scenario": scenario,
        "eval_basename": eval_basename,
        "eval_path": eval_path,
        "num_samples": num_samples,
        "embedding_model": embedding_model,
        "llm_model": llm_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k": top_k,
        "collection_name": collection_name,
        "provider_label": provider_label,
        "git_commit": git.get("commit", "unknown"),
        "git_commit_short": git.get("commit_short", "unknown"),
        "git_branch": git.get("branch", "unknown"),
        "dirty_marker": "(dirty)" if git.get("dirty") else "",
        # metrics
        "hit_rate": _fmt_num(_get_metric("hit_rate@5")),
        "mrr": _fmt_num(_get_metric("mrr")),
        "ndcg": _fmt_num(_get_metric("ndcg@5")),
        "faithfulness": _fmt_num(_get_metric("faithfulness")),
        "answer_relevance": _fmt_num(_get_metric("answer_relevance")),
        "context_precision": _fmt_num(_get_metric("context_precision")),
        "context_recall": _fmt_num(_get_metric("context_recall")),
        # latency
        "latency_avg_s": _fmt_num(latency_avg_s, digits=3),
        "latency_p95_s": _fmt_num(latency_p95_s, digits=3),
        # tokens
        "prompt_tokens_sum": _fmt_int(prompt_tokens_sum),
        "completion_tokens_sum": _fmt_int(completion_tokens_sum),
        "total_tokens": _fmt_int(total_tokens),
        # costs
        "generation_cost": _fmt_num(generation_cost, digits=4),
        "embedding_cost": (
            _fmt_num(embedding_cost, digits=4) if data.embedding_meta else "N/A (미수집)"
        ),
        "judge_cost": _fmt_num(judge_cost, digits=4),
        "grand_total_cost": _fmt_num(grand_total, digits=4),
        "cost_warning": cost_warning,
        # paths
        "run_jsonl_path": str(data.runs_dir / f"{data.run_id}.jsonl"),
        "benchmark_parquet_path": str(
            data.benchmarks_dir / f"{data.experiment_name}.parquet"
        ),
        "meta_json_path": str(data.runs_dir / f"{data.run_id}.meta.json"),
        "config_links": config_links,
    }


def _percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    # statistics.quantiles(n=100) → 99 cut points; index pct-1
    try:
        quantiles = statistics.quantiles(sorted_vals, n=100)
        return quantiles[pct - 1]
    except statistics.StatisticsError:
        return sorted_vals[-1]
