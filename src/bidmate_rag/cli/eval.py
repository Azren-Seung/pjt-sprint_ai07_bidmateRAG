"""CLI entrypoint for benchmark execution.

Usage::

    uv run bidmate-eval \\
        --evaluation-path data/eval/eval_batch_01.csv \\
        --provider-config configs/providers/openai_gpt5mini.yaml \\
        --limit 5 --filter-type A,B
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

from bidmate_rag.evaluation.benchmark import (
    BenchmarkRunner,
    persist_benchmark_summary,
    persist_run_results,
)
from bidmate_rag.evaluation.dataset import load_eval_samples
from bidmate_rag.evaluation.judge import LLMJudge
from bidmate_rag.evaluation.metrics import calc_hit_rate, calc_mrr, calc_ndcg
from bidmate_rag.pipelines.runtime import build_runtime_pipeline, collection_name_for_config
from bidmate_rag.schema import EvalSample, GenerationResult
from bidmate_rag.tracking.git_info import capture_git_info


def _split_csv(value: str | None) -> list[str] | None:
    """Parse a comma-separated CLI flag like 'A,B' into ['A', 'B']."""
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _apply_filters(
    samples: list[EvalSample],
    *,
    types: list[str] | None,
    difficulties: list[str] | None,
    limit: int | None,
) -> list[EvalSample]:
    """Filter samples by metadata.type / metadata.difficulty, then truncate."""
    filtered = samples
    if types:
        type_set = {t for t in types}
        filtered = [s for s in filtered if str(s.metadata.get("type", "")) in type_set]
    if difficulties:
        diff_set = {d for d in difficulties}
        filtered = [
            s for s in filtered if str(s.metadata.get("difficulty", "")) in diff_set
        ]
    if limit is not None and limit >= 0:
        filtered = filtered[:limit]
    return filtered


def _write_run_meta(
    runs_dir: Path,
    run_id: str,
    experiment_name: str,
    runtime,
    collection_name: str,
    eval_path: str,
    config_paths: dict[str, str | None],
) -> Path:
    """Persist a sidecar meta.json next to the run jsonl."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    now_utc = datetime.now(UTC)
    meta = {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "timestamp_utc": now_utc.isoformat(),
        "timestamp_kst": now_utc.astimezone(ZoneInfo("Asia/Seoul")).strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "git": capture_git_info(),
        "configs": {k: v for k, v in config_paths.items() if v},
        "config_snapshot": runtime.model_dump(),
        "eval_path": eval_path,
        "collection_name": collection_name,
        "judge_total_cost_usd": 0.0,
        "judge_total_tokens": 0,
    }
    out_path = runs_dir / f"{run_id}.meta.json"
    out_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def _update_run_meta(meta_path: Path, **fields) -> None:
    if not meta_path.exists():
        return
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    data.update(fields)
    meta_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _print_summary(
    samples: list[EvalSample],
    results: list[GenerationResult],
    overall_metrics: dict[str, float],
) -> None:
    """Render a console-friendly summary of a benchmark run."""
    if not results:
        print("(no results)")
        return

    rows = []
    for sample, result in zip(samples, results, strict=False):
        rows.append(
            {
                "id": result.question_id,
                "type": sample.metadata.get("type", ""),
                "difficulty": sample.metadata.get("difficulty", ""),
                "tokens": int(result.token_usage.get("total", 0) or 0),
                "latency_ms": round(result.latency_ms),
                "error": bool(result.error),
            }
        )
    df = pd.DataFrame(rows)

    print()
    print(f"=== Run summary ({len(df)} questions) ===")
    print(
        f"errors={int(df['error'].sum())}  "
        f"avg_tokens={df['tokens'].mean():.0f}  "
        f"avg_latency_ms={df['latency_ms'].mean():.0f}"
    )
    if overall_metrics:
        metric_str = "  ".join(f"{k}={v}" for k, v in overall_metrics.items())
        print(f"retrieval: {metric_str}")

    if df["type"].astype(bool).any():
        print("\n-- by type --")
        by_type = (
            df.groupby("type")
            .agg(
                n=("id", "count"),
                avg_tokens=("tokens", "mean"),
                avg_latency_ms=("latency_ms", "mean"),
                errors=("error", "sum"),
            )
            .round(0)
        )
        print(by_type.to_string())

    if df["difficulty"].astype(bool).any():
        print("\n-- by difficulty --")
        by_diff = (
            df.groupby("difficulty")
            .agg(
                n=("id", "count"),
                avg_tokens=("tokens", "mean"),
                avg_latency_ms=("latency_ms", "mean"),
                errors=("error", "sum"),
            )
            .round(0)
        )
        print(by_diff.to_string())


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="bidmate-eval",
        description="Run the BidMate RAG benchmark from the command line.",
    )
    parser.add_argument("--evaluation-path", required=True)
    parser.add_argument("--provider-config", required=True)
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--experiment-config", default=None)
    parser.add_argument("--runs-dir", default="artifacts/logs/runs")
    parser.add_argument("--benchmarks-dir", default="artifacts/logs/benchmarks")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N questions after other filters are applied.",
    )
    parser.add_argument(
        "--filter-type",
        default=None,
        help="Comma-separated question types to keep, e.g. 'A,B'.",
    )
    parser.add_argument(
        "--filter-difficulty",
        default=None,
        help="Comma-separated difficulties to keep, e.g. '하,중'.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Override run_id (otherwise auto-generated as bench-XXXXXXXX).",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip LLM-judge evaluation (faithfulness etc.).",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4o-mini",
        help="LLM model used by the judge (default: gpt-4o-mini).",
    )
    args = parser.parse_args()

    pipeline, runtime, embedder, _ = build_runtime_pipeline(
        base_config_path=args.base_config,
        provider_config_path=args.provider_config,
        experiment_config_path=args.experiment_config,
    )

    all_samples = load_eval_samples(args.evaluation_path)
    samples = _apply_filters(
        all_samples,
        types=_split_csv(args.filter_type),
        difficulties=_split_csv(args.filter_difficulty),
        limit=args.limit,
    )
    print(
        f"Loaded {len(all_samples)} samples from {args.evaluation_path}; "
        f"running {len(samples)} after filters."
    )
    if not samples:
        raise SystemExit("No samples remain after filtering.")

    run_id = args.run_id or f"bench-{uuid4().hex[:8]}"

    runs_dir = Path(args.runs_dir)
    meta_path = _write_run_meta(
        runs_dir=runs_dir,
        run_id=run_id,
        experiment_name=runtime.experiment.name,
        runtime=runtime,
        collection_name=collection_name_for_config(runtime),
        eval_path=args.evaluation_path,
        config_paths={
            "base": args.base_config,
            "provider": args.provider_config,
            "experiment": args.experiment_config,
        },
    )

    def answer_fn(sample: EvalSample) -> GenerationResult:
        return pipeline.answer(
            sample.question,
            question_id=sample.question_id,
            scenario=runtime.provider.scenario or runtime.provider.provider,
            run_id=run_id,
            embedding_provider=embedder.provider_name,
            embedding_model=embedder.model_name,
        )

    benchmark = BenchmarkRunner(answer_fn).run(
        experiment_name=runtime.experiment.name,
        scenario=runtime.provider.scenario or runtime.provider.provider,
        provider_label=f"{runtime.provider.provider}:{runtime.provider.model}",
        samples=samples,
    )

    retrieval_totals = {"hit_rate@5": 0.0, "mrr": 0.0, "ndcg@5": 0.0}
    scored = 0
    for sample, result in zip(samples, benchmark.results, strict=False):
        if not sample.expected_doc_ids:
            continue
        hit = calc_hit_rate(result.retrieved_chunks, sample.expected_doc_ids, k=5)
        mrr = calc_mrr(result.retrieved_chunks, sample.expected_doc_ids)
        ndcg = calc_ndcg(result.retrieved_chunks, sample.expected_doc_ids, k=5)
        if hit is not None:
            retrieval_totals["hit_rate@5"] += hit
            retrieval_totals["mrr"] += mrr or 0.0
            retrieval_totals["ndcg@5"] += ndcg or 0.0
            scored += 1
    if scored:
        averaged = {key: round(value / scored, 4) for key, value in retrieval_totals.items()}
        benchmark.metrics.update(averaged)

    judge_total_cost_usd = 0.0
    judge_total_tokens = 0
    if not args.skip_judge:
        judge = LLMJudge(model=args.judge_model)
        judge_totals = {key: 0.0 for key in LLMJudge.METRIC_KEYS}
        judged = 0
        for sample, result in zip(samples, benchmark.results, strict=False):
            contexts = [chunk.chunk.text for chunk in result.retrieved_chunks]
            scores = judge.evaluate(
                question=sample.question,
                answer=result.answer,
                contexts=contexts,
                expected_answer=sample.metadata.get("ground_truth_answer"),
            )
            result.judge_scores = scores.to_dict()
            if scores.error:
                continue
            for key in LLMJudge.METRIC_KEYS:
                judge_totals[key] += getattr(scores, key)
            judged += 1
        if judged:
            benchmark.metrics.update(
                {key: round(value / judged, 4) for key, value in judge_totals.items()}
            )
        judge_total_cost_usd = round(judge.cumulative_cost_usd, 6)
        judge_total_tokens = judge.cumulative_tokens
        print(
            f"judge: scored {judged}/{len(samples)} samples "
            f"(cost ${judge_total_cost_usd:.4f}, {judge_total_tokens} tokens)"
        )

    _update_run_meta(
        meta_path,
        judge_total_cost_usd=judge_total_cost_usd,
        judge_total_tokens=judge_total_tokens,
    )

    run_path = persist_run_results(benchmark.results, runs_dir=args.runs_dir, run_id=run_id)
    summary_path = persist_benchmark_summary(
        [benchmark.to_summary_record()],
        benchmarks_dir=args.benchmarks_dir,
        experiment_name=runtime.experiment.name,
    )

    _print_summary(samples, benchmark.results, benchmark.metrics)
    print()
    print(f"run_id:    {run_id}")
    print(f"run jsonl: {run_path}")
    print(f"summary:   {summary_path}")
    print(f"meta:      {meta_path}")


if __name__ == "__main__":
    main()
