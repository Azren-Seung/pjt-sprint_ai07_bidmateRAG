"""CLI entrypoint for benchmark execution.

This module is intentionally thin: it parses CLI args, builds the runtime
pipeline, then delegates to ``bidmate_rag.evaluation.pipeline.execute_evaluation``.
All business logic (metrics, judge, persistence, meta.json) lives there so
the Streamlit UI can call the exact same code path.

Usage::

    uv run bidmate-eval \\
        --evaluation-path data/eval/eval_batch_01.csv \\
        --provider-config configs/providers/openai_gpt5mini.yaml \\
        --limit 5 --filter-type A,B
"""

from __future__ import annotations

import argparse

import pandas as pd
from dotenv import load_dotenv

from bidmate_rag.evaluation.dataset import load_eval_samples
from bidmate_rag.evaluation.pipeline import EvaluationArtifacts, execute_evaluation
from bidmate_rag.pipelines.runtime import build_runtime_pipeline
from bidmate_rag.schema import EvalSample, GenerationResult


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


def _print_artifacts(artifacts: EvaluationArtifacts) -> None:
    """Print the file paths and judge totals at the end of a run."""
    print()
    print(f"run_id:    {artifacts.run_id}")
    print(f"run jsonl: {artifacts.run_path}")
    print(f"summary:   {artifacts.summary_path}")
    print(f"meta:      {artifacts.meta_path}")
    if not artifacts.judge_skipped:
        print(
            f"judge:     ${artifacts.judge_total_cost_usd:.4f} "
            f"({artifacts.judge_total_tokens} tokens)"
        )


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

    artifacts = execute_evaluation(
        samples,
        pipeline=pipeline,
        runtime=runtime,
        embedder=embedder,
        eval_path=args.evaluation_path,
        config_paths={
            "base": args.base_config,
            "provider": args.provider_config,
            "experiment": args.experiment_config,
        },
        runs_dir=args.runs_dir,
        benchmarks_dir=args.benchmarks_dir,
        run_id=args.run_id,
        skip_judge=args.skip_judge,
        judge_model=args.judge_model,
    )

    _print_summary(samples, artifacts.benchmark.results, artifacts.metrics)
    _print_artifacts(artifacts)


if __name__ == "__main__":
    main()
