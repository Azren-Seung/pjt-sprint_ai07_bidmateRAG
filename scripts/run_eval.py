"""CLI entrypoint for benchmark execution."""

from __future__ import annotations

import argparse

from dotenv import load_dotenv
load_dotenv()
from uuid import uuid4

from bidmate_rag.evaluation.benchmark import (
    BenchmarkRunner,
    persist_benchmark_summary,
    persist_run_results,
)
from bidmate_rag.evaluation.dataset import load_eval_samples
from bidmate_rag.evaluation.metrics import calc_hit_rate, calc_mrr, calc_ndcg
from bidmate_rag.pipelines.runtime import build_runtime_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the shared benchmark runner.")
    parser.add_argument("--evaluation-path", required=True)
    parser.add_argument("--provider-config", required=True)
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--experiment-config", default=None)
    parser.add_argument("--runs-dir", default="artifacts/logs/runs")
    parser.add_argument("--benchmarks-dir", default="artifacts/logs/benchmarks")
    args = parser.parse_args()

    pipeline, runtime, embedder, _ = build_runtime_pipeline(
        base_config_path=args.base_config,
        provider_config_path=args.provider_config,
        experiment_config_path=args.experiment_config,
    )
    samples = load_eval_samples(args.evaluation_path)
    run_id = f"bench-{uuid4().hex[:8]}"

    def answer_fn(sample):
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

    retrieval_metrics = {"hit_rate@5": 0.0, "mrr": 0.0, "ndcg@5": 0.0}
    scored = 0
    for sample, result in zip(samples, benchmark.results, strict=False):
        if not sample.expected_doc_ids:
            continue
        hit = calc_hit_rate(result.retrieved_chunks, sample.expected_doc_ids, k=5)
        mrr = calc_mrr(result.retrieved_chunks, sample.expected_doc_ids)
        ndcg = calc_ndcg(result.retrieved_chunks, sample.expected_doc_ids, k=5)
        if hit is not None:
            retrieval_metrics["hit_rate@5"] += hit
            retrieval_metrics["mrr"] += mrr or 0.0
            retrieval_metrics["ndcg@5"] += ndcg or 0.0
            scored += 1
    if scored:
        benchmark.metrics.update(
            {key: round(value / scored, 4) for key, value in retrieval_metrics.items()}
        )

    run_path = persist_run_results(benchmark.results, runs_dir=args.runs_dir, run_id=run_id)
    summary_path = persist_benchmark_summary(
        [benchmark.to_summary_record()],
        benchmarks_dir=args.benchmarks_dir,
        experiment_name=runtime.experiment.name,
    )
    print(run_path)
    print(summary_path)


if __name__ == "__main__":
    main()
