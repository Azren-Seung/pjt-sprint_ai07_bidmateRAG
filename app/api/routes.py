"""UI helper functions shared by Streamlit tabs."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pandas as pd

from bidmate_rag.evaluation.dataset import load_eval_samples
from bidmate_rag.evaluation.metrics import calc_hit_rate, calc_mrr, calc_ndcg
from bidmate_rag.evaluation.runner import (
    BenchmarkRunner,
    persist_benchmark_summary,
    persist_run_results,
)
from bidmate_rag.pipelines.runtime import build_runtime_pipeline


def list_provider_configs(config_dir: str | Path = "configs/providers") -> list[Path]:
    return sorted(Path(config_dir).glob("*.yaml"))


def load_benchmark_frames(benchmarks_dir: str | Path = "artifacts/logs/benchmarks") -> pd.DataFrame:
    benchmark_dir = Path(benchmarks_dir)
    files = sorted(benchmark_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    frames = []
    for file in files:
        frame = pd.read_parquet(file)
        frame["source_file"] = file.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_run_records(run_file: str | Path) -> list[dict]:
    path = Path(run_file)
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def run_live_query(
    question: str,
    provider_config_path: str | Path,
    base_config_path: str | Path = "configs/base.yaml",
    experiment_config_path: str | Path | None = None,
    top_k: int = 5,
):
    pipeline, runtime, embedder, _ = build_runtime_pipeline(
        base_config_path=base_config_path,
        provider_config_path=provider_config_path,
        experiment_config_path=experiment_config_path,
    )
    run_id = f"live-{uuid4().hex[:8]}"
    return pipeline.answer(
        question,
        top_k=top_k,
        scenario=runtime.provider.scenario or runtime.provider.provider,
        run_id=run_id,
        embedding_provider=embedder.provider_name,
        embedding_model=embedder.model_name,
    )


def run_benchmark_experiment(
    evaluation_path: str | Path,
    provider_config_path: str | Path,
    base_config_path: str | Path = "configs/base.yaml",
    experiment_config_path: str | Path | None = None,
    runs_dir: str | Path = "artifacts/logs/runs",
    benchmarks_dir: str | Path = "artifacts/logs/benchmarks",
) -> dict:
    pipeline, runtime, embedder, _ = build_runtime_pipeline(
        base_config_path=base_config_path,
        provider_config_path=provider_config_path,
        experiment_config_path=experiment_config_path,
    )
    samples = load_eval_samples(evaluation_path)
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

    retrieval_metrics = {
        "hit_rate@5": 0.0,
        "mrr": 0.0,
        "ndcg@5": 0.0,
    }
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
        retrieval_metrics = {
            key: round(value / scored, 4) for key, value in retrieval_metrics.items()
        }
    benchmark.metrics.update(retrieval_metrics)

    run_path = persist_run_results(benchmark.results, runs_dir=runs_dir, run_id=run_id)
    summary_path = persist_benchmark_summary(
        [benchmark.to_summary_record()],
        benchmarks_dir=benchmarks_dir,
        experiment_name=runtime.experiment.name,
    )
    return {"run_path": run_path, "summary_path": summary_path, "benchmark": benchmark}
