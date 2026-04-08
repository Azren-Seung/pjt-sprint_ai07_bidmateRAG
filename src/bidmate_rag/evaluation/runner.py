"""Benchmark runner and result persistence helpers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from bidmate_rag.schema import BenchmarkRunResult, EvalSample, GenerationResult


def persist_run_results(
    results: list[GenerationResult], runs_dir: str | Path, run_id: str | None = None
) -> Path:
    runs_path = Path(runs_dir)
    runs_path.mkdir(parents=True, exist_ok=True)
    resolved_run_id = run_id or (
        results[0].run_id if results else datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    )
    output_path = runs_path / f"{resolved_run_id}.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(
                json.dumps(result.to_record(), ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    return output_path


def persist_benchmark_summary(
    records: list[dict], benchmarks_dir: str | Path, experiment_name: str
) -> Path:
    benchmark_path = Path(benchmarks_dir)
    benchmark_path.mkdir(parents=True, exist_ok=True)
    output_path = benchmark_path / f"{experiment_name}.parquet"
    pd.DataFrame(records).to_parquet(output_path, index=False)
    return output_path


class BenchmarkRunner:
    """Run an evaluation dataset through an answer function and persist results."""

    def __init__(self, answer_fn: Callable[[EvalSample], GenerationResult]):
        self.answer_fn = answer_fn

    def run(
        self,
        experiment_name: str,
        scenario: str,
        provider_label: str,
        samples: list[EvalSample],
    ) -> BenchmarkRunResult:
        results = [self.answer_fn(sample) for sample in samples]
        return BenchmarkRunResult(
            experiment_name=experiment_name,
            run_id=results[0].run_id if results else datetime.now(UTC).strftime("%Y%m%d%H%M%S"),
            scenario=scenario,
            provider_label=provider_label,
            samples=samples,
            results=results,
            metrics={},
        )
