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


def load_metadata_options(parquet_path: str | Path = "data/processed/cleaned_documents.parquet") -> dict:
    """사이드바 필터용 메타데이터 옵션을 로딩한다."""
    path = Path(parquet_path)
    if not path.exists():
        return {"agencies": [], "domains": [], "agency_types": []}
    df = pd.read_parquet(path)
    return {
        "agencies": sorted(df["발주 기관"].dropna().unique().tolist()),
        "domains": sorted(
            df.apply(
                lambda r: _classify_domain_simple(r.get("사업명", "")), axis=1
            ).unique().tolist()
        ) if "사업명" in df.columns else [],
        "agency_types": sorted(
            df.apply(
                lambda r: _classify_agency_simple(r.get("발주 기관", "")), axis=1
            ).unique().tolist()
        ) if "발주 기관" in df.columns else [],
    }


def _classify_agency_simple(name: str) -> str:
    name = str(name)
    if any(k in name for k in ["대학", "학교"]):
        return "대학교"
    if any(k in name for k in ["공사", "공단", "진흥원", "진흥회", "평가원", "정보원"]):
        return "공기업/준정부기관"
    if any(k in name for k in ["시", "도", "군", "구", "광역"]):
        return "지방자치단체"
    if any(k in name for k in ["연구원", "연구소", "과학"]):
        return "연구기관"
    if any(k in name for k in ["부 ", "처 ", "청 ", "위원회"]):
        return "중앙행정기관"
    return "기타"


def _classify_domain_simple(name: str) -> str:
    name = str(name)
    if any(k in name for k in ["교육", "이러닝", "학습", "학사"]):
        return "교육/학습"
    if any(k in name for k in ["안전", "재난", "관제", "선량"]):
        return "안전/재난"
    if any(k in name for k in ["홈페이지", "포털", "웹"]):
        return "웹/포털"
    if any(k in name for k in ["ERP", "그룹웨어", "경영"]):
        return "경영/행정"
    if any(k in name for k in ["GIS", "지도", "수문"]):
        return "공간정보/GIS"
    if any(k in name for k in ["의료", "바이오", "병원"]):
        return "의료/바이오"
    return "기타 정보시스템"


def run_live_query(
    question: str,
    provider_config_path: str | Path,
    base_config_path: str | Path = "configs/base.yaml",
    experiment_config_path: str | Path | None = None,
    top_k: int = 5,
    manual_filters: dict | None = None,
):
    pipeline, runtime, embedder, _ = build_runtime_pipeline(
        base_config_path=base_config_path,
        provider_config_path=provider_config_path,
        experiment_config_path=experiment_config_path,
    )
    run_id = f"live-{uuid4().hex[:8]}"

    # 수동 필터가 있으면 retriever에 전달하기 위해 generation_config에 포함
    gen_config = {}
    if manual_filters:
        gen_config["manual_filters"] = manual_filters

    return pipeline.answer(
        question,
        top_k=top_k,
        scenario=runtime.provider.scenario or runtime.provider.provider,
        run_id=run_id,
        embedding_provider=embedder.provider_name,
        embedding_model=embedder.model_name,
        generation_config=gen_config,
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
