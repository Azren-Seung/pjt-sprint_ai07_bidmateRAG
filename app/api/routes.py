"""UI helper functions shared by Streamlit tabs."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pandas as pd

from bidmate_rag.evaluation.dataset import (
    find_latest_metadata_path,
    load_eval_samples,
)
from bidmate_rag.evaluation.pipeline import EvaluationArtifacts, execute_evaluation
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


def load_metadata_options(parquet_path: str | Path | None = None) -> dict:
    """사이드바 필터용 메타데이터 옵션을 로딩한다.

    ``parquet_path``가 None이면 ``find_latest_metadata_path()``로 가장 최근
    실험별 cleaned_documents.parquet를 사용 (없으면 top-level fallback).
    """
    if parquet_path is None:
        parquet_path = find_latest_metadata_path()
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
    system_prompt: str | None = None,
    max_context_chars: int = 8000,
):
    pipeline, runtime, embedder, _ = build_runtime_pipeline(
        base_config_path=base_config_path,
        provider_config_path=provider_config_path,
        experiment_config_path=experiment_config_path,
    )
    run_id = f"live-{uuid4().hex[:8]}"

    # 시스템 프롬프트 오버라이드
    if system_prompt:
        pipeline.system_prompt = system_prompt

    # max_context_chars는 LLM generation_config로, manual_filters는 retriever
    # explicit filter로 별도 전달 (이전에는 둘 다 generation_config에 넣었지만
    # retriever가 generation_config를 보지 않아 manual_filters가 무시되던 버그)
    gen_config = {"max_context_chars": max_context_chars}
    metadata_filter: dict | None = None
    if manual_filters:
        # "필터 없음" 모드에서는 자동 추출도 비활성화하기 위해 빈 dict 전달
        if manual_filters.get("_no_filter"):
            metadata_filter = {}
        else:
            metadata_filter = dict(manual_filters)

    return pipeline.answer(
        question,
        top_k=top_k,
        metadata_filter=metadata_filter,
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
    *,
    run_id: str | None = None,
    skip_judge: bool = False,
    judge_model: str = "gpt-4o-mini",
    progress_callback=None,
) -> EvaluationArtifacts:
    """Build the runtime pipeline and run the full evaluation.

    Thin wrapper around ``execute_evaluation`` so the Streamlit UI can stay
    free of business logic. CLI (``cli/eval.py``) and this function call the
    same underlying pipeline.
    """
    pipeline, runtime, embedder, _ = build_runtime_pipeline(
        base_config_path=base_config_path,
        provider_config_path=provider_config_path,
        experiment_config_path=experiment_config_path,
    )
    samples = load_eval_samples(evaluation_path)

    return execute_evaluation(
        samples,
        pipeline=pipeline,
        runtime=runtime,
        embedder=embedder,
        eval_path=str(evaluation_path),
        config_paths={
            "base": str(base_config_path),
            "provider": str(provider_config_path),
            "experiment": str(experiment_config_path) if experiment_config_path else None,
        },
        runs_dir=runs_dir,
        benchmarks_dir=benchmarks_dir,
        run_id=run_id,
        skip_judge=skip_judge,
        judge_model=judge_model,
        progress_callback=progress_callback,
    )
