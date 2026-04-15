"""Tests for eval correctness fixes (top_k / metadata_filter / history /
chunking isolation / parquet append).

Each test targets a single regression that the fix branch addressed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from bidmate_rag.config.settings import (
    ExperimentConfig,
    ProjectConfig,
    ProviderConfig,
    RuntimeConfig,
)
from bidmate_rag.evaluation.dataset import (
    EVAL_FILTER_KEY_MAP,
    normalize_metadata_filter,
)
from bidmate_rag.evaluation.runner import persist_benchmark_summary
from bidmate_rag.pipelines.runtime import collection_name_for_config
from bidmate_rag.retrieval.retriever import RAGRetriever

# ---------------------------------------------------------------------------
# Phase 2: normalize_metadata_filter
# ---------------------------------------------------------------------------


def test_normalize_metadata_filter_maps_english_to_korean_keys():
    out = normalize_metadata_filter({"agency": "한국가스공사", "year": "2024"})
    assert out == {"발주 기관": "한국가스공사", "공개연도": 2024}


def test_normalize_metadata_filter_institution_aliased_to_agency():
    out = normalize_metadata_filter({"institution": "한국철도공사"})
    assert out == {"발주 기관": "한국철도공사"}


def test_normalize_metadata_filter_year_string_coerced_to_int():
    out = normalize_metadata_filter({"year": "2023"})
    assert out == {"공개연도": 2023}
    assert isinstance(out["공개연도"], int)


def test_normalize_metadata_filter_none_or_empty_returns_none():
    assert normalize_metadata_filter(None) is None
    assert normalize_metadata_filter({}) is None


def test_normalize_metadata_filter_unknown_key_passes_through():
    # 매핑에 없는 키는 그대로 보존 (warning 로그만 발생)
    out = normalize_metadata_filter({"weird_key": "x"})
    assert out == {"weird_key": "x"}


def test_eval_filter_key_map_covers_known_eval_csv_keys():
    # 평가셋 580개 샘플에서 발견된 키 (agency/institution/project/domain/year)
    for key in ("agency", "institution", "project", "domain", "year"):
        assert key in EVAL_FILTER_KEY_MAP


# ---------------------------------------------------------------------------
# Phase 2: RAGRetriever.retrieve metadata_filter override semantics
# ---------------------------------------------------------------------------


def _make_retriever_with_mock_store():
    vector_store = MagicMock()
    vector_store.query.return_value = []
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.0] * 8
    metadata_store = MagicMock()
    metadata_store.agency_list = ["한국가스공사"]
    metadata_store.find_relevant_docs.return_value = []
    return (
        RAGRetriever(vector_store=vector_store, embedder=embedder, metadata_store=metadata_store),
        vector_store,
    )


def test_retriever_explicit_metadata_filter_bypasses_auto_extraction():
    retriever, vector_store = _make_retriever_with_mock_store()
    explicit = {"발주 기관": "한국가스공사", "공개연도": 2024}
    retriever.retrieve("아무 질문", metadata_filter=explicit)
    call = vector_store.query.call_args
    assert call.kwargs["where"] == explicit


def test_retriever_empty_dict_filter_means_no_filter():
    """metadata_filter={}는 자동 추출까지 비활성화 (Streamlit '필터 없음' 모드)."""
    retriever, vector_store = _make_retriever_with_mock_store()
    retriever.retrieve("한국가스공사 사업 알려줘", metadata_filter={})
    call = vector_store.query.call_args
    assert call.kwargs["where"] is None


def test_retriever_none_filter_falls_back_to_auto_extraction():
    """metadata_filter=None은 legacy 자동 추출 동작."""
    retriever, vector_store = _make_retriever_with_mock_store()
    retriever.retrieve("한국가스공사 사업", metadata_filter=None)
    call = vector_store.query.call_args
    # 자동 추출이 발주 기관을 잡았어야 함
    where = call.kwargs["where"]
    assert where is not None
    assert where.get("발주 기관") == "한국가스공사"


def test_retriever_without_reranker_expands_query_pool():
    retriever, vector_store = _make_retriever_with_mock_store()
    retriever.retrieve("질문", top_k=12)
    assert vector_store.query.call_args.kwargs["top_k"] == 36


# ---------------------------------------------------------------------------
# Phase 3: collection_name_for_config mode-aware isolation
# ---------------------------------------------------------------------------


def _make_runtime(exp_name: str, mode: str, explicit: str | None = None) -> RuntimeConfig:
    return RuntimeConfig(
        project=ProjectConfig(),
        provider=ProviderConfig(
            provider="openai",
            model="gpt-5-mini",
            embedding_model="text-embedding-3-small",
            collection_name=explicit,
        ),
        experiment=ExperimentConfig(name=exp_name, mode=mode),
    )


def test_collection_generation_only_shares_explicit_collection():
    rt = _make_runtime("generation-compare", "generation_only", "bidmate-shared")
    assert collection_name_for_config(rt) == "bidmate-shared"


def test_collection_full_rag_isolates_with_prefix_when_explicit():
    rt = _make_runtime("chunk_500_100", "full_rag", "bidmate-shared")
    assert collection_name_for_config(rt) == "chunk_500_100-bidmate-shared"


def test_collection_full_rag_isolates_via_default_format():
    rt = _make_runtime("chunk_500_100", "full_rag", None)
    name = collection_name_for_config(rt)
    assert "chunk_500_100" in name
    assert name == "bidmate-chunk_500_100-openai-text-embedding-3-small"


def test_collection_ad_hoc_preserves_legacy_explicit_name():
    rt = _make_runtime("ad-hoc", "full_rag", "bidmate-legacy")
    assert collection_name_for_config(rt) == "bidmate-legacy"


# ---------------------------------------------------------------------------
# Phase 4: persist_benchmark_summary append-or-replace
# ---------------------------------------------------------------------------


def _summary(run_id: str, hit: float) -> dict:
    return {
        "experiment_name": "exp",
        "run_id": run_id,
        "scenario": "openai",
        "provider_label": "openai:gpt-5-mini",
        "num_samples": 2,
        "avg_latency_ms": 1000.0,
        "total_cost_usd": 0.001,
        "hit_rate@5": hit,
    }


def test_persist_benchmark_summary_first_write_creates_file(tmp_path):
    persist_benchmark_summary([_summary("run-1", 0.5)], tmp_path, "exp")
    df = pd.read_parquet(tmp_path / "exp.parquet")
    assert len(df) == 1
    assert df.iloc[0]["run_id"] == "run-1"


def test_persist_benchmark_summary_appends_new_run_ids(tmp_path):
    persist_benchmark_summary([_summary("run-1", 0.5)], tmp_path, "exp")
    persist_benchmark_summary([_summary("run-2", 0.8)], tmp_path, "exp")
    persist_benchmark_summary([_summary("run-3", 1.0)], tmp_path, "exp")
    df = pd.read_parquet(tmp_path / "exp.parquet")
    assert set(df["run_id"]) == {"run-1", "run-2", "run-3"}
    assert len(df) == 3


def test_persist_benchmark_summary_replaces_existing_run_id(tmp_path):
    persist_benchmark_summary([_summary("run-1", 0.5)], tmp_path, "exp")
    persist_benchmark_summary([_summary("run-1", 0.9)], tmp_path, "exp")
    df = pd.read_parquet(tmp_path / "exp.parquet")
    assert len(df) == 1
    assert df.iloc[0]["hit_rate@5"] == 0.9  # 마지막 값으로 replace


def test_persist_benchmark_summary_provider_compare_scenario(tmp_path):
    """generation_compare처럼 같은 experiment에 여러 provider를 순차로 평가:
    이전에는 마지막 provider만 남았지만 이제 전부 보존되어야 함.
    """
    persist_benchmark_summary([_summary("nano-run", 0.6)], tmp_path, "generation-compare")
    persist_benchmark_summary([_summary("mini-run", 0.7)], tmp_path, "generation-compare")
    persist_benchmark_summary([_summary("full-run", 0.8)], tmp_path, "generation-compare")
    df = pd.read_parquet(tmp_path / "generation-compare.parquet")
    assert len(df) == 3
    assert set(df["run_id"]) == {"nano-run", "mini-run", "full-run"}
