"""Tests for tracking/markdown_report.py."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bidmate_rag.tracking.markdown_report import (
    load_report_data,
    render_markdown,
    write_report,
)


def _make_fixture(
    tmp_path: Path,
    *,
    with_meta: bool = True,
    with_embedding: bool = True,
    judge_skipped: bool = False,
    llm_model: str = "gpt-5-mini",
) -> Path:
    runs_dir = tmp_path / "runs"
    benchmarks_dir = tmp_path / "benchmarks"
    embeddings_dir = tmp_path / "embeddings"
    runs_dir.mkdir()
    benchmarks_dir.mkdir()
    embeddings_dir.mkdir()

    run_id = "bench-test1234"
    exp_name = "test-exp"

    # JSONL with two questions
    jsonl_path = runs_dir / f"{run_id}.jsonl"
    rows = [
        {
            "question_id": "q1",
            "question": "Q1",
            "scenario": "openai",
            "run_id": run_id,
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "llm_provider": "openai",
            "llm_model": "gpt-5-mini",
            "answer": "A1",
            "retrieved_chunks": [],
            "latency_ms": 1500.0,
            "token_usage": {"prompt": 100, "completion": 50, "total": 150},
            "cost_usd": 0.0012,
            "judge_scores": {
                "faithfulness": 0.9,
                "answer_relevance": 0.85,
                "context_precision": 0.7,
                "context_recall": 0.8,
            },
        },
        {
            "question_id": "q2",
            "question": "Q2",
            "scenario": "openai",
            "run_id": run_id,
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "llm_provider": "openai",
            "llm_model": "gpt-5-mini",
            "answer": "A2",
            "retrieved_chunks": [],
            "latency_ms": 2500.0,
            "token_usage": {"prompt": 200, "completion": 100, "total": 300},
            "cost_usd": 0.0024,
            "judge_scores": {},
        },
    ]
    jsonl_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )

    # Parquet summary
    summary_row = {
        "experiment_name": exp_name,
        "run_id": run_id,
        "scenario": "openai",
        "provider_label": "openai:gpt-5-mini",
        "num_samples": 2,
        "avg_latency_ms": 2000.0,
        "total_cost_usd": 0.0036,
        "hit_rate@5": 0.85,
        "mrr": 0.72,
        "ndcg@5": 0.79,
        "faithfulness": 0.9,
        "answer_relevance": 0.85,
        "context_precision": 0.7,
        "context_recall": 0.8,
    }
    pd.DataFrame([summary_row]).to_parquet(
        benchmarks_dir / f"{exp_name}.parquet", index=False
    )

    # meta.json
    if with_meta:
        meta = {
            "run_id": run_id,
            "experiment_name": exp_name,
            "timestamp_utc": "2026-04-09T05:32:11+00:00",
            "timestamp_kst": "2026-04-09 14:32:11",
            "git": {"commit": "abc1234def", "commit_short": "abc1234", "branch": "main", "dirty": False},
            "configs": {
                "base": "configs/base.yaml",
                "provider": "configs/providers/openai_gpt5mini.yaml",
                "experiment": "configs/experiments/test-exp.yaml",
            },
            "config_snapshot": {
                "project": {
                    "default_chunk_size": 800,
                    "default_chunk_overlap": 100,
                    "default_retrieval_top_k": 5,
                },
                "provider": {
                    "provider": "openai",
                    "model": llm_model,
                    "embedding_model": "text-embedding-3-small",
                    "scenario": "openai",
                    "collection_name": "test-collection",
                },
                "experiment": {
                    "name": exp_name,
                    "chunk_size": 800,
                    "chunk_overlap": 100,
                    "retrieval_top_k": 5,
                },
            },
            "eval_path": "data/eval/eval_batch_02.csv",
            "collection_name": "test-collection",
            "judge_skipped": judge_skipped,
            "judge_total_cost_usd": 0.0 if judge_skipped else 0.0008,
            "judge_total_tokens": 0 if judge_skipped else 400,
        }
        (runs_dir / f"{run_id}.meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # Embedding meta
    if with_embedding:
        emb_meta = {
            "collection_name": "test-collection",
            "embedding_model": "text-embedding-3-small",
            "total_tokens": 50_000,
            "total_cost_usd": 0.001,
            "num_chunks": 100,
            "built_at": "2026-04-09T03:12:08+00:00",
        }
        (embeddings_dir / "test-collection.json").write_text(
            json.dumps(emb_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return tmp_path


def test_load_report_data_full(tmp_path):
    _make_fixture(tmp_path)
    data = load_report_data(
        run_id="bench-test1234",
        runs_dir=tmp_path / "runs",
        benchmarks_dir=tmp_path / "benchmarks",
        embeddings_dir=tmp_path / "embeddings",
    )
    assert data.experiment_name == "test-exp"
    assert len(data.results) == 2
    assert data.summary_row["hit_rate@5"] == 0.85
    assert data.embedding_meta is not None
    assert data.embedding_meta["total_cost_usd"] == 0.001
    assert data.meta["judge_total_cost_usd"] == 0.0008


def test_render_markdown_includes_key_sections(tmp_path):
    _make_fixture(tmp_path)
    data = load_report_data(
        run_id="bench-test1234",
        runs_dir=tmp_path / "runs",
        benchmarks_dir=tmp_path / "benchmarks",
        embeddings_dir=tmp_path / "embeddings",
    )
    md = render_markdown(data)
    # 노션 속성 영역
    assert "📋 노션 속성" in md
    assert "bench-test1234" in md
    assert "test-exp" in md
    # 자동 생성 본문
    assert "🤖 자동 생성 본문" in md
    assert "Hit Rate@5" in md
    assert "0.8500" in md  # hit_rate
    # 사람 작성 영역
    assert "✍️ 사람이 작성하는 영역" in md
    assert "## 1. 가설" in md
    # 비용 표시
    assert "0.0036" in md  # generation cost
    assert "0.0010" in md  # embedding cost
    assert "0.0008" in md  # judge cost
    # 본문 표 cost 명칭이 노션 속성과 동일하게 "Cost (USD)"로 통일
    assert "**Cost (USD)**" in md
    # 토큰 합계
    assert "300" in md or "150" in md
    # git
    assert "abc1234" in md
    # gpt-5-mini는 reasoning 주의 문구가 표시되어야 함
    assert "gpt-5 계열은 reasoning tokens" in md


def test_render_markdown_handles_missing_embedding(tmp_path):
    _make_fixture(tmp_path, with_embedding=False)
    data = load_report_data(
        run_id="bench-test1234",
        runs_dir=tmp_path / "runs",
        benchmarks_dir=tmp_path / "benchmarks",
        embeddings_dir=tmp_path / "embeddings",
    )
    md = render_markdown(data)
    assert "임베딩 비용 미수집" in md
    assert data.embedding_meta is None


def test_render_markdown_handles_missing_meta(tmp_path):
    _make_fixture(tmp_path, with_meta=False)
    data = load_report_data(
        run_id="bench-test1234",
        runs_dir=tmp_path / "runs",
        benchmarks_dir=tmp_path / "benchmarks",
        embeddings_dir=tmp_path / "embeddings",
    )
    md = render_markdown(data)
    # meta가 없어도 렌더링은 성공해야 함
    assert "bench-test1234" in md
    assert "test-exp" in md
    # config 정보가 비어있어야 함
    assert "(미기록)" in md or "configs/base.yaml" not in md


def test_write_report_creates_file(tmp_path):
    _make_fixture(tmp_path)
    data = load_report_data(
        run_id="bench-test1234",
        runs_dir=tmp_path / "runs",
        benchmarks_dir=tmp_path / "benchmarks",
        embeddings_dir=tmp_path / "embeddings",
    )
    out = write_report(data, output_dir=tmp_path / "reports")
    assert out.exists()
    assert out.name == "test-exp_bench-test1234.md"
    assert "📋 노션 속성" in out.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Polish UX 회귀 방지
# ---------------------------------------------------------------------------


def test_judge_skipped_shows_미실행_in_judge_cost(tmp_path):
    """--skip-judge로 돌린 run은 'Judge 비용 (USD) | (미실행)' 으로 표시."""
    _make_fixture(tmp_path, judge_skipped=True)
    data = load_report_data(
        run_id="bench-test1234",
        runs_dir=tmp_path / "runs",
        benchmarks_dir=tmp_path / "benchmarks",
        embeddings_dir=tmp_path / "embeddings",
    )
    md = render_markdown(data)
    assert "| Judge 비용 (USD) | (미실행) |" in md


def test_gpt5_warning_only_for_gpt5_models(tmp_path):
    """non-gpt-5 모델에서는 reasoning 주의 문구가 나타나지 않아야 함."""
    _make_fixture(tmp_path, llm_model="gpt-4o-mini")
    data = load_report_data(
        run_id="bench-test1234",
        runs_dir=tmp_path / "runs",
        benchmarks_dir=tmp_path / "benchmarks",
        embeddings_dir=tmp_path / "embeddings",
    )
    md = render_markdown(data)
    assert "gpt-5 계열은 reasoning tokens" not in md


def test_no_blank_line_clutter_when_no_warnings(tmp_path):
    """warnings/gpt5 주의문이 모두 비어있을 때 본문에 빈 줄 3개 이상이 없어야 함."""
    _make_fixture(tmp_path, llm_model="gpt-4o-mini")  # gpt5_warning 비활성
    data = load_report_data(
        run_id="bench-test1234",
        runs_dir=tmp_path / "runs",
        benchmarks_dir=tmp_path / "benchmarks",
        embeddings_dir=tmp_path / "embeddings",
    )
    md = render_markdown(data)
    # Cost 표 직후 곧바로 "## 리소스 링크"가 와야 함 (사이에 빈 줄 3개 이상 없어야)
    assert "\n\n\n\n" not in md
