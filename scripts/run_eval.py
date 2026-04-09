"""RAG 평가 벤치마크 CLI.

평가 데이터셋(CSV/JSON)을 로딩하여 RAG 파이프라인으로 답변을 생성하고
hit_rate, MRR, nDCG 등 리트리버 성능 지표를 산출한다.

사용 예시::

    uv run python scripts/run_eval.py \\
        --evaluation-path data/eval/eval_batch_01.csv \\
        --provider-config configs/providers/openai_gpt5mini.yaml
"""

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
    """평가셋을 로딩하고 벤치마크를 실행하여 결과를 저장"""
    # CLI 인자 정의
    parser = argparse.ArgumentParser(description="Run the shared benchmark runner.")
    parser.add_argument("--evaluation-path", required=True)      # 평가 데이터셋 (CSV/JSON)
    parser.add_argument("--provider-config", required=True)      # LLM/임베딩 설정 YAML
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--experiment-config", default=None)
    parser.add_argument("--runs-dir", default="artifacts/logs/runs")           # 실행 로그 저장 경로
    parser.add_argument("--benchmarks-dir", default="artifacts/logs/benchmarks")  # 벤치마크 요약 저장 경로
    args = parser.parse_args()

    # RAG 파이프라인 구성
    pipeline, runtime, embedder, _ = build_runtime_pipeline(
        base_config_path=args.base_config,
        provider_config_path=args.provider_config,
        experiment_config_path=args.experiment_config,
    )

    # 평가 데이터셋 로딩 (질문 + 정답 문서 ID)
    samples = load_eval_samples(args.evaluation_path)
    run_id = f"bench-{uuid4().hex[:8]}"

    # 각 평가 질문에 대해 RAG 답변을 생성하는 함수
    def answer_fn(sample):
        return pipeline.answer(
            sample.question,
            question_id=sample.question_id,
            scenario=runtime.provider.scenario or runtime.provider.provider,
            run_id=run_id,
            embedding_provider=embedder.provider_name,
            embedding_model=embedder.model_name,
        )

    # 벤치마크 실행: 전체 평가셋에 대해 답변 생성
    benchmark = BenchmarkRunner(answer_fn).run(
        experiment_name=runtime.experiment.name,
        scenario=runtime.provider.scenario or runtime.provider.provider,
        provider_label=f"{runtime.provider.provider}:{runtime.provider.model}",
        samples=samples,
    )

    # 리트리버 성능 지표 계산 (hit_rate, MRR, nDCG)
    retrieval_metrics = {"hit_rate@5": 0.0, "mrr": 0.0, "ndcg@5": 0.0}
    scored = 0
    for sample, result in zip(samples, benchmark.results, strict=False):
        if not sample.expected_doc_ids:  # 정답 문서가 없는 샘플은 건너뜀
            continue
        hit = calc_hit_rate(result.retrieved_chunks, sample.expected_doc_ids, k=5)
        mrr = calc_mrr(result.retrieved_chunks, sample.expected_doc_ids)
        ndcg = calc_ndcg(result.retrieved_chunks, sample.expected_doc_ids, k=5)
        if hit is not None:
            retrieval_metrics["hit_rate@5"] += hit
            retrieval_metrics["mrr"] += mrr or 0.0
            retrieval_metrics["ndcg@5"] += ndcg or 0.0
            scored += 1

    # 평균 지표 계산 후 벤치마크에 추가
    if scored:
        benchmark.metrics.update(
            {key: round(value / scored, 4) for key, value in retrieval_metrics.items()}
        )

    # 결과 저장: 개별 실행 로그 + 벤치마크 요약
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
