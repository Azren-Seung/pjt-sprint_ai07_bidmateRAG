"""단일 질문 RAG CLI.

터미널에서 질문 하나를 던지고 답변을 받는 스크립트.
UI 없이 빠르게 리트리버 + LLM 응답을 확인할 때 사용한다.

사용 예시::

    uv run python scripts/run_rag.py \\
        --provider-config configs/providers/openai_gpt5mini.yaml \\
        --question "국민연금공단 이러닝시스템 요구사항 정리해줘"
"""

from __future__ import annotations

import argparse
from uuid import uuid4

from dotenv import load_dotenv
load_dotenv()

from bidmate_rag.pipelines.runtime import build_runtime_pipeline


def main() -> None:
    """CLI 인자를 파싱하고 RAG 파이프라인으로 질문-답변을 실행"""
    # CLI 인자 정의
    parser = argparse.ArgumentParser(description="Run a single RAG query.")
    parser.add_argument("--question", required=True)           # 질문 문자열
    parser.add_argument("--provider-config", required=True)    # LLM/임베딩 설정 YAML
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--experiment-config", default=None)   # 실험별 설정 (선택)
    args = parser.parse_args()

    # 설정 로딩 → RAG 파이프라인 구성 (리트리버 + LLM)
    pipeline, runtime, embedder, _ = build_runtime_pipeline(
        base_config_path=args.base_config,
        provider_config_path=args.provider_config,
        experiment_config_path=args.experiment_config,
    )

    # 질문을 파이프라인에 전달 → 벡터 검색 → LLM 답변 생성
    result = pipeline.answer(
        args.question,
        question_id=f"q-{uuid4().hex[:8]}",        # 랜덤 질문 ID
        scenario=runtime.provider.scenario or runtime.provider.provider,
        run_id=f"cli-{uuid4().hex[:8]}",            # 랜덤 실행 ID
        embedding_provider=embedder.provider_name,
        embedding_model=embedder.model_name,
    )

    # 최종 답변 출력
    print(result.answer)


if __name__ == "__main__":
    main()
