"""벡터 인덱스 생성 CLI.

chunks.parquet를 읽어 임베딩을 생성하고 ChromaDB에 저장한다.

사용 예시::

    uv run python scripts/build_index.py \\
        --provider-config configs/providers/openai_gpt5mini.yaml
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv
load_dotenv()

from bidmate_rag.config.settings import load_runtime_config
from bidmate_rag.pipelines.build_index import build_index_from_parquet
from bidmate_rag.pipelines.runtime import collection_name_for_config
from bidmate_rag.providers.llm.registry import build_embedding_provider
from bidmate_rag.retrieval.vector_store import ChromaVectorStore


def main() -> None:
    """설정 파일을 파싱하고 벡터 인덱스를 생성한다.

    흐름:
        1. base + provider + experiment 설정 로딩
        2. 임베딩 프로바이더 생성 (OpenAI, HF 등)
        3. ChromaDB 벡터 스토어 연결
        4. chunks.parquet → 임베딩 → ChromaDB 저장
    """
    # CLI 인자 파싱
    parser = argparse.ArgumentParser(description="Build a Chroma index from chunks.parquet.")
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--provider-config", required=True)
    parser.add_argument("--experiment-config", default=None)
    parser.add_argument("--chunks-path", default="data/processed/chunks.parquet")
    parser.add_argument("--persist-dir", default="artifacts/chroma_db")
    parser.add_argument("--min-chars", type=int, default=50)
    args = parser.parse_args()

    # 설정 로딩 → 프로바이더 생성 → 인덱스 빌드
    runtime = load_runtime_config(args.base_config, args.provider_config, args.experiment_config)
    embedder = build_embedding_provider(runtime.provider)
    vector_store = ChromaVectorStore(args.persist_dir, collection_name_for_config(runtime))
    stats = build_index_from_parquet(
        args.chunks_path, embedder=embedder, vector_store=vector_store, min_chars=args.min_chars
    )
    print(stats)


if __name__ == "__main__":
    main()
