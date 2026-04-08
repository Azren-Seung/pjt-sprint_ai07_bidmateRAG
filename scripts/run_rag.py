"""CLI entrypoint for one-off RAG questions."""

from __future__ import annotations

import argparse
from uuid import uuid4

from dotenv import load_dotenv
load_dotenv()

from bidmate_rag.pipelines.runtime import build_runtime_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single RAG query.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--provider-config", required=True)
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--experiment-config", default=None)
    args = parser.parse_args()

    pipeline, runtime, embedder, _ = build_runtime_pipeline(
        base_config_path=args.base_config,
        provider_config_path=args.provider_config,
        experiment_config_path=args.experiment_config,
    )
    result = pipeline.answer(
        args.question,
        question_id=f"q-{uuid4().hex[:8]}",
        scenario=runtime.provider.scenario or runtime.provider.provider,
        run_id=f"cli-{uuid4().hex[:8]}",
        embedding_provider=embedder.provider_name,
        embedding_model=embedder.model_name,
    )
    print(result.answer)


if __name__ == "__main__":
    main()
