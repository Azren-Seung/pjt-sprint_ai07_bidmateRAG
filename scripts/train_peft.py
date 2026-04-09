"""PEFT 파인튜닝 데이터 준비 CLI.

학습용 JSONL을 SFT 포맷으로 변환하여 어댑터 디렉터리에 저장한다.

사용 예시::

    uv run python scripts/train_peft.py \\
        --train-jsonl data/train.jsonl \\
        --base-model gpt-5-mini
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bidmate_rag.training.peft import build_sft_record, default_adapter_dir


def main() -> None:
    """학습 JSONL을 SFT 포맷으로 변환하여 저장"""
    # CLI 인자 정의
    parser = argparse.ArgumentParser(description="Prepare PEFT training artifacts.")
    parser.add_argument("--train-jsonl", required=True)                  # 학습 데이터 JSONL 경로
    parser.add_argument("--base-model", required=True)                   # 기반 모델명
    parser.add_argument("--method", default="lora")                      # PEFT 방식 (기본: lora)
    parser.add_argument("--output-root", default="artifacts/training")   # 출력 루트 경로
    args = parser.parse_args()

    # 어댑터 디렉터리 생성 (artifacts/training/<모델>/<방식>/)
    output_dir = default_adapter_dir(args.output_root, args.base_model, args.method)
    output_dir.mkdir(parents=True, exist_ok=True)
    formatted_path = output_dir / "sft_train.jsonl"

    # 원본 JSONL → SFT 포맷으로 변환하여 저장
    with (
        Path(args.train_jsonl).open("r", encoding="utf-8") as src,
        formatted_path.open("w", encoding="utf-8") as dst,
    ):
        for line in src:
            if not line.strip():
                continue
            record = build_sft_record(json.loads(line))  # SFT 학습 형식으로 변환
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 결과 경로 및 안내 출력
    print(formatted_path)
    print("Optional ML dependencies are required to run the actual fine-tuning step.")


if __name__ == "__main__":
    main()
