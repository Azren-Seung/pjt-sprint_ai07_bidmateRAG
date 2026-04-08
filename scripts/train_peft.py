"""CLI scaffold for PEFT fine-tuning preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bidmate_rag.training.peft import build_sft_record, default_adapter_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PEFT training artifacts.")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--method", default="lora")
    parser.add_argument("--output-root", default="artifacts/training")
    args = parser.parse_args()

    output_dir = default_adapter_dir(args.output_root, args.base_model, args.method)
    output_dir.mkdir(parents=True, exist_ok=True)
    formatted_path = output_dir / "sft_train.jsonl"

    with (
        Path(args.train_jsonl).open("r", encoding="utf-8") as src,
        formatted_path.open("w", encoding="utf-8") as dst,
    ):
        for line in src:
            if not line.strip():
                continue
            record = build_sft_record(json.loads(line))
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(formatted_path)
    print("Optional ML dependencies are required to run the actual fine-tuning step.")


if __name__ == "__main__":
    main()
