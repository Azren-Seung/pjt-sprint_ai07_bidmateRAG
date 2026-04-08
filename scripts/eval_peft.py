"""CLI scaffold for PEFT adapter evaluation."""

from __future__ import annotations

import argparse

from bidmate_rag.training.peft import build_training_artifact_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Describe the PEFT artifact that should be evaluated."
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--method", default="lora")
    args = parser.parse_args()

    print(build_training_artifact_name(args.base_model, args.method))
    print("Run the benchmark pipeline with the adapter-aware provider config after training.")


if __name__ == "__main__":
    main()
