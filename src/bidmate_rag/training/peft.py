"""PEFT-ready utility helpers."""

from __future__ import annotations

from pathlib import Path


def build_sft_record(example: dict) -> dict[str, str]:
    instruction = example.get("instruction", "").strip()
    output = example.get("output", "").strip()
    return {"text": f"### Instruction:\n{instruction}\n\n### Response:\n{output}"}


def build_training_artifact_name(base_model: str, method: str) -> str:
    return f"{base_model.replace('/', '_')}-{method}"


def default_adapter_dir(output_root: str | Path, base_model: str, method: str) -> Path:
    return Path(output_root) / build_training_artifact_name(base_model, method)
