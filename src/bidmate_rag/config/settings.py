"""Runtime settings helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    project_name: str = "bidmate-rag"
    default_retrieval_top_k: int = 5
    default_chunk_size: int = 1000
    default_chunk_overlap: int = 150


class ProviderConfig(BaseModel):
    provider: str
    model: str
    api_base: str | None = None
    scenario: str | None = None
    embedding_model: str | None = None
    collection_name: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    name: str
    mode: str = "full_rag"
    retrieval_top_k: int | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    provider_configs: list[str] = Field(default_factory=list)


class RuntimeConfig(BaseModel):
    project: ProjectConfig
    provider: ProviderConfig
    experiment: ExperimentConfig


def _load_yaml(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data or {}


def load_runtime_config(
    base_config_path: str | Path,
    provider_config_path: str | Path,
    experiment_config_path: str | Path | None = None,
) -> RuntimeConfig:
    base_data = _load_yaml(base_config_path)
    provider_data = _load_yaml(provider_config_path)
    experiment_data = _load_yaml(experiment_config_path)

    project = ProjectConfig.model_validate(base_data)
    experiment = ExperimentConfig.model_validate(experiment_data or {"name": "ad-hoc"})
    if experiment.retrieval_top_k is None:
        experiment.retrieval_top_k = project.default_retrieval_top_k
    if experiment.chunk_size is None:
        experiment.chunk_size = project.default_chunk_size
    if experiment.chunk_overlap is None:
        experiment.chunk_overlap = project.default_chunk_overlap

    provider_known_keys = {
        "provider",
        "model",
        "api_base",
        "scenario",
        "embedding_model",
        "collection_name",
    }
    provider_extra = {
        key: value for key, value in provider_data.items() if key not in provider_known_keys
    }
    provider = ProviderConfig.model_validate({**provider_data, "extra": provider_extra})

    return RuntimeConfig(project=project, provider=provider, experiment=experiment)
