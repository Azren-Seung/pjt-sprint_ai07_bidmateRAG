from pathlib import Path

from bidmate_rag.config.settings import load_runtime_config


def test_load_runtime_config_merges_base_provider_and_experiment(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    provider = tmp_path / "provider.yaml"
    experiment = tmp_path / "experiment.yaml"

    base.write_text("default_retrieval_top_k: 5\ndefault_chunk_size: 1000\n")
    provider.write_text("provider: openai\nmodel: gpt-5-mini\nscenario: scenario_b\n")
    experiment.write_text("name: generation-compare\nmode: generation_only\nretrieval_top_k: 8\n")

    config = load_runtime_config(base, provider, experiment)

    assert config.project.default_retrieval_top_k == 5
    assert config.provider.model == "gpt-5-mini"
    assert config.experiment.retrieval_top_k == 8
    assert config.experiment.mode == "generation_only"
