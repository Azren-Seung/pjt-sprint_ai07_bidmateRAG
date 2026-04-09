"""CLI entrypoint for raw document ingestion."""

from __future__ import annotations

import argparse

from dotenv import load_dotenv
load_dotenv()

from bidmate_rag.pipelines.ingest import run_ingest_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse, clean, and chunk RFP documents.")
    parser.add_argument("--metadata-path", default="data/raw/metadata/data_list.csv")
    parser.add_argument("--raw-dir", default="data/raw/rfp")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--experiment-config", default=None,
                        help="실험 config (YAML). chunk_size/overlap 설정을 오버라이드")
    args = parser.parse_args()

    # experiment config에서 chunk 설정 로딩
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap

    if args.experiment_config:
        import yaml
        from pathlib import Path
        exp_cfg = yaml.safe_load(Path(args.experiment_config).read_text())
        chunk_size = exp_cfg.get("chunk_size", chunk_size)
        chunk_overlap = exp_cfg.get("chunk_overlap", chunk_overlap)
        # 실험별 output 디렉터리 분리
        exp_name = exp_cfg.get("name", "default")
        output_dir = f"data/processed/{exp_name}"
        print(f"실험 config: {args.experiment_config}")
    else:
        output_dir = args.output_dir

    print(f"청킹 설정: size={chunk_size}, overlap={chunk_overlap}")
    print(f"출력 경로: {output_dir}")

    outputs = run_ingest_pipeline(
        metadata_path=args.metadata_path,
        raw_dir=args.raw_dir,
        output_dir=output_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
