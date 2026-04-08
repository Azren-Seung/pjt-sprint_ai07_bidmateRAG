"""CLI entrypoint for raw document ingestion."""

from __future__ import annotations

import argparse

from bidmate_rag.pipelines.ingest import run_ingest_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse, clean, and chunk RFP documents.")
    parser.add_argument("--metadata-path", default="data/raw/metadata/data_list.csv")
    parser.add_argument("--raw-dir", default="data/raw/rfp")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    args = parser.parse_args()

    outputs = run_ingest_pipeline(
        metadata_path=args.metadata_path,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
