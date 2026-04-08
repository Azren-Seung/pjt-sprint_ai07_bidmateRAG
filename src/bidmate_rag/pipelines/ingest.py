"""Ingestion pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bidmate_rag.loaders.hwp_loader import parse_hwp
from bidmate_rag.loaders.metadata_loader import load_metadata_frame
from bidmate_rag.loaders.pdf_loader import parse_pdf
from bidmate_rag.preprocessing.chunker import (
    chunk_document,
    classify_agency,
    classify_domain,
    extract_tech_stack,
)
from bidmate_rag.preprocessing.cleaner import clean_text


def _default_parse(file_path: Path) -> dict:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(file_path)
    return parse_hwp(file_path)


def _public_year(value) -> int:
    text = str(value or "")
    return int(text[:4]) if len(text) >= 4 and text[:4].isdigit() else 0


def run_ingest_pipeline(
    metadata_path: str | Path,
    raw_dir: str | Path,
    output_dir: str | Path,
    parse_fn=None,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> dict[str, Path]:
    parser = parse_fn or _default_parse
    raw_root = Path(raw_dir)
    processed_root = Path(output_dir)
    processed_root.mkdir(parents=True, exist_ok=True)

    metadata_df = load_metadata_frame(metadata_path)

    parsed_rows: list[dict] = []
    for _, row in metadata_df.iterrows():
        file_path = raw_root / row["파일명"]
        parsed = parser(file_path)
        parsed_rows.append(parsed)

    parsed_df = metadata_df.copy()
    parsed_map = {row["파일명"]: row for row in parsed_rows}
    parsed_df["본문_마크다운"] = parsed_df["파일명"].map(lambda name: parsed_map[name]["텍스트"])
    parsed_df["본문_글자수"] = parsed_df["파일명"].map(lambda name: parsed_map[name]["글자수"])

    parsed_path = processed_root / "parsed_documents.parquet"
    parsed_df.to_parquet(parsed_path, index=False)

    cleaned_df = parsed_df.copy()
    cleaned_df["본문_정제"] = cleaned_df["본문_마크다운"].fillna("").map(clean_text)
    cleaned_df["정제_글자수"] = cleaned_df["본문_정제"].str.len()
    cleaned_df["기관유형"] = cleaned_df["발주 기관"].map(classify_agency)
    cleaned_df["사업도메인"] = [
        classify_domain(name, text)
        for name, text in zip(cleaned_df["사업명"], cleaned_df["본문_정제"], strict=False)
    ]
    cleaned_df["기술스택"] = cleaned_df["본문_정제"].map(extract_tech_stack)
    cleaned_df["공개연도"] = cleaned_df["공개 일자"].map(_public_year)

    cleaned_path = processed_root / "cleaned_documents.parquet"
    cleaned_df.to_parquet(cleaned_path, index=False)

    all_chunks = []
    for _, row in cleaned_df.iterrows():
        metadata = row.to_dict()
        metadata["doc_id"] = str(metadata.get("공고 번호") or metadata.get("파일명"))
        chunks = chunk_document(
            row["본문_정제"],
            metadata,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        all_chunks.extend(chunk.to_record() for chunk in chunks)

    chunk_df = pd.DataFrame(all_chunks)
    chunks_path = processed_root / "chunks.parquet"
    chunk_df.to_parquet(chunks_path, index=False)

    return {"parsed": parsed_path, "cleaned": cleaned_path, "chunks": chunks_path}
