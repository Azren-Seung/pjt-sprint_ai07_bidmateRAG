"""문서 수집 파이프라인.

원본 RFP 문서(HWP/PDF)를 파싱 → 텍스트 정제 → 청킹하여
parquet 파일로 저장하는 end-to-end 인제스트 파이프라인.
"""

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
    """파일 확장자에 따라 적절한 파서를 선택하여 문서를 파싱한다.

    Args:
        file_path: 원본 문서 경로.

    Returns:
        파싱 결과 딕셔너리 (텍스트, 글자수, 성공 여부 등).
    """
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(file_path)
    return parse_hwp(file_path)


def _public_year(value) -> int:
    """공개 일자 값에서 연도(4자리)를 추출한다.

    Args:
        value: 공개 일자 문자열 또는 숫자.

    Returns:
        연도 정수. 파싱 실패 시 0.
    """
    text = str(value or "")
    return int(text[:4]) if len(text) >= 4 and text[:4].isdigit() else 0


def run_ingest_pipeline(
    metadata_path: str | Path,
    raw_dir: str | Path,
    output_dir: str | Path,
    parse_fn=None,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    parsed_path:str | Path | None = None, #기존 파싱 결과 재사용 시지정
) -> dict[str, Path]:
    """인제스트 파이프라인 전체를 실행한다.

    1. 메타데이터 CSV 로딩
    2. 원본 문서 파싱 (HWP/PDF → 마크다운)
    3. 텍스트 정제 및 메타데이터 보강 (기관유형, 도메인, 기술스택)
    4. 청킹 → chunks.parquet 저장

    Args:
        metadata_path: 메타데이터 CSV 경로.
        raw_dir: 원본 RFP 문서 디렉터리.
        output_dir: parquet 산출물 저장 디렉터리.
        parse_fn: 커스텀 파서 함수 (기본: HWP/PDF 자동 선택).
        chunk_size: 청크 최대 글자수.
        chunk_overlap: 청크 간 겹침 글자수.

    Returns:
        산출물 이름-경로 딕셔너리 (parsed, cleaned, chunks).
    """
    parser = parse_fn or _default_parse
    raw_root = Path(raw_dir)
    processed_root = Path(output_dir)
    processed_root.mkdir(parents=True, exist_ok=True)

    # ── 1단계: 메타데이터 CSV 로딩 ──
    # data_list.csv에서 사업명, 발주기관, 파일명 등 기본 정보를 읽어온다
    metadata_df = load_metadata_frame(metadata_path)

    # ── 2단계: 원본 문서 파싱 (HWP/PDF → 마크다운 텍스트) ──
    # 각 문서를 kordoc(HWP) 또는 pdfplumber(PDF)로 텍스트 추출
    if parsed_path:
        # 기존 파싱 결과 재사용 (파싱 건너뜀)
        print(f"파싱 결과 재사용: {parsed_path}")
        parsed_df = pd.read_parquet(parsed_path)
    else:
        # 기존 파싱 로직
        parsed_rows: list[dict] = []
        total = len(metadata_df)
        for i, (_, row) in enumerate(metadata_df.iterrows(), 1):
            file_path = raw_root / row["파일명"]
            print(f"[{i}/{total}] 파싱 중: {row['파일명']}", end=" ... ")
            try:
                parsed = parser(file_path)
            except Exception as exc:
                parsed = {
                    "파일명": row["파일명"],
                    "파서": "error",
                    "텍스트": "",
                    "글자수": 0,
                    "성공": False,
                    "에러": str(exc),
                }
            status = "OK" if parsed.get("성공") else f"실패({parsed.get('에러', '?')})"
            print(f"{status} ({parsed.get('글자수', 0):,}자)")
            parsed_rows.append(parsed)

        parsed_df = metadata_df.copy()
        parsed_map = {row["파일명"]: row for row in parsed_rows}
        parsed_df["본문_마크다운"] = parsed_df["파일명"].map(lambda name: parsed_map[name]["텍스트"])
        parsed_df["본문_글자수"] = parsed_df["파일명"].map(lambda name: parsed_map[name]["글자수"])

        parsed_path_out = processed_root / "parsed_documents.parquet"
        parsed_df.to_parquet(parsed_path_out, index=False)
    # ── 3단계: 텍스트 정제 + 메타데이터 보강 ──
    # 노이즈 제거, 기관유형/도메인/기술스택 자동 분류
    cleaned_df = parsed_df.copy()
    cleaned_df["본문_정제"] = cleaned_df["본문_마크다운"].fillna("").map(clean_text)
    cleaned_df["정제_글자수"] = cleaned_df["본문_정제"].str.len()
    cleaned_df["기관유형"] = cleaned_df["발주 기관"].map(classify_agency)       # 대학교, 공기업 등
    cleaned_df["사업도메인"] = [                                                # 교육, 안전, 의료 등
        classify_domain(name, text)
        for name, text in zip(cleaned_df["사업명"], cleaned_df["본문_정제"], strict=False)
    ]
    cleaned_df["기술스택"] = cleaned_df["본문_정제"].map(extract_tech_stack)     # AI, 클라우드 등
    cleaned_df["공개연도"] = cleaned_df["공개 일자"].map(_public_year)

    # 정제된 문서 전체를 cleaned_documents.parquet에 저장 (원문 보관용)
    cleaned_path = processed_root / "cleaned_documents.parquet"
    cleaned_df.to_parquet(cleaned_path, index=False)

    # ── 4단계: 청킹 → chunks.parquet ──
    # 정제된 본문을 검색 가능한 작은 조각(청크)으로 분할
    # 본문 전체(본문_마크다운, 본문_정제)는 청크에 넣지 않음
    # → 이미 cleaned_documents.parquet에 보관되어 있고,
    #   청크마다 복사하면 파일 크기가 수 GB로 커져 임베딩 시 메모리/토큰 한도 초과
    skip_cols = {"본문_마크다운", "본문_정제", "본문_글자수", "정제_글자수"}
    all_chunks = []
    for _, row in cleaned_df.iterrows():
        # 가벼운 메타데이터만 청크에 포함 (사업명, 발주기관, 도메인 등)
        metadata = {k: v for k, v in row.to_dict().items() if k not in skip_cols}
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
    parsed_out = Path(parsed_path) if parsed_path else parsed_path_out


    return {"parsed": parsed_out, "cleaned": cleaned_path, "chunks": chunks_path}
