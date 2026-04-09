from pathlib import Path

import pandas as pd

from bidmate_rag.pipelines.ingest import run_ingest_pipeline


def test_run_ingest_pipeline_creates_parsed_cleaned_and_chunk_outputs(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "rfp"
    metadata_dir = tmp_path / "raw" / "metadata"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)

    sample_file = raw_dir / "sample.hwp"
    sample_file.write_text("dummy", encoding="utf-8")

    pd.DataFrame(
        [
            {
                "공고 번호": "20240001",
                "공고 차수": 0,
                "사업명": "교육 플랫폼 고도화",
                "사업 금액": 300000000,
                "발주 기관": "국민연금공단",
                "공개 일자": "2024-10-01",
                "입찰 참여 시작일": "",
                "입찰 참여 마감일": "",
                "사업 요약": "이러닝 시스템 고도화",
                "파일형식": "hwp",
                "파일명": "sample.hwp",
            }
        ]
    ).to_csv(metadata_dir / "data_list.csv", index=False)

    def fake_parse(file_path: Path) -> dict:
        return {
            "파일명": file_path.name,
            "텍스트": (
                "Warning: test\n# 개요\n교육 플랫폼 구축<br>세부 내용\n"
                "# 예산\n| 항목 | 값 |\n| --- | --- |\n| 인건비 | 3억 |"
            ),
            "글자수": 120,
            "성공": True,
            "파서": "fake",
            "에러": None,
        }

    outputs = run_ingest_pipeline(
        metadata_path=metadata_dir / "data_list.csv",
        raw_dir=raw_dir,
        output_dir=output_dir,
        parse_fn=fake_parse,
    )

    assert outputs["parsed"].exists()
    assert outputs["cleaned"].exists()
    assert outputs["chunks"].exists()

    cleaned_df = pd.read_parquet(outputs["cleaned"])
    chunk_df = pd.read_parquet(outputs["chunks"])

    assert cleaned_df.loc[0, "본문_정제"].startswith("# 개요")
    assert "사업도메인" in chunk_df.columns
    assert chunk_df.loc[0, "발주 기관"] == "국민연금공단"
