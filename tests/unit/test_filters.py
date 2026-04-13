from bidmate_rag.retrieval.filters import (
    extract_metadata_filters,
    extract_range_filters,
    extract_section_hint,
    is_comparison_query,
    should_boost_tables,
)


def test_extract_metadata_filters_prefers_exact_agency_match() -> None:
    filters = extract_metadata_filters(
        query="국민연금공단 이러닝시스템 요구사항을 정리해줘",
        agency_list=["국민연금공단", "기초과학연구원"],
    )

    assert filters == {"발주 기관": "국민연금공단"}


def test_extract_metadata_filters_uses_domain_when_agency_absent() -> None:
    filters = extract_metadata_filters(
        query="교육 관련 사업 찾아줘",
        agency_list=["국민연금공단"],
    )

    assert filters == {"사업도메인": "교육/학습"}


def test_extract_range_and_section_filters() -> None:
    range_filters = extract_range_filters("2024년 5억 이상 사업의 요구사항 알려줘")

    assert range_filters == {"사업 금액": {"$gte": 500000000}, "공개연도": 2024}
    assert extract_section_hint("보안 요구사항 알려줘") == "보안"
    assert should_boost_tables("예산과 일정 표로 정리해줘") is True


def test_is_comparison_query_detects_comparison_and_shared_keywords() -> None:
    assert is_comparison_query("국민연금공단과 기초과학연구원 사업을 비교해줘") is True
    assert is_comparison_query("두 기관의 차액과 공통 요구사항을 각각 알려줘") is True
    assert is_comparison_query("국민연금공단 요구사항 알려줘") is False
