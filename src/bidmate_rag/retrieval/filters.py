"""Metadata and heuristic retrieval filters."""

from __future__ import annotations

import re

DOMAIN_KEYWORDS = {
    "교육/학습": ["교육", "학습", "이러닝", "학사", "LMS", "연수", "대학"],
    "안전/재난": ["안전", "재난", "방재", "관제", "선량", "방사선"],
    "웹/포털": ["홈페이지", "포털", "웹", "온라인서비스"],
    "경영/행정": ["ERP", "그룹웨어", "경영", "인사", "회계", "오피스"],
    "공간정보/GIS": ["GIS", "지도", "공간정보", "측량", "수문", "관개"],
    "의료/바이오": ["의료", "건강", "바이오", "병원", "보험"],
    "ISP/컨설팅": ["ISP", "전략", "컨설팅", "타당성"],
    "AI/데이터": ["AI", "인공지능", "빅데이터", "데이터분석", "머신러닝"],
    "교통/물류": ["버스", "교통", "BIS", "ITS", "물류"],
    "농축수산": ["축산", "농업", "수산", "어촌"],
    "문화/콘텐츠": ["문화", "예술", "박물관", "아카이브", "영화"],
    "복지/사회서비스": ["복지", "돌봄", "사회보험", "서민금융"],
}

AGENCY_TYPE_KEYWORDS = {
    "대학교": ["대학", "학교"],
    "공기업/준정부기관": ["공사", "공단", "진흥원"],
    "지방자치단체": ["시청", "군청", "구청", "지자체"],
    "연구기관": ["연구원", "연구소"],
}

SECTION_KEYWORDS = {
    "보안": ["보안", "개인정보", "암호", "접근통제"],
    "예산": ["예산", "금액", "비용", "산출", "단가", "대금"],
    "일정": ["일정", "기간", "착수", "완료", "마감"],
    "평가": ["평가", "배점", "선정", "심사", "기준"],
    "인력": ["인력", "투입", "PM", "기술자", "조직"],
    "요구사항": ["요구사항", "기능", "성능", "요건", "조건"],
    "사업개요": ["목적", "배경", "왜", "개요", "일반", "기간", "규모"],
}

TABLE_KEYWORDS = ["요구사항", "정리", "목록", "예산", "금액", "일정", "배점", "기준", "표"]
RELEASE_KEYWORDS = ["다른 기관", "다른 사업", "그 외", "외에", "이외", "말고"]
COMPARISON_KEYWORDS = [
    "비교",
    "차이",
    "차액",
    "합산",
    "더 큰",
    "더 작은",
    "공통",
    "대비",
]
PER_SOURCE_KEYWORDS = ["각각"]
MULTI_SOURCE_HINT_KEYWORDS = [
    "각 기관",
    "기관별",
    "기관마다",
    "두 기관",
    "양 기관",
    "각 사업",
    "사업별",
    "사업마다",
    "두 사업",
    "양 사업",
]


def extract_matched_agencies(query: str, agency_list: list[str]) -> list[str]:
    """쿼리에 명시적으로 언급된 발주기관 목록을 추출."""
    matched_agencies: list[str] = []
    for agency in agency_list:
        short = agency.replace("(주)", "").replace("㈜", "").strip()
        for part in [short, short[:6], short[:4]]:
            if len(part) >= 3 and part in query:
                if agency not in matched_agencies:
                    matched_agencies.append(agency)
                break
    return matched_agencies


def extract_metadata_filters(
    query: str, agency_list: list[str], chat_history: list[dict] | None = None
) -> dict | None:
    """쿼리에서 발주기관·도메인·기관유형 메타데이터 필터를 추출

    Args:
        query: 사용자 질의 문자열.
        agency_list: 전체 발주기관 목록.
        chat_history: 이전 대화 이력.

    Returns:
        Chroma where 필터 딕셔너리 또는 None.
    """
    if any(keyword in query for keyword in RELEASE_KEYWORDS):
        return None
    matched_agencies = extract_matched_agencies(query, agency_list)
    if len(matched_agencies) >= 2:
        return None
    if len(matched_agencies) == 1:
        return {"발주 기관": matched_agencies[0]}
    where: dict[str, str] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(keyword in query for keyword in keywords):
            where["사업도메인"] = domain
            break
    if "사업도메인" not in where:
        for agency_type, keywords in AGENCY_TYPE_KEYWORDS.items():
            if any(keyword in query for keyword in keywords):
                where["기관유형"] = agency_type
                break
    if where:
        return where
    if chat_history:
        for message in reversed(chat_history):
            if message.get("filter"):
                return message["filter"]
    return None


def extract_range_filters(query: str) -> dict | None:
    """쿼리에서 금액·연도 범위 필터를 추출

    Args:
        query: 사용자 질의 문자열.

    Returns:
        범위 필터 딕셔너리 또는 None.
    """
    where: dict[str, object] = {}
    lower = re.search(r"(\d+)억\s*이상", query)
    if lower:
        where["사업 금액"] = {"$gte": int(lower.group(1)) * 100_000_000}
    upper = re.search(r"(\d+)억\s*이하", query)
    if upper:
        where["사업 금액"] = {"$lte": int(upper.group(1)) * 100_000_000}
    year = re.search(r"(202[0-9])년", query)
    if year:
        where["공개연도"] = int(year.group(1))
    return where or None


def extract_section_hint(query: str) -> str | None:
    """쿼리에서 섹션 키워드 힌트를 추출

    Args:
        query: 사용자 질의 문자열.

    Returns:
        매칭된 섹션 이름 또는 None.
    """
    for section, keywords in SECTION_KEYWORDS.items():
        if any(keyword in query for keyword in keywords):
            return section
    return None


def should_boost_tables(query: str) -> bool:
    """쿼리가 테이블 관련 키워드를 포함하는지 판별

    Args:
        query: 사용자 질의 문자열.

    Returns:
        테이블 부스팅 여부.
    """
    return any(keyword in query for keyword in TABLE_KEYWORDS)


def is_comparison_query(query: str) -> bool:
    """쿼리가 비교·대조·합산형 질문인지 판별

    Args:
        query: 사용자 질의 문자열.

    Returns:
        비교형 질의 여부.
    """
    if any(keyword in query for keyword in COMPARISON_KEYWORDS):
        return True
    return any(keyword in query for keyword in PER_SOURCE_KEYWORDS) and any(
        hint in query for hint in MULTI_SOURCE_HINT_KEYWORDS
    )
