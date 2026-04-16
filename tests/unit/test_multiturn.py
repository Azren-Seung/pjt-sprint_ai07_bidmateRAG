from unittest.mock import MagicMock

from bidmate_rag.retrieval.multiturn import (
    extract_recent_agency_filter,
    rewrite_query_with_history,
)


def _make_mock_llm(
    rewritten_text: str,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> MagicMock:
    from bidmate_rag.providers.llm.base import RewriteResponse

    mock_llm = MagicMock()
    mock_llm.rewrite.return_value = RewriteResponse(
        text=rewritten_text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    mock_llm.model_name = "gpt-5-mini"
    return mock_llm


def test_rewrite_query_with_history_replaces_follow_up_reference_with_recent_topic() -> None:
    rewritten, trace = rewrite_query_with_history(
        query="그 사업 예산은?",
        chat_history=[{"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"}],
        agency_list=["국민연금공단", "기초과학연구원"],
    )

    assert rewritten == "국민연금공단 차세대 ERP 사업 예산은?"
    assert trace["rewrite_reason"] == "rule_fallback"


def test_rewrite_query_with_history_uses_llm_for_implicit_followup() -> None:
    mock_llm = _make_mock_llm("국민연금공단 차세대 ERP 사업의 평가기준은?")

    rewritten, trace = rewrite_query_with_history(
        query="평가기준은?",
        chat_history=[
            {"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"},
            {"role": "assistant", "content": "해당 사업은 국민연금공단의 차세대 ERP 구축 사업입니다."},
        ],
        agency_list=["국민연금공단"],
        llm=mock_llm,
    )

    assert rewritten == "국민연금공단 차세대 ERP 사업의 평가기준은?"
    mock_llm.rewrite.assert_called_once()
    assert trace["rewrite_reason"] == "llm"
    assert trace["rewrite_applied"] is True


def test_rewrite_query_with_history_includes_slot_memory_in_llm_prompt() -> None:
    mock_llm = _make_mock_llm("국민연금공단 차세대 ERP 사업의 평가기준은?")

    rewritten, _trace = rewrite_query_with_history(
        query="평가기준은?",
        chat_history=[
            {"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"},
            {"role": "assistant", "content": "해당 사업은 국민연금공단의 차세대 ERP 구축 사업입니다."},
        ],
        agency_list=["국민연금공단"],
        llm=mock_llm,
        slot_memory={
            "발주기관": "국민연금공단",
            "사업명": "차세대 ERP 사업",
            "관심속성": "평가기준",
        },
    )

    prompt = mock_llm.rewrite.call_args.args[0]
    assert "발주기관: 국민연금공단" in prompt
    assert "사업명: 차세대 ERP 사업" in prompt
    assert "관심속성: 평가기준" in prompt
    assert rewritten == "국민연금공단 차세대 ERP 사업의 평가기준은?"


def test_rewrite_query_with_history_skips_llm_when_no_history() -> None:
    mock_llm = _make_mock_llm("무시되는 응답")

    rewritten, trace = rewrite_query_with_history(
        query="서버 구축 사업 찾아줘",
        chat_history=[],
        agency_list=[],
        llm=mock_llm,
    )

    assert rewritten == "서버 구축 사업 찾아줘"
    mock_llm.rewrite.assert_not_called()
    assert trace["rewrite_reason"] == "original"


def test_rewrite_query_with_history_falls_back_to_rule_based_on_llm_error() -> None:
    mock_llm = MagicMock()
    mock_llm.model_name = "gpt-5-mini"
    mock_llm.rewrite.side_effect = Exception("timeout")

    rewritten, trace = rewrite_query_with_history(
        query="그 사업 예산은?",
        chat_history=[{"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"}],
        agency_list=["국민연금공단"],
        llm=mock_llm,
    )

    assert rewritten == "국민연금공단 차세대 ERP 사업 예산은?"
    assert trace["rewrite_reason"] == "rule_fallback"
    assert trace["rewrite_error"] == "timeout"


def test_extract_recent_agency_filter_prefers_latest_single_agency_from_history() -> None:
    agency_filter = extract_recent_agency_filter(
        chat_history=[
            {"role": "user", "content": "기초과학연구원 사업 알려줘"},
            {"role": "assistant", "content": "설명"},
            {"role": "user", "content": "국민연금공단 차세대 ERP 사업 알려줘"},
        ],
        agency_list=["국민연금공단", "기초과학연구원"],
    )

    assert agency_filter == {"발주 기관": "국민연금공단"}
