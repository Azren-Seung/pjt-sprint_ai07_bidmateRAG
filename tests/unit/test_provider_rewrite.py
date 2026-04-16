"""LLM 프로바이더의 rewrite 인터페이스 단위 테스트."""

from unittest.mock import MagicMock

import pytest

from bidmate_rag.providers.llm.base import BaseLLMProvider, RewriteResponse
from bidmate_rag.providers.llm.openai_compat import OpenAICompatibleLLM


def _make_openai_client(response_text: str, prompt_tokens: int = 50,
                        completion_tokens: int = 10) -> MagicMock:
    client = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = response_text
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    response.usage.total_tokens = prompt_tokens + completion_tokens
    client.chat.completions.create.return_value = response
    return client


def test_base_provider_rewrite_raises_by_default() -> None:
    """기본 구현은 NotImplementedError — 서브클래스가 override해야 한다."""

    class BareProvider(BaseLLMProvider):
        provider_name = "bare"
        model_name = "bare-model"

        def generate(self, *args, **kwargs):
            raise NotImplementedError

    with pytest.raises(NotImplementedError):
        BareProvider().rewrite("test prompt")


def test_openai_provider_rewrite_returns_structured_response() -> None:
    client = _make_openai_client(
        response_text="국민연금공단 차세대 ERP 사업의 예산은 얼마인가요?",
        prompt_tokens=250,
        completion_tokens=40,
    )
    provider = OpenAICompatibleLLM(
        provider_name="openai", model_name="gpt-5-mini", client=client
    )

    response = provider.rewrite("재작성 프롬프트", max_tokens=16000, timeout=30)

    assert isinstance(response, RewriteResponse)
    assert response.text == "국민연금공단 차세대 ERP 사업의 예산은 얼마인가요?"
    assert response.prompt_tokens == 250
    assert response.completion_tokens == 40
    assert response.total_tokens == 290

    create_kwargs = client.chat.completions.create.call_args.kwargs
    assert create_kwargs["model"] == "gpt-5-mini"
    assert create_kwargs["max_completion_tokens"] == 16000
    assert create_kwargs["timeout"] == 30
    assert create_kwargs["messages"] == [
        {"role": "user", "content": "재작성 프롬프트"}
    ]


def test_openai_provider_rewrite_handles_empty_content() -> None:
    client = _make_openai_client(response_text="")
    provider = OpenAICompatibleLLM(
        provider_name="openai", model_name="gpt-5-mini", client=client
    )

    response = provider.rewrite("test", max_tokens=1000)
    assert response.text == ""
