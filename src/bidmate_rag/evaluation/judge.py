"""LLM-as-judge for RAG generation quality.

Computes 4 RAGAS-style metrics in a single LLM call per sample:
- faithfulness:      답변이 검색된 context에만 근거하는가? (hallucination 검출)
- answer_relevance:  답변이 질문에 직접적으로 답하는가?
- context_precision: 검색된 context가 답변에 실제로 사용된 비율
- context_recall:    expected_answer 정보가 검색된 context에 포함된 비율
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any

from openai import OpenAI

from bidmate_rag.tracking.pricing import calc_llm_cost, load_pricing

logger = logging.getLogger(__name__)


JUDGE_SYSTEM_PROMPT = """당신은 RAG 시스템의 품질을 평가하는 전문 심사자입니다.
주어진 질문, 검색된 context, 생성된 답변을 보고 4가지 항목을 0.0~1.0 사이의 점수로 평가하세요.

평가 기준:
1. faithfulness: 답변의 모든 주장이 context에 근거하는가? (1.0 = 완전 근거, 0.0 = 환각)
2. answer_relevance: 답변이 질문에 직접 답하는가? (1.0 = 정확히 답함, 0.0 = 무관)
3. context_precision: 검색된 context 중 답변 생성에 실제 사용된 비율 (1.0 = 모두 활용)
4. context_recall: expected_answer가 주어졌다면, 그 정보가 context에 포함된 비율

반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트 금지:
{"faithfulness": 0.0, "answer_relevance": 0.0, "context_precision": 0.0, "context_recall": 0.0}
"""


@dataclass
class JudgeScores:
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LLMJudge:
    """Single-model judge that scores 4 RAG quality metrics in one call."""

    METRIC_KEYS = ("faithfulness", "answer_relevance", "context_precision", "context_recall")

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_base: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        client: OpenAI | None = None,
    ) -> None:
        self.model = model
        self.client = client or OpenAI(
            api_key=os.getenv(api_key_env, "EMPTY"),
            base_url=api_base,
        )
        self.pricing = load_pricing()
        self.cumulative_cost_usd: float = 0.0
        self.cumulative_tokens: int = 0

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        expected_answer: str | None = None,
    ) -> JudgeScores:
        """Score one (question, answer, contexts) sample. Never raises."""
        user_prompt = self._build_user_prompt(question, answer, contexts, expected_answer)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=500,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Judge LLM call failed: %s", exc)
            return JudgeScores(error=f"llm_call_failed: {exc}")

        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        cached_tokens = int(getattr(prompt_details, "cached_tokens", 0) or 0)
        self.cumulative_tokens += prompt_tokens + completion_tokens
        self.cumulative_cost_usd += calc_llm_cost(
            self.model,
            prompt_tokens,
            completion_tokens,
            self.pricing,
            cached_tokens=cached_tokens,
        )

        raw_text = response.choices[0].message.content or ""
        return self._parse_scores(raw_text)

    def _build_user_prompt(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        expected_answer: str | None,
    ) -> str:
        context_block = "\n\n---\n\n".join(
            f"[Context {i + 1}]\n{ctx}" for i, ctx in enumerate(contexts)
        ) or "(검색된 context 없음)"
        expected_block = (
            f"\n\n[Expected Answer]\n{expected_answer}" if expected_answer else ""
        )
        return (
            f"[Question]\n{question}\n\n"
            f"[Generated Answer]\n{answer}\n\n"
            f"[Retrieved Contexts]\n{context_block}"
            f"{expected_block}"
        )

    def _parse_scores(self, raw_text: str) -> JudgeScores:
        try:
            data = json.loads(raw_text)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Judge response not valid JSON: %s", exc)
            return JudgeScores(error=f"json_parse_failed: {raw_text[:100]}")

        scores = JudgeScores()
        for key in self.METRIC_KEYS:
            value = data.get(key, 0.0)
            try:
                setattr(scores, key, max(0.0, min(1.0, float(value))))
            except (TypeError, ValueError):
                setattr(scores, key, 0.0)
        return scores
