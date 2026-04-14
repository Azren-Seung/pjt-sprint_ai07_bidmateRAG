"""FastAPI routes for BidMate web_api."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from app.api.routes import run_live_query
from bidmate_rag.schema import GenerationResult, RetrievedChunk
from bidmate_rag.web_api.commands import COMMAND_REGISTRY, SlashCommand
from bidmate_rag.web_api.retrieval_helpers import per_doc_split_query
from bidmate_rag.web_api.schemas import (
    Citation,
    DocumentDetail,
    DocumentSummary,
    QueryMetadata,
    QueryRequest,
    QueryResponse,
    SlashCommandMeta,
)

router = APIRouter()


def _format_budget_label(amount: float) -> str:
    """숫자 금액을 한국어 라벨로."""
    if not amount or (isinstance(amount, float) and math.isnan(amount)):
        return "-"
    if amount >= 100_000_000:
        return f"{amount / 100_000_000:.1f}억"
    if amount >= 10_000:
        return f"{amount / 10_000:.0f}만"
    return f"{int(amount)}"


def _row_to_summary(row: pd.Series) -> DocumentSummary:
    budget = float(row.get("사업 금액") or 0)
    return DocumentSummary(
        id=str(row.get("공고 번호", "")),
        title=str(row.get("사업명", "")),
        agency=str(row.get("발주 기관", "")),
        agency_type=str(row.get("기관유형", "")),
        domain=str(row.get("사업도메인", "")),
        budget=budget,
        budget_label=_format_budget_label(budget),
        deadline=str(row.get("입찰 참여 마감일", "")) or None,
        char_count=int(row.get("정제_글자수") or 0),
    )


@router.get("/documents")
def list_documents(request: Request) -> dict[str, Any]:
    frame: pd.DataFrame = request.app.state.metadata_store.frame
    if frame.empty:
        return {"documents": [], "total": 0}
    documents = [_row_to_summary(row).model_dump() for _, row in frame.iterrows()]
    return {"documents": documents, "total": len(documents)}


@router.get("/documents/{doc_id}")
def get_document(doc_id: str, request: Request) -> dict[str, Any]:
    frame: pd.DataFrame = request.app.state.metadata_store.frame
    match = frame[frame["공고 번호"].astype(str) == doc_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"document not found: {doc_id}")
    row = match.iloc[0]
    summary = _row_to_summary(row)
    summary_oneline = str(row.get("사업 요약") or "") or None
    quick_facts = [
        {"label": "발주기관", "value": summary.agency or "-"},
        {"label": "사업금액", "value": summary.budget_label},
        {"label": "마감일", "value": summary.deadline or "-"},
        {"label": "도메인", "value": summary.domain or "-"},
        {"label": "문서크기", "value": f"{summary.char_count:,}자"},
    ]
    detail = DocumentDetail(
        **summary.model_dump(),
        summary_oneline=summary_oneline,
        quick_facts=quick_facts,
    )
    return detail.model_dump()


@router.get("/commands")
def list_commands() -> dict[str, Any]:
    commands = [
        SlashCommandMeta(
            id=cmd.id,
            label=cmd.label,
            description=cmd.description,
            icon=cmd.icon,
            requires_doc=cmd.requires_doc,
            requires_multi_doc=cmd.requires_multi_doc,
        ).model_dump()
        for cmd in COMMAND_REGISTRY.values()
    ]
    return {"commands": commands}


def _build_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    citations: list[Citation] = []
    for idx, rc in enumerate(chunks, start=1):
        chunk = rc.chunk
        citations.append(
            Citation(
                id=idx,
                doc_id=chunk.doc_id,
                doc_title=str(chunk.metadata.get("사업명", "")),
                section=chunk.section or "",
                content_type=chunk.content_type or "text",
                text=chunk.text[:500],
                score=float(rc.score),
            )
        )
    return citations


def _validate_command(cmd: SlashCommand, mentioned_doc_ids: list[str]) -> None:
    if cmd.requires_doc and not mentioned_doc_ids:
        raise HTTPException(status_code=400, detail=f"{cmd.label}는 문서 멘션이 필요합니다")
    if cmd.requires_multi_doc and len(mentioned_doc_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"{cmd.label}는 2개 이상 문서 멘션이 필요합니다",
        )


def _static_response(cmd: SlashCommand) -> QueryResponse:
    payload = cmd.static_payload or {}
    return QueryResponse(
        answer=payload.get("answer", ""),
        citations=[],
        metadata=QueryMetadata(
            model="-",
            token_usage={},
            latency_ms=0.0,
            cost_usd=0.0,
            command_applied=cmd.id,
            filter_applied=None,
            retrieval_strategy="static",
            per_doc_k=None,
        ),
    )


def _result_to_response(
    result: GenerationResult,
    cmd: SlashCommand | None,
    filter_applied: dict | None,
    strategy: str,
    per_doc_k: int | None,
) -> QueryResponse:
    return QueryResponse(
        answer=result.answer,
        citations=_build_citations(result.retrieved_chunks),
        metadata=QueryMetadata(
            model=result.llm_model,
            token_usage=result.token_usage,
            latency_ms=result.latency_ms,
            cost_usd=result.cost_usd,
            command_applied=cmd.id if cmd else None,
            filter_applied=filter_applied,
            retrieval_strategy=strategy,
            per_doc_k=per_doc_k,
        ),
    )


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    cmd = COMMAND_REGISTRY.get(req.command) if req.command else None

    # 1. 알 수 없는 커맨드
    if req.command and cmd is None:
        raise HTTPException(status_code=400, detail=f"unknown command: {req.command}")

    # 2. validation
    if cmd:
        _validate_command(cmd, req.mentioned_doc_ids)

    # 3. static response (/도움말, /초기화)
    if cmd and cmd.static_response:
        return _static_response(cmd)

    # 4. 쿼리 증강
    augmented_query = req.question
    if cmd and cmd.query_augmentation:
        augmented_query = f"{req.question} {cmd.query_augmentation}".strip()

    # 5. 시스템 프롬프트
    system_prompt = cmd.system_prompt if cmd and cmd.system_prompt else None

    # 6. top_k
    top_k = cmd.top_k if cmd else req.top_k

    # 7. 분기: 멘션 2개+ → per-doc split, 아니면 run_live_query
    if len(req.mentioned_doc_ids) >= 2:
        result = per_doc_split_query(
            question=req.question,
            augmented_query=augmented_query,
            mentioned_doc_ids=req.mentioned_doc_ids,
            provider_config=req.provider_config,
            chunking_config=req.chunking_config,
            system_prompt=system_prompt,
            top_k=top_k,
            max_context_chars=req.max_context_chars,
        )
        return _result_to_response(
            result,
            cmd,
            filter_applied={"doc_id": {"$in": req.mentioned_doc_ids}},
            strategy="per_doc_split",
            per_doc_k=max(top_k // len(req.mentioned_doc_ids), 3) + 2,
        )

    # 단일/0 문서 경로
    manual_filters: dict | None = None
    if len(req.mentioned_doc_ids) == 1:
        manual_filters = {"doc_id": req.mentioned_doc_ids[0]}

    result = run_live_query(
        question=augmented_query,
        provider_config_path=f"configs/providers/{req.provider_config}.yaml",
        experiment_config_path=f"configs/chunking/{req.chunking_config}.yaml",
        top_k=top_k,
        manual_filters=manual_filters,
        system_prompt=system_prompt,
        max_context_chars=req.max_context_chars,
    )
    return _result_to_response(
        result,
        cmd,
        filter_applied=manual_filters,
        strategy="single",
        per_doc_k=None,
    )
