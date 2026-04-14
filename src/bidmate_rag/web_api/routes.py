"""FastAPI routes for BidMate web_api."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from bidmate_rag.web_api.commands import COMMAND_REGISTRY
from bidmate_rag.web_api.schemas import (
    DocumentDetail,
    DocumentSummary,
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
        id=str(row.get("파일명", "")),
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
    match = frame[frame["파일명"] == doc_id]
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
