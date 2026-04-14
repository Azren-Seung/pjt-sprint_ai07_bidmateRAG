"""Integration tests for web_api routes (TestClient based)."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from bidmate_rag.storage.metadata_store import MetadataStore
from bidmate_rag.web_api.routes import router


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "파일명": "doc-1.hwp",
                "사업명": "학사 시스템 고도화",
                "발주 기관": "한영대학",
                "기관유형": "대학교",
                "사업도메인": "교육/학습",
                "사업 금액": 130000000.0,
                "입찰 참여 마감일": "2024-12-15",
                "정제_글자수": 45230,
                "사업 요약": "학사 시스템 고도화 사업",
            },
            {
                "파일명": "doc-2.hwp",
                "사업명": "이러닝 시스템",
                "발주 기관": "국민연금공단",
                "기관유형": "공기업/준정부기관",
                "사업도메인": "교육/학습",
                "사업 금액": 1230000000.0,
                "입찰 참여 마감일": "2024-11-30",
                "정제_글자수": 60000,
                "사업 요약": "",
            },
        ]
    )


@pytest.fixture
def client(sample_frame):
    @asynccontextmanager
    async def test_lifespan(app: FastAPI):
        app.state.metadata_store = MetadataStore(sample_frame)
        yield

    test_app = FastAPI(title="BidMate Web API (Test)", version="0.1.0", lifespan=test_lifespan)
    test_app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    test_app.include_router(router, prefix="/api")

    with TestClient(test_app) as test_client:
        yield test_client


def test_get_documents_returns_list(client) -> None:
    response = client.get("/api/documents")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["documents"]) == 2
    first = data["documents"][0]
    assert first["id"] == "doc-1.hwp"
    assert first["agency"] == "한영대학"
    assert first["budget_label"].endswith("억")


def test_get_document_detail(client) -> None:
    response = client.get("/api/documents/doc-1.hwp")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "doc-1.hwp"
    assert len(data["quick_facts"]) == 5
    labels = [f["label"] for f in data["quick_facts"]]
    assert "발주기관" in labels
    assert "사업금액" in labels


def test_get_document_detail_not_found(client) -> None:
    response = client.get("/api/documents/nonexistent.hwp")
    assert response.status_code == 404


def test_get_commands_returns_twelve(client) -> None:
    response = client.get("/api/commands")
    assert response.status_code == 200
    data = response.json()
    assert len(data["commands"]) == 12
    ids = {c["id"] for c in data["commands"]}
    assert "비교" in ids
    compare = next(c for c in data["commands"] if c["id"] == "비교")
    assert compare["requires_multi_doc"] is True
