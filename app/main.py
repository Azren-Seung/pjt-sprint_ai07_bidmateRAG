"""BidMate RAG — Streamlit UI with chat interface and sidebar controls."""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from app.api.routes import (
    list_provider_configs,
    load_benchmark_frames,
    load_metadata_options,
    load_run_records,
    run_live_query,
)
from app.eval_ui import render_eval_tabs

EXAMPLE_QUESTIONS = [
    "국민연금공단이 발주한 이러닝시스템 관련 사업 요구사항을 정리해 줘",
    "한국원자력연구원 선량평가시스템 고도화 사업의 목적을 알려줘",
    "고려대학교 차세대 포털이랑 광주과학기술원 학사 시스템을 비교해줘",
    "교육 관련 사업 찾아줘",
    "5억 이상 대규모 시스템 구축 사업이 있어?",
    "기초과학연구원 극저온시스템에서 AI 기반 예측 요구사항이 있나?",
]


def _running_under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


def _render_streamlit_app() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="BidMate RAG",
        page_icon="📄",
        layout="wide",
    )

    # 세션 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_stats" not in st.session_state:
        st.session_state.session_stats = {"queries": 0, "total_tokens": 0, "total_latency": 0}

    # 메타데이터 옵션 로딩 (캐싱)
    meta_options = load_metadata_options()

    # ── 사이드바 ──
    with st.sidebar:
        st.header("⚙️ 설정")

        # 1) Provider 선택
        provider_configs = list_provider_configs()
        if not provider_configs:
            st.warning("Provider config가 없습니다.")
            st.stop()

        selected_provider = st.selectbox(
            "LLM Provider",
            provider_configs,
            format_func=lambda p: p.stem,
            help="configs/providers/ 안의 YAML 파일",
        )

        # 2) 검색 설정
        st.subheader("🔍 검색 설정")
        top_k = st.slider("Top-K (검색 청크 수)", min_value=1, max_value=20, value=5)
        search_mode = st.radio(
            "검색 모드",
            ["자동 필터", "수동 필터", "필터 없음"],
            help="자동: 질문에서 필터 추출 | 수동: 아래 설정 사용 | 필터 없음: 벡터 검색만",
        )

        # 3) 메타데이터 필터 수동 설정
        manual_filters = {}
        if search_mode == "수동 필터":
            st.subheader("🏷️ 메타데이터 필터")

            # 발주기관
            if meta_options["agencies"]:
                selected_agency = st.selectbox(
                    "발주기관",
                    ["전체"] + meta_options["agencies"],
                )
                if selected_agency != "전체":
                    manual_filters["발주기관"] = selected_agency

            # 사업도메인
            if meta_options["domains"]:
                selected_domain = st.selectbox(
                    "사업도메인",
                    ["전체"] + meta_options["domains"],
                )
                if selected_domain != "전체":
                    manual_filters["사업도메인"] = selected_domain

            # 기관유형
            if meta_options["agency_types"]:
                selected_type = st.selectbox(
                    "기관유형",
                    ["전체"] + meta_options["agency_types"],
                )
                if selected_type != "전체":
                    manual_filters["기관유형"] = selected_type

            # 사업금액 범위
            budget_filter = st.selectbox(
                "사업금액",
                ["전체", "1억 이하", "1~5억", "5~10억", "10억 이상"],
            )
            if budget_filter == "1억 이하":
                manual_filters["사업금액"] = {"$lte": 100_000_000}
            elif budget_filter == "1~5억":
                manual_filters["사업금액"] = {"$gte": 100_000_000, "$lte": 500_000_000}
            elif budget_filter == "5~10억":
                manual_filters["사업금액"] = {"$gte": 500_000_000, "$lte": 1_000_000_000}
            elif budget_filter == "10억 이상":
                manual_filters["사업금액"] = {"$gte": 1_000_000_000}

            if manual_filters:
                st.caption(f"적용 필터: {manual_filters}")

        elif search_mode == "필터 없음":
            manual_filters = {"_no_filter": True}  # 필터 해제 신호

        # 4) 모델 정보
        st.subheader("📊 모델 정보")
        try:
            import yaml
            config = yaml.safe_load(selected_provider.read_text())
            st.markdown(f"""
- **Provider**: `{config.get('provider', '-')}`
- **Model**: `{config.get('model', '-')}`
- **Embedding**: `{config.get('embedding_model', '-')}`
- **Scenario**: `{config.get('scenario', '-')}`
            """)
        except Exception:
            pass

        # 5) 응답 통계
        stats = st.session_state.session_stats
        if stats["queries"] > 0:
            st.subheader("📈 세션 통계")
            cols = st.columns(2)
            cols[0].metric("질문 수", f"{stats['queries']}회")
            cols[1].metric("총 토큰", f"{stats['total_tokens']:,}")
            cols = st.columns(2)
            avg_latency = stats["total_latency"] / stats["queries"]
            cols[0].metric("평균 응답", f"{avg_latency:.0f}ms")
            est_cost = stats["total_tokens"] * 0.15 / 1_000_000
            cols[1].metric("예상 비용", f"${est_cost:.4f}")

        st.divider()

        # 대화 히스토리 내보내기
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ 초기화", use_container_width=True):
                st.session_state.messages = []
                st.session_state.session_stats = {"queries": 0, "total_tokens": 0, "total_latency": 0}
                st.session_state.pop("pending_example", None)
                st.rerun()
        with col2:
            if st.session_state.messages and st.button("💾 내보내기", use_container_width=True):
                export = []
                for msg in st.session_state.messages:
                    entry = {"role": msg["role"], "content": msg["content"]}
                    if msg.get("metadata"):
                        entry["metadata"] = {
                            k: v for k, v in msg["metadata"].items() if k != "retrieved"
                        }
                    export.append(entry)
                st.download_button(
                    "⬇️ JSON 다운로드",
                    data=json.dumps(export, ensure_ascii=False, indent=2),
                    file_name="bidmate_rag_chat.json",
                    mime="application/json",
                    use_container_width=True,
                )

        st.caption("BidMate RAG v0.1 — 시나리오 B Baseline")

    # ── 탭 구성 ──
    demo_tab, docs_tab, eval_tab = st.tabs(["💬 라이브 데모", "📁 문서 목록", "📊 평가"])

    # ── 탭 1: 채팅 UI ──
    with demo_tab:
        st.subheader("RFP 문서 질의응답")

        # 빈 상태: 예시 질문
        if not st.session_state.messages:
            st.caption("아래 예시를 클릭하거나 직접 질문을 입력하세요.")
            cols = st.columns(3)
            for i, q in enumerate(EXAMPLE_QUESTIONS):
                if cols[i % 3].button(
                    q[:30] + "..." if len(q) > 30 else q,
                    key=f"example_{i}",
                    use_container_width=True,
                ):
                    st.session_state["pending_example"] = q
                    st.rerun()

        # 기존 메시지 렌더링
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("metadata"):
                    _render_metadata_expander(st, msg["metadata"])

        # 예시 질문 처리
        pending = st.session_state.pop("pending_example", None)
        prompt = pending or st.chat_input("질문을 입력하세요...")

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                status = st.status("RAG 파이프라인 실행 중...", expanded=True)
                try:
                    status.write("🔍 관련 문서 검색 중...")

                    # 수동 필터 전달
                    filters_to_pass = manual_filters if manual_filters else None

                    result = run_live_query(
                        question=prompt,
                        provider_config_path=selected_provider,
                        top_k=top_k,
                        manual_filters=filters_to_pass,
                    )
                    status.write("✏️ 답변 생성 완료")
                    status.update(label="완료", state="complete", expanded=False)

                    st.markdown(result.answer)

                    retrieved_records = []
                    if result.retrieved_chunks:
                        retrieved_records = [
                            {
                                "순위": c.rank,
                                "유사도": round(c.score, 4),
                                "사업명": c.chunk.metadata.get("사업명", "")[:25],
                                "발주기관": c.chunk.metadata.get("발주기관", ""),
                                "도메인": c.chunk.metadata.get("사업도메인", ""),
                                "유형": c.chunk.content_type,
                                "내용": c.chunk.text[:100],
                            }
                            for c in result.retrieved_chunks
                        ]

                    meta = {
                        "chunks": len(result.retrieved_chunks),
                        "context_chars": len(result.context) if result.context else 0,
                        "tokens": result.token_usage.get("total", 0),
                        "latency": round(result.latency_ms),
                        "retrieved": retrieved_records,
                    }

                    _render_metadata_expander(st, meta)

                    st.toast(
                        f"답변 완료 — {meta['tokens']:,}토큰, {meta['latency']}ms",
                        icon="✅",
                    )

                    # 세션 통계 업데이트
                    st.session_state.session_stats["queries"] += 1
                    st.session_state.session_stats["total_tokens"] += meta["tokens"]
                    st.session_state.session_stats["total_latency"] += meta["latency"]

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result.answer,
                        "metadata": meta,
                    })

                except Exception as exc:
                    status.update(label="오류 발생", state="error", expanded=False)
                    error_msg = str(exc)
                    st.error(f"오류가 발생했습니다: {error_msg}")
                    st.info("💡 해결 방법:\n"
                            "- `.env` 파일에 OPENAI_API_KEY가 설정되어 있는지 확인\n"
                            "- 사이드바에서 올바른 Provider를 선택했는지 확인\n"
                            "- ChromaDB 인덱스가 생성되어 있는지 확인 (`05_embedding` 실행)")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"오류: {error_msg}",
                    })

    # ── 탭 2: 문서 목록 ──
    with docs_tab:
        st.subheader("RFP 문서 목록")
        chunks_path = Path("data/processed/cleaned_documents.parquet")
        if not chunks_path.exists():
            st.info("문서 데이터가 없습니다. `02_preprocessing` → `03_cleaning`을 실행해주세요.")
        else:
            import pandas as pd
            docs_df = pd.read_parquet(chunks_path)

            # 검색 필터
            search_term = st.text_input("문서 검색 (사업명, 발주기관)", placeholder="키워드 입력...")
            if search_term:
                mask = (
                    docs_df["사업명"].str.contains(search_term, case=False, na=False) |
                    docs_df["발주 기관"].str.contains(search_term, case=False, na=False)
                )
                docs_df = docs_df[mask]

            st.caption(f"총 {len(docs_df)}건")

            display_cols = ["사업명", "발주 기관", "사업 금액", "공개 일자", "파일형식", "정제_글자수"]
            available_cols = [c for c in display_cols if c in docs_df.columns]
            display = docs_df[available_cols].copy()
            if "사업 금액" in display.columns:
                display["사업 금액"] = display["사업 금액"].apply(
                    lambda x: f"{x/1e8:.1f}억" if x and x > 0 else "-"
                )
            st.dataframe(display, use_container_width=True, height=500)

            # 문서 상세 보기
            if len(docs_df) > 0:
                selected_doc = st.selectbox(
                    "문서 상세 보기",
                    docs_df["사업명"].tolist(),
                )
                if selected_doc:
                    doc = docs_df[docs_df["사업명"] == selected_doc].iloc[0]
                    with st.expander(f"📄 {selected_doc}", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        col1.metric("발주기관", doc.get("발주 기관", "-"))
                        budget = doc.get("사업 금액", 0)
                        col2.metric("사업금액", f"{budget/1e8:.1f}억" if budget and budget > 0 else "-")
                        col3.metric("본문 글자수", f"{doc.get('정제_글자수', 0):,}자")

                        if "사업 요약" in doc.index:
                            st.markdown(f"**사업 요약**: {doc['사업 요약']}")

                        if "본문_정제" in doc.index:
                            st.text_area("본문 미리보기 (앞 2000자)", doc["본문_정제"][:2000], height=300)

    # ── 탭 3: 평가 ──
    with eval_tab:
        render_eval_tabs(st, run_live_query, list_provider_configs, load_benchmark_frames, load_run_records)


def _render_metadata_expander(st_module, meta: dict) -> None:
    """검색 상세 메타데이터를 expander로 렌더링한다."""
    with st_module.expander("📋 검색 상세", expanded=False):
        cols = st_module.columns(4)
        cols[0].metric("검색 청크", f"{meta.get('chunks', '-')}개")
        cols[1].metric("컨텍스트", f"{meta.get('context_chars', 0):,}자")
        cols[2].metric("토큰", f"{meta.get('tokens', 0):,}")
        cols[3].metric("응답 시간", f"{meta.get('latency', '-')}ms")
        if meta.get("retrieved"):
            st_module.dataframe(meta["retrieved"], use_container_width=True)


def main() -> None:
    if not _running_under_streamlit():
        print("BidMate RAG scaffold is ready.")
        return
    _render_streamlit_app()


if __name__ == "__main__":
    main()
