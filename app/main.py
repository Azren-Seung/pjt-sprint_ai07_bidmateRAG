"""BidMate RAG — Streamlit UI with chat interface and sidebar controls."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from app.api.routes import (
    list_provider_configs,
    load_benchmark_frames,
    load_run_records,
    run_live_query,
)

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

    # ── 사이드바 ──
    with st.sidebar:
        st.header("⚙️ 설정")

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

        st.subheader("🔍 검색 설정")
        top_k = st.slider("Top-K (검색 청크 수)", min_value=1, max_value=20, value=5)

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

        st.divider()

        if st.button("🗑️ 대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pop("pending_example", None)
            st.rerun()

        st.caption("BidMate RAG v0.1 — 시나리오 B Baseline")

    # ── 탭 구성 ──
    demo_tab, eval_tab = st.tabs(["💬 라이브 데모", "📊 평가 비교"])

    # ── 탭 1: 채팅 UI ──
    with demo_tab:
        st.subheader("RFP 문서 질의응답")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 빈 상태: 예시 질문 칩
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
            # 사용자 메시지
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # 어시스턴트 응답
            with st.chat_message("assistant"):
                status = st.status("RAG 파이프라인 실행 중...", expanded=True)
                try:
                    status.write("🔍 관련 문서 검색 중...")
                    result = run_live_query(
                        question=prompt,
                        provider_config_path=selected_provider,
                        top_k=top_k,
                    )
                    status.write("✏️ 답변 생성 완료")
                    status.update(label="완료", state="complete", expanded=False)

                    # 답변
                    st.markdown(result.answer)

                    # 메타데이터
                    retrieved_records = []
                    if result.retrieved_chunks:
                        retrieved_records = [
                            {
                                "순위": c.rank,
                                "유사도": round(c.score, 4),
                                "사업명": c.chunk.metadata.get("사업명", "")[:25],
                                "발주기관": c.chunk.metadata.get("발주기관", ""),
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

                    # 성공 토스트
                    st.toast(
                        f"답변 완료 — {meta['tokens']:,}토큰, {meta['latency']}ms",
                        icon="✅",
                    )

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

    # ── 탭 2: 평가 비교 ──
    with eval_tab:
        st.subheader("벤치마크 결과 비교")
        benchmark_df = load_benchmark_frames()
        if benchmark_df.empty:
            st.info("아직 저장된 benchmark 결과가 없습니다.")
            st.markdown("""
**실행 방법:**
```bash
uv run python scripts/run_eval.py \\
    --evaluation-path data/eval/eval_set.json \\
    --provider-config configs/providers/openai_gpt5mini.yaml
```
            """)
        else:
            source_files = sorted(benchmark_df["source_file"].unique().tolist())
            selected_files = st.multiselect("요약 결과 파일", source_files, default=source_files)
            filtered = (
                benchmark_df[benchmark_df["source_file"].isin(selected_files)]
                if selected_files
                else benchmark_df
            )
            st.dataframe(filtered, use_container_width=True)

            run_files = sorted(Path("artifacts/logs/runs").glob("*.jsonl"))
            if run_files:
                selected_run = st.selectbox(
                    "질문 단위 run 기록", run_files, format_func=lambda path: path.name
                )
                records = load_run_records(selected_run)
                if records:
                    st.dataframe(records, use_container_width=True)


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
