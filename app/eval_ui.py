"""평가 탭 — 4개 서브탭 (실행, 디버깅, 비교, 편집)."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pandas as pd

EVAL_DIR = Path("data/eval")
EVAL_SET_PATH = EVAL_DIR / "eval_set.json"
RUNS_DIR = Path("artifacts/logs/runs")
BENCHMARKS_DIR = Path("artifacts/logs/benchmarks")


def _parse_json_field(value) -> object:
    """CSV에서 읽은 JSON 문자열 필드를 파싱한다."""
    if pd.isna(value) or value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    s = str(value).strip()
    if not s or s == "[]" or s == "{}":
        return None
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return s


def load_eval_set_from_csv(path: Path) -> list[dict]:
    """CSV 파일에서 평가셋을 로딩한다."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    records = []
    for _, row in df.iterrows():
        q = {
            "id": str(row.get("id", "")),
            "type": str(row.get("type", "A")),
            "difficulty": str(row.get("difficulty", "중")),
            "question": str(row.get("question", "")),
            "ground_truth_answer": str(row.get("ground_truth_answer", "")),
            "ground_truth_docs": _parse_json_field(row.get("ground_truth_docs")) or [],
            "metadata_filter": _parse_json_field(row.get("metadata_filter")),
            "history": _parse_json_field(row.get("history")),
        }
        records.append(q)
    return records


def load_eval_set() -> list[dict]:
    """평가셋을 로딩한다. CSV가 있으면 CSV 우선, 없으면 JSON."""
    csv_files = sorted(EVAL_DIR.glob("eval_batch_*.csv"))
    if csv_files:
        # 가장 최신 CSV 로딩
        all_records = []
        for csv_path in csv_files:
            all_records.extend(load_eval_set_from_csv(csv_path))
        return all_records
    if EVAL_SET_PATH.exists():
        return json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
    return []


def save_eval_set(data: list[dict], fmt: str = "csv") -> Path:
    """평가셋을 저장한다."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        save_path = EVAL_DIR / "eval_set_edited.csv"
        rows = []
        for q in data:
            rows.append({
                "id": q["id"],
                "type": q["type"],
                "difficulty": q.get("difficulty", "중"),
                "question": q["question"],
                "ground_truth_answer": q.get("ground_truth_answer", ""),
                "ground_truth_docs": json.dumps(q.get("ground_truth_docs", []), ensure_ascii=False),
                "metadata_filter": json.dumps(q.get("metadata_filter") or {}, ensure_ascii=False),
                "history": json.dumps(q.get("history") or [], ensure_ascii=False),
            })
        pd.DataFrame(rows).to_csv(save_path, index=False, encoding="utf-8-sig")
        return save_path
    else:
        EVAL_SET_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return EVAL_SET_PATH


def render_eval_tabs(st, run_live_query, list_provider_configs, load_benchmark_frames, load_run_records):
    """평가 탭 메인 렌더러."""

    run_tab, debug_tab, compare_tab, edit_tab = st.tabs(
        ["🏃 평가 실행", "🔍 질문 디버깅", "⚖️ 결과 비교", "✏️ 평가셋 편집"]
    )

    # 평가셋 session 관리
    if "eval_set" not in st.session_state:
        st.session_state.eval_set = load_eval_set()
    if "eval_results" not in st.session_state:
        st.session_state.eval_results = {}

    eval_set = st.session_state.eval_set

    # ── 서브탭 1: 평가 실행 ──
    with run_tab:
        _render_run_tab(st, eval_set, run_live_query, list_provider_configs)

    # ── 서브탭 2: 질문 디버깅 ──
    with debug_tab:
        _render_debug_tab(st, eval_set, run_live_query, list_provider_configs)

    # ── 서브탭 3: 결과 비교 ──
    with compare_tab:
        _render_compare_tab(st, load_benchmark_frames, load_run_records)

    # ── 서브탭 4: 평가셋 편집 ──
    with edit_tab:
        _render_edit_tab(st, eval_set)


def _render_run_tab(st, eval_set, run_live_query, list_provider_configs):
    st.subheader("평가셋 일괄 실행")

    if not eval_set:
        st.warning("평가셋이 비어있습니다. '평가셋 편집' 탭에서 질문을 추가하세요.")
        return

    provider_configs = list_provider_configs()
    col1, col2, col3 = st.columns(3)
    with col1:
        provider = st.selectbox("Provider", provider_configs, format_func=lambda p: p.stem, key="run_provider")
    with col2:
        top_k = st.slider("Top-K", 1, 20, 5, key="run_topk")
    with col3:
        run_scope = st.selectbox("실행 범위", ["전체", "유형별", "난이도별"], key="run_scope")

    # 필터
    filtered = eval_set
    if run_scope == "유형별":
        types = sorted(set(q["type"] for q in eval_set))
        selected_type = st.multiselect("유형 선택", types, default=types, key="run_types")
        filtered = [q for q in eval_set if q["type"] in selected_type]
    elif run_scope == "난이도별":
        diffs = sorted(set(q.get("difficulty", "중") for q in eval_set))
        selected_diff = st.multiselect("난이도 선택", diffs, default=diffs, key="run_diffs")
        filtered = [q for q in eval_set if q.get("difficulty", "중") in selected_diff]

    st.caption(f"실행 대상: {len(filtered)}개 질문")

    if st.button("▶️ 평가 실행", type="primary", key="run_eval_btn"):
        run_id = f"eval-{uuid4().hex[:8]}"
        results = []
        progress = st.progress(0, text="평가 실행 중...")

        for i, q in enumerate(filtered):
            progress.progress((i + 1) / len(filtered), text=f"[{i+1}/{len(filtered)}] {q['question'][:40]}...")
            try:
                result = run_live_query(
                    question=q["question"],
                    provider_config_path=provider,
                    top_k=top_k,
                )
                results.append({
                    "id": q["id"],
                    "type": q["type"],
                    "difficulty": q.get("difficulty", "중"),
                    "question": q["question"][:40],
                    "answer_preview": result.answer[:100] if result.answer else "",
                    "chunks": len(result.retrieved_chunks),
                    "tokens": result.token_usage.get("total", 0),
                    "latency_ms": round(result.latency_ms),
                    "ground_truth": q.get("ground_truth_answer", "")[:50],
                })
            except Exception as e:
                results.append({
                    "id": q["id"], "type": q["type"],
                    "difficulty": q.get("difficulty", "중"),
                    "question": q["question"][:40],
                    "answer_preview": f"오류: {str(e)[:60]}",
                    "chunks": 0, "tokens": 0, "latency_ms": 0,
                    "ground_truth": "",
                })

        progress.empty()
        st.session_state.eval_results[run_id] = results

        # 요약
        df = pd.DataFrame(results)
        st.success(f"실행 완료: {len(results)}건 (run_id: {run_id})")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**유형별 요약**")
            if len(df) > 0:
                type_summary = df.groupby("type").agg(
                    질문수=("id", "count"),
                    평균토큰=("tokens", "mean"),
                    평균응답ms=("latency_ms", "mean"),
                ).round(0)
                st.dataframe(type_summary, use_container_width=True)
        with col2:
            st.markdown("**난이도별 요약**")
            if len(df) > 0:
                diff_summary = df.groupby("difficulty").agg(
                    질문수=("id", "count"),
                    평균토큰=("tokens", "mean"),
                    평균응답ms=("latency_ms", "mean"),
                ).round(0)
                st.dataframe(diff_summary, use_container_width=True)

        st.dataframe(df, use_container_width=True)


def _render_debug_tab(st, eval_set, run_live_query, list_provider_configs):
    st.subheader("질문별 디버깅")

    if not eval_set:
        st.warning("평가셋이 비어있습니다.")
        return

    # 필터
    col1, col2 = st.columns(2)
    with col1:
        type_filter = st.selectbox("유형 필터", ["전체"] + sorted(set(q["type"] for q in eval_set)), key="debug_type")
    with col2:
        diff_filter = st.selectbox("난이도 필터", ["전체"] + sorted(set(q.get("difficulty", "중") for q in eval_set)), key="debug_diff")

    filtered = eval_set
    if type_filter != "전체":
        filtered = [q for q in filtered if q["type"] == type_filter]
    if diff_filter != "전체":
        filtered = [q for q in filtered if q.get("difficulty", "중") == diff_filter]

    if not filtered:
        st.info("조건에 맞는 질문이 없습니다.")
        return

    selected_q = st.selectbox(
        "질문 선택",
        filtered,
        format_func=lambda q: f"[{q['id']}|{q['type']}|{q.get('difficulty','중')}] {q['question'][:50]}",
        key="debug_question",
    )

    if not selected_q:
        return

    # 설정
    provider_configs = list_provider_configs()
    col1, col2 = st.columns(2)
    with col1:
        provider = st.selectbox("Provider", provider_configs, format_func=lambda p: p.stem, key="debug_provider")
    with col2:
        top_k = st.slider("Top-K", 1, 20, 5, key="debug_topk")

    if st.button("🔍 이 질문 실행", type="primary", key="debug_run_btn"):
        with st.status("디버깅 실행 중...", expanded=True) as status:
            status.write("🔍 검색 중...")
            try:
                result = run_live_query(
                    question=selected_q["question"],
                    provider_config_path=provider,
                    top_k=top_k,
                )
                status.update(label="완료", state="complete")

                # 1) 검색 단계
                st.markdown("### 1단계: 검색")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**의도된 필터** (metadata_filter)")
                    st.json(selected_q.get("metadata_filter") or "자동 추출")
                with col2:
                    st.markdown("**기대 문서** (ground_truth_docs)")
                    st.json(selected_q.get("ground_truth_docs", []))

                if result.retrieved_chunks:
                    chunks_data = [
                        {
                            "순위": c.rank,
                            "유사도": round(c.score, 4),
                            "사업명": c.chunk.metadata.get("사업명", "")[:25],
                            "발주기관": c.chunk.metadata.get("발주기관", ""),
                            "도메인": c.chunk.metadata.get("사업도메인", ""),
                            "유형": c.chunk.content_type,
                            "내용": c.chunk.text[:120],
                        }
                        for c in result.retrieved_chunks
                    ]
                    st.dataframe(chunks_data, use_container_width=True)

                    # 정답 문서 포함 여부
                    retrieved_docs = [c.chunk.metadata.get("사업명", "") for c in result.retrieved_chunks]
                    expected = selected_q.get("ground_truth_docs", [])
                    if expected:
                        found = [e for e in expected if any(e in d for d in retrieved_docs)]
                        if len(found) == len(expected):
                            st.success(f"검색 성공: 기대 문서 {len(found)}/{len(expected)}건 포함")
                        else:
                            st.warning(f"검색 부분 성공: 기대 문서 {len(found)}/{len(expected)}건만 포함")

                # 2) 생성 단계
                st.markdown("### 2단계: 생성 — 답변 vs 정답 비교")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**생성된 답변**")
                    st.text_area("답변", result.answer[:2000], height=300, key="debug_answer", disabled=True)
                with col2:
                    st.markdown("**정답 (Ground Truth)**")
                    st.text_area("정답", selected_q.get("ground_truth_answer", "없음"), height=300, key="debug_gt", disabled=True)

                # 3) 메트릭
                st.markdown("### 3단계: 메트릭")
                cols = st.columns(4)
                cols[0].metric("검색 청크", f"{len(result.retrieved_chunks)}개")
                cols[1].metric("컨텍스트", f"{len(result.context) if result.context else 0:,}자")
                cols[2].metric("토큰", f"{result.token_usage.get('total', 0):,}")
                cols[3].metric("응답시간", f"{round(result.latency_ms)}ms")

            except Exception as e:
                status.update(label="오류", state="error")
                st.error(str(e))


def _render_compare_tab(st, load_benchmark_frames, load_run_records):
    st.subheader("결과 비교")

    # session에 저장된 run 결과 사용
    eval_runs = st.session_state.get("eval_results", {})

    # 파일 기반 결과도 로딩
    run_files = sorted(RUNS_DIR.glob("*.jsonl")) if RUNS_DIR.exists() else []

    all_sources = {}
    for run_id, results in eval_runs.items():
        all_sources[f"[세션] {run_id}"] = pd.DataFrame(results)
    for rf in run_files:
        try:
            records = [json.loads(line) for line in rf.read_text().splitlines() if line.strip()]
            if records:
                all_sources[f"[파일] {rf.stem}"] = pd.DataFrame(records)
        except Exception:
            pass

    if len(all_sources) < 2:
        st.info("비교하려면 최소 2개 이상의 평가 결과가 필요합니다. '평가 실행' 탭에서 다른 설정으로 실행해보세요.")
        if all_sources:
            st.caption(f"현재 결과: {len(all_sources)}개")
            for name, df in all_sources.items():
                with st.expander(name):
                    st.dataframe(df, use_container_width=True)
        return

    col1, col2 = st.columns(2)
    source_names = list(all_sources.keys())
    with col1:
        left = st.selectbox("왼쪽 결과", source_names, index=0, key="compare_left")
    with col2:
        right_default = 1 if len(source_names) > 1 else 0
        right = st.selectbox("오른쪽 결과", source_names, index=right_default, key="compare_right")

    left_df = all_sources[left]
    right_df = all_sources[right]

    # 나란히 비교
    st.markdown("### 질문별 비교")
    if "id" in left_df.columns and "id" in right_df.columns:
        merged = left_df.merge(right_df, on="id", suffixes=("_좌", "_우"), how="outer")
        display_cols = [c for c in merged.columns if "id" in c or "answer" in c or "tokens" in c or "latency" in c]
        st.dataframe(merged[display_cols] if display_cols else merged, use_container_width=True)
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{left}**")
            st.dataframe(left_df, use_container_width=True)
        with col2:
            st.markdown(f"**{right}**")
            st.dataframe(right_df, use_container_width=True)

    # 집계 비교
    st.markdown("### 집계 비교")
    if "type" in left_df.columns and "type" in right_df.columns:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{left}** — 유형별")
            if "tokens" in left_df.columns:
                st.dataframe(left_df.groupby("type")["tokens"].mean().round(0), use_container_width=True)
        with col2:
            st.markdown(f"**{right}** — 유형별")
            if "tokens" in right_df.columns:
                st.dataframe(right_df.groupby("type")["tokens"].mean().round(0), use_container_width=True)


def _render_edit_tab(st, eval_set):
    st.subheader("평가셋 편집")
    st.caption(f"현재 {len(eval_set)}개 질문 (세션 편집 중 — 파일 저장은 별도)")

    # 질문 목록 테이블
    if eval_set:
        table_data = [
            {
                "ID": q["id"],
                "유형": q["type"],
                "난이도": q.get("difficulty", "중"),
                "질문": q["question"][:50],
                "기대문서": ", ".join(q.get("ground_truth_docs", []))[:30],
            }
            for q in eval_set
        ]
        st.dataframe(table_data, use_container_width=True)

    # 편집할 질문 선택 또는 새로 추가
    action = st.radio("작업", ["기존 질문 편집", "새 질문 추가", "질문 삭제"], horizontal=True, key="edit_action")

    if action == "기존 질문 편집" and eval_set:
        selected_idx = st.selectbox(
            "편집할 질문",
            range(len(eval_set)),
            format_func=lambda i: f"[{eval_set[i]['id']}] {eval_set[i]['question'][:50]}",
            key="edit_select",
        )
        q = eval_set[selected_idx]
        _render_question_form(st, q, key_prefix="edit", edit_mode=True, idx=selected_idx)

    elif action == "새 질문 추가":
        new_q = {
            "id": f"Q{len(eval_set)+1:03d}",
            "type": "A",
            "difficulty": "중",
            "question": "",
            "ground_truth_answer": "",
            "ground_truth_docs": [],
            "metadata_filter": None,
            "history": None,
        }
        _render_question_form(st, new_q, key_prefix="new", edit_mode=False, idx=None)

    elif action == "질문 삭제" and eval_set:
        del_idx = st.selectbox(
            "삭제할 질문",
            range(len(eval_set)),
            format_func=lambda i: f"[{eval_set[i]['id']}] {eval_set[i]['question'][:50]}",
            key="del_select",
        )
        if st.button("🗑️ 삭제", type="secondary", key="del_btn"):
            st.session_state.eval_set.pop(del_idx)
            st.toast(f"질문 삭제됨", icon="🗑️")
            st.rerun()

    # 파일 저장/로딩
    st.divider()

    # 소스 파일 선택
    csv_files = sorted(EVAL_DIR.glob("eval_batch_*.csv"))
    source_options = [f.name for f in csv_files]
    if EVAL_SET_PATH.exists():
        source_options.append(EVAL_SET_PATH.name)

    if source_options:
        st.caption(f"사용 가능한 평가 파일: {', '.join(source_options)}")

    col1, col2, col3 = st.columns(3)
    with col1:
        save_fmt = st.selectbox("저장 형식", ["csv", "json"], key="save_fmt")
        if st.button("💾 파일에 저장", type="primary", use_container_width=True, key="save_eval"):
            path = save_eval_set(st.session_state.eval_set, fmt=save_fmt)
            st.success(f"저장 완료: {path} ({len(st.session_state.eval_set)}개)")
    with col2:
        if st.button("🔄 파일에서 다시 로딩", use_container_width=True, key="reload_eval"):
            st.session_state.eval_set = load_eval_set()
            st.toast(f"로딩 완료: {len(st.session_state.eval_set)}개 질문", icon="🔄")
            st.rerun()
    with col3:
        # CSV 업로드
        uploaded = st.file_uploader("CSV 업로드", type=["csv"], key="upload_csv", label_visibility="collapsed")
        if uploaded:
            import io
            df = pd.read_csv(io.StringIO(uploaded.read().decode("utf-8-sig")))
            new_records = []
            for _, row in df.iterrows():
                new_records.append({
                    "id": str(row.get("id", "")),
                    "type": str(row.get("type", "A")),
                    "difficulty": str(row.get("difficulty", "중")),
                    "question": str(row.get("question", "")),
                    "ground_truth_answer": str(row.get("ground_truth_answer", "")),
                    "ground_truth_docs": _parse_json_field(row.get("ground_truth_docs")) or [],
                    "metadata_filter": _parse_json_field(row.get("metadata_filter")),
                    "history": _parse_json_field(row.get("history")),
                })
            st.session_state.eval_set = new_records
            st.toast(f"업로드 완료: {len(new_records)}개 질문", icon="📤")
            st.rerun()


def _render_question_form(st, q: dict, key_prefix: str, edit_mode: bool, idx: int | None):
    """질문 편집 폼."""
    col1, col2, col3 = st.columns(3)
    with col1:
        q_id = st.text_input("ID", value=q["id"], key=f"{key_prefix}_id")
    with col2:
        q_type = st.selectbox("유형", ["A", "B", "C", "D", "E"],
                               index=["A", "B", "C", "D", "E"].index(q.get("type", "A")),
                               key=f"{key_prefix}_type")
    with col3:
        q_diff = st.selectbox("난이도", ["하", "중", "상"],
                               index=["하", "중", "상"].index(q.get("difficulty", "중")),
                               key=f"{key_prefix}_diff")

    q_question = st.text_area("질문", value=q.get("question", ""), height=80, key=f"{key_prefix}_question")
    q_gt_answer = st.text_area("정답 (Ground Truth)", value=q.get("ground_truth_answer", ""), height=120, key=f"{key_prefix}_gt")
    q_gt_docs = st.text_input("기대 문서 (쉼표 구분)", value=", ".join(q.get("ground_truth_docs", [])), key=f"{key_prefix}_docs")
    q_meta_filter = st.text_input("메타데이터 필터 (JSON)", value=json.dumps(q.get("metadata_filter") or {}, ensure_ascii=False), key=f"{key_prefix}_filter")

    # C유형: history
    q_history = None
    if q_type == "C":
        history_str = json.dumps(q.get("history") or [], ensure_ascii=False, indent=2)
        q_history_raw = st.text_area("대화 히스토리 (JSON)", value=history_str, height=150, key=f"{key_prefix}_history")
        try:
            q_history = json.loads(q_history_raw) if q_history_raw.strip() else None
        except json.JSONDecodeError:
            st.warning("히스토리 JSON 형식이 올바르지 않습니다.")
            q_history = q.get("history")

    # 저장 버튼
    btn_label = "✏️ 수정 반영" if edit_mode else "➕ 질문 추가"
    if st.button(btn_label, type="primary", key=f"{key_prefix}_save_btn"):
        updated = {
            "id": q_id,
            "type": q_type,
            "difficulty": q_diff,
            "question": q_question,
            "ground_truth_answer": q_gt_answer,
            "ground_truth_docs": [d.strip() for d in q_gt_docs.split(",") if d.strip()],
            "metadata_filter": json.loads(q_meta_filter) if q_meta_filter.strip() and q_meta_filter.strip() != "{}" else None,
            "history": q_history,
        }

        if edit_mode and idx is not None:
            st.session_state.eval_set[idx] = updated
            st.toast(f"질문 {q_id} 수정됨", icon="✏️")
        else:
            st.session_state.eval_set.append(updated)
            st.toast(f"질문 {q_id} 추가됨", icon="➕")
        st.rerun()
