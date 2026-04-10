"""Multi-run benchmark 비교 모듈.

여러 평가 run의 메트릭을 한 표로 합치고, 메트릭별 best/worst를 분석해
마크다운으로 렌더링합니다. ``bidmate-compare`` CLI 본체에서 사용.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


# 비교 표에 보일 메트릭 컬럼들 (정해진 순서로 노출)
_METRIC_COLUMNS = [
    "hit_rate@5",
    "mrr",
    "ndcg@5",
    "faithfulness",
    "answer_relevance",
    "context_precision",
    "context_recall",
    "avg_latency_ms",
    "total_cost_usd",
]

# 각 메트릭의 방향성 — best는 max인지 min인지
_HIGHER_IS_BETTER = {
    "hit_rate@5",
    "mrr",
    "ndcg@5",
    "faithfulness",
    "answer_relevance",
    "context_precision",
    "context_recall",
}


@dataclass
class ComparisonData:
    rows: pd.DataFrame
    metric_columns: list[str]


def load_runs_for_comparison(
    run_ids: list[str] | None = None,
    experiment_name: str | None = None,
    benchmarks_dir: str | Path = "artifacts/logs/benchmarks",
) -> ComparisonData:
    """run_id 리스트 또는 experiment_name으로 비교 대상 로드.

    - ``experiment_name`` 명시: 그 experiment의 모든 run 비교 (parquet 1개)
    - ``run_ids`` 명시: benchmarks_dir 안의 모든 parquet 스캔 후 해당 run_id
      행 수집 (다른 experiment끼리도 비교 가능)
    - 둘 다 None: ``ValueError``
    """
    if not experiment_name and not run_ids:
        raise ValueError("Either experiment_name or run_ids is required")

    benchmark_path = Path(benchmarks_dir)
    if not benchmark_path.exists():
        raise FileNotFoundError(f"benchmarks dir not found: {benchmark_path}")

    frames: list[pd.DataFrame] = []
    if experiment_name:
        target = benchmark_path / f"{experiment_name}.parquet"
        if not target.exists():
            raise FileNotFoundError(f"benchmark file not found: {target}")
        frames.append(pd.read_parquet(target))
    else:
        run_id_set = set(run_ids or [])
        for parquet_file in sorted(benchmark_path.glob("*.parquet")):
            try:
                df = pd.read_parquet(parquet_file)
            except Exception:
                continue
            if "run_id" not in df.columns:
                continue
            matched = df[df["run_id"].astype(str).isin(run_id_set)]
            if not matched.empty:
                frames.append(matched)

    if not frames:
        raise ValueError("No matching runs found")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    metric_cols = [c for c in _METRIC_COLUMNS if c in combined.columns]
    return ComparisonData(rows=combined, metric_columns=metric_cols)


def render_comparison_markdown(data: ComparisonData) -> str:
    """비교 결과를 마크다운으로 렌더링.

    섹션:
        # Run Comparison (요약 + 실험 목록)
        ## 메트릭 비교    (run × metric 표)
        ## 메트릭별 최우/최저 (각 metric의 best/worst run)
    """
    df = data.rows.copy()
    if df.empty:
        return "(no runs to compare)"

    lines = ["# Run Comparison", ""]
    lines.append(f"**Total runs**: {len(df)}")
    if "experiment_name" in df.columns:
        exps = sorted(df["experiment_name"].dropna().astype(str).unique().tolist())
        if exps:
            lines.append(f"**Experiments**: {', '.join(exps)}")
    lines.append("")

    # 메트릭 표 (run × metric)
    lines.append("## 메트릭 비교")
    lines.append("")
    label_cols = [c for c in ("run_id", "experiment_name", "provider_label") if c in df.columns]
    show = df[label_cols + data.metric_columns]
    lines.append(_df_to_markdown(show))
    lines.append("")

    # 메트릭별 best/worst
    if data.metric_columns:
        lines.append("## 메트릭별 최우/최저")
        lines.append("")
        lines.append("| 메트릭 | 최우 run | 값 | 최저 run | 값 |")
        lines.append("| --- | --- | --- | --- | --- |")
        run_id_col = df["run_id"] if "run_id" in df.columns else pd.Series(["?"] * len(df))
        for col in data.metric_columns:
            s = df[col].dropna()
            if s.empty:
                continue
            if col in _HIGHER_IS_BETTER:
                best_idx, worst_idx = s.idxmax(), s.idxmin()
            else:
                best_idx, worst_idx = s.idxmin(), s.idxmax()
            best_run = run_id_col.iloc[best_idx] if best_idx in run_id_col.index else run_id_col[best_idx]
            worst_run = run_id_col.iloc[worst_idx] if worst_idx in run_id_col.index else run_id_col[worst_idx]
            lines.append(
                f"| {col} | `{best_run}` | {s[best_idx]:.4f} | "
                f"`{worst_run}` | {s[worst_idx]:.4f} |"
            )

    return "\n".join(lines)


def _df_to_markdown(df: pd.DataFrame) -> str:
    """tabulate 의존 없이 마크다운 표 직접 렌더."""
    if df.empty:
        return "(empty)"
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        cells = []
        for h in headers:
            v = row[h]
            if pd.isna(v):
                cells.append("N/A")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
