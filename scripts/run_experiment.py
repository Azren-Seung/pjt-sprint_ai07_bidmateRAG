"""실험 전체 파이프라인 실행 — ingest → build_index → eval → report.

실험 config YAML 하나로 인제스트부터 평가, 리포트까지 한 번에 돌린다.
청킹 설정(size/overlap)과 provider를 실험별로 바꿔가며 비교할 때 사용.

사용 예시::

    uv run python scripts/run_experiment.py \\
        --experiment-config configs/experiments/exp_01.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

import yaml
from dotenv import load_dotenv

load_dotenv()

from bidmate_rag.evaluation.dataset import find_latest_eval_dir
from bidmate_rag.experiments.matrix import apply_overrides_to_yaml_dict, expand_matrix


def _run_single_experiment(
    exp_cfg: dict,
    args,
    experiment_config_path: str,
    chunks_basename: str | None = None,
) -> None:
    """단일 실험 1회 실행 (ingest → build_index → eval → report).

    matrix expand로 만들어진 임시 yaml도 동일하게 처리하기 위해 본 함수로
    분리. ``experiment_config_path``는 자식 subprocess에 그대로 전달되므로
    임시 파일도 자식이 정상 로드 가능.

    Args:
        chunks_basename: 청크 디렉토리 이름. matrix expand 시 청킹 변경 없는
            sub-experiment는 base 실험의 chunks를 재사용해야 하므로 명시.
            None이면 ``exp_cfg['name']`` 사용 (기본/단일 실행).
    """
    exp_name = exp_cfg["name"]
    chunk_size = exp_cfg.get("chunk_size", 1000)
    chunk_overlap = exp_cfg.get("chunk_overlap", 150)
    provider_configs = exp_cfg.get(
        "provider_configs", ["configs/providers/openai_gpt5mini.yaml"]
    )

    # 청크 경로: matrix에서 chunking 변경 없는 sub-experiment는 base 청크 재사용
    chunks_dir_name = chunks_basename or exp_name
    processed_dir = f"data/processed/{chunks_dir_name}"
    chunks_path = f"{processed_dir}/chunks.parquet"
    # Fallback: 실험별 sub-dir에 청크가 없으면 top-level 청크 사용
    # (legacy ingest는 --experiment-config 없이 돌리면 data/processed/ 직접 씀)
    if args.skip_ingest and not Path(chunks_path).exists():
        legacy = "data/processed/chunks.parquet"
        if Path(legacy).exists():
            print(f"⚠️ {chunks_path} 없음 → top-level {legacy} 사용")
            chunks_path = legacy
    persist_dir = "artifacts/chroma_db"

    print(f"{'='*60}")
    print(f"실험: {exp_name}")
    print(f"청킹: size={chunk_size}, overlap={chunk_overlap}")
    print(f"Provider: {provider_configs}")
    print(f"{'='*60}")

    # Step 1: Ingest — 원본 문서를 파싱 → 정제 → 청킹
    if not args.skip_ingest:
        print("\n[1/3] Ingest (파싱 → 정제 → 청킹)...")
        subprocess.run([
            sys.executable, "scripts/ingest_data.py",
            "--experiment-config", experiment_config_path,
        ], check=True)
    else:
        print("\n[1/3] Ingest 건너뜀")

    # Step 2: Build Index — 청크를 임베딩하여 ChromaDB에 저장 (프로바이더별)
    for provider_config in provider_configs:
        print(f"\n[2/3] Build Index ({provider_config})...")
        subprocess.run([
            sys.executable, "scripts/build_index.py",
            "--provider-config", provider_config,
            "--chunks-path", chunks_path,
            "--persist-dir", persist_dir,
        ], check=True)

    # Step 3: Eval + Report — 평가셋으로 성능 측정 후 리포트 생성 (프로바이더별)
    generated_reports: list[str] = []
    if Path(args.eval_path).exists():
        for provider_config in provider_configs:
            run_id = f"bench-{uuid4().hex[:8]}"
            print(f"\n[3/3] Eval ({provider_config}) → run_id={run_id}")
            eval_cmd = [
                sys.executable, "scripts/run_eval.py",
                "--evaluation-path", args.eval_path,
                "--provider-config", provider_config,
                "--experiment-config", experiment_config_path,
                "--run-id", run_id,
            ]
            if args.skip_judge:
                eval_cmd.append("--skip-judge")
            subprocess.run(eval_cmd, check=True)

            # 리포트 자동 생성
            print(f"\n[리포트 생성] run_id={run_id}")
            try:
                subprocess.run([
                    sys.executable, "scripts/generate_report.py",
                    "--run-id", run_id,
                ], check=True)
                generated_reports.append(run_id)
            except subprocess.CalledProcessError as exc:
                print(f"리포트 생성 실패 ({run_id}): {exc}")
    else:
        print(f"\n[3/3] Eval 건너뜀 (평가셋 없음: {args.eval_path})")

    # sub-experiment 종료 요약
    print(f"\n{'─'*60}")
    print(f"sub-experiment 완료: {exp_name}")
    if generated_reports:
        print(f"리포트: artifacts/reports/ ({len(generated_reports)}개)")
        for rid in generated_reports:
            print(f"  - {exp_name}_{rid}.md")
    print(f"{'─'*60}")


def main() -> None:
    """실험 config를 읽고 ingest → build_index → eval → report를 순차 실행

    yaml에 ``matrix:`` 섹션이 있으면 카르테시안 곱으로 expand해 N개 sub-
    experiment를 자동 순차 실행. 없으면 기존처럼 단일 실행.
    """
    # CLI 인자 정의
    parser = argparse.ArgumentParser(description="Run a full chunking experiment.")
    parser.add_argument("--experiment-config", required=True, help="실험 config YAML")
    parser.add_argument(
        "--eval-path",
        default=str(find_latest_eval_dir() / "eval_batch_01.csv"),
        help="평가셋 CSV 경로 (기본: 가장 최신 eval_v* 디렉토리의 eval_batch_01.csv)",
    )
    parser.add_argument("--skip-ingest", action="store_true", help="ingest 건너뛰기 (이미 실행된 경우)")
    parser.add_argument("--skip-judge", action="store_true", help="LLM judge 평가 건너뛰기")
    args = parser.parse_args()

    # 실험 config 로드 + matrix 확인
    base_cfg = yaml.safe_load(Path(args.experiment_config).read_text())
    matrix = base_cfg.get("matrix") or {}
    cells = expand_matrix(matrix)

    if not cells:
        # 단일 실행 (기존 동작)
        _run_single_experiment(base_cfg, args, experiment_config_path=args.experiment_config)
        print(f"\n{'='*60}")
        print(f"실험 완료: {base_cfg['name']}")
        print("결과: artifacts/logs/benchmarks/")
        print(f"{'='*60}")
        return

    # Matrix expand → N개 sub-experiment 순차 실행
    base_name = base_cfg["name"]
    chunking_keys = {"chunk_size", "chunk_overlap"}
    print(f"\n{'='*60}")
    print(f"Matrix expand: {len(cells)} sub-experiments (base={base_name})")
    print(f"{'='*60}")

    completed: list[str] = []
    for i, cell in enumerate(cells, start=1):
        sub_cfg = apply_overrides_to_yaml_dict(base_cfg, cell.overrides)
        sub_cfg["name"] = f"{base_name}_{cell.name_suffix}"
        sub_cfg.pop("matrix", None)  # 자식이 다시 expand하지 않도록

        # 청킹 변경이 있는 cell만 새 chunks 디렉토리, 아니면 base 청크 재사용
        chunking_changed = bool(set(cell.overrides.keys()) & chunking_keys)
        chunks_basename = sub_cfg["name"] if chunking_changed else base_name

        # 임시 yaml 파일로 자식 subprocess에 전달
        with tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as tmp:
            yaml.safe_dump(sub_cfg, tmp, allow_unicode=True)
            tmp_path = tmp.name

        try:
            print(f"\n\n>>> [{i}/{len(cells)}] {sub_cfg['name']}")
            print(f"    chunks_dir: data/processed/{chunks_basename} ({'reuse' if not chunking_changed else 'new'})")
            _run_single_experiment(
                sub_cfg,
                args,
                experiment_config_path=tmp_path,
                chunks_basename=chunks_basename,
            )
            completed.append(sub_cfg["name"])
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    print(f"\n{'='*60}")
    print(f"Matrix grid 완료: {len(completed)}/{len(cells)} sub-experiments")
    for name in completed:
        print(f"  - {name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
