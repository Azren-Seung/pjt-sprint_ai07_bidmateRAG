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
from pathlib import Path
from uuid import uuid4

import yaml
from dotenv import load_dotenv
load_dotenv()


def main() -> None:
    """실험 config를 읽고 ingest → build_index → eval → report를 순차 실행"""
    # CLI 인자 정의
    parser = argparse.ArgumentParser(description="Run a full chunking experiment.")
    parser.add_argument("--experiment-config", required=True, help="실험 config YAML")
    parser.add_argument("--eval-path", default="data/eval/eval_batch_01.csv")
    parser.add_argument("--skip-ingest", action="store_true", help="ingest 건너뛰기 (이미 실행된 경우)")
    parser.add_argument("--skip-judge", action="store_true", help="LLM judge 평가 건너뛰기")
    args = parser.parse_args()

    # 실험 config에서 청킹/프로바이더 설정 로딩
    exp_cfg = yaml.safe_load(Path(args.experiment_config).read_text())
    exp_name = exp_cfg["name"]
    chunk_size = exp_cfg.get("chunk_size", 1000)
    chunk_overlap = exp_cfg.get("chunk_overlap", 150)
    provider_configs = exp_cfg.get("provider_configs", ["configs/providers/openai_gpt5mini.yaml"])

    # 실험별 경로 설정
    processed_dir = f"data/processed/{exp_name}"
    chunks_path = f"{processed_dir}/chunks.parquet"
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
            "--experiment-config", args.experiment_config,
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
                "--experiment-config", args.experiment_config,
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

    # 결과 요약 출력
    print(f"\n{'='*60}")
    print(f"실험 완료: {exp_name}")
    print(f"결과: artifacts/logs/benchmarks/")
    if generated_reports:
        print(f"리포트: artifacts/reports/ ({len(generated_reports)}개 생성)")
        for rid in generated_reports:
            print(f"  - {exp_name}_{rid}.md")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
