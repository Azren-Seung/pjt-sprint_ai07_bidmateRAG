"""실험 전체 파이프라인 실행 — ingest → build_index → eval → report."""

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
    parser = argparse.ArgumentParser(description="Run a full chunking experiment.")
    parser.add_argument("--experiment-config", required=True, help="실험 config YAML")
    parser.add_argument("--eval-path", default="data/eval/eval_batch_01.csv")
    parser.add_argument("--skip-ingest", action="store_true", help="ingest 건너뛰기 (이미 실행된 경우)")
    parser.add_argument("--skip-judge", action="store_true", help="LLM judge 평가 건너뛰기")
    args = parser.parse_args()

    exp_cfg = yaml.safe_load(Path(args.experiment_config).read_text())
    exp_name = exp_cfg["name"]
    chunk_size = exp_cfg.get("chunk_size", 1000)
    chunk_overlap = exp_cfg.get("chunk_overlap", 150)
    provider_configs = exp_cfg.get("provider_configs", ["configs/providers/openai_gpt5mini.yaml"])

    processed_dir = f"data/processed/{exp_name}"
    chunks_path = f"{processed_dir}/chunks.parquet"
    persist_dir = "artifacts/chroma_db"

    print(f"{'='*60}")
    print(f"실험: {exp_name}")
    print(f"청킹: size={chunk_size}, overlap={chunk_overlap}")
    print(f"Provider: {provider_configs}")
    print(f"{'='*60}")

    # Step 1: Ingest
    if not args.skip_ingest:
        print("\n[1/3] Ingest (파싱 → 정제 → 청킹)...")
        subprocess.run([
            sys.executable, "scripts/ingest_data.py",
            "--experiment-config", args.experiment_config,
        ], check=True)
    else:
        print("\n[1/3] Ingest 건너뜀")

    # Step 2: Build Index (각 provider별)
    for provider_config in provider_configs:
        print(f"\n[2/3] Build Index ({provider_config})...")
        subprocess.run([
            sys.executable, "scripts/build_index.py",
            "--provider-config", provider_config,
            "--chunks-path", chunks_path,
            "--persist-dir", persist_dir,
        ], check=True)

    # Step 3: Eval + Report (각 provider별)
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

            print(f"\n[리포트 생성] run_id={run_id}")
            try:
                subprocess.run([
                    sys.executable, "scripts/generate_report.py",
                    "--run-id", run_id,
                ], check=True)
                generated_reports.append(run_id)
            except subprocess.CalledProcessError as exc:
                print(f"⚠️ 리포트 생성 실패 ({run_id}): {exc}")
    else:
        print(f"\n[3/3] Eval 건너뜀 (평가셋 없음: {args.eval_path})")

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
