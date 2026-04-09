# BidMate RAG

RFP(제안요청서) 기반 질의응답과 비교 분석을 위한 RAG 프로젝트입니다. 현재 구조는 노트북 실험 결과를 공통 파이프라인으로 이전하고, 시나리오 A/B 비교와 Streamlit UI까지 같은 결과 포맷으로 연결하는 것을 목표로 합니다.

## 현재 구현 범위

- 공통 파이프라인: 파싱 → 정제 → 청킹 → 인덱싱
- 시나리오 B: OpenAI `gpt-5`, `gpt-5-mini`, `gpt-5-nano`
- 시나리오 A: local HF/vLLM 경로 + PEFT 준비 스크립트
- 평가: JSONL run 기록 + parquet benchmark 요약 저장
- UI: Streamlit 2탭 구조
  - 평가 비교
  - 라이브 데모

## Quick Start

```bash
uv sync --group dev
cp .env.example .env
uv run python main.py
```

선택 설치:

```bash
uv sync --group dev --group ui
uv sync --group dev --group ml
```

## 주요 실행 명령

```bash
uv run python scripts/ingest_data.py
uv run python scripts/build_index.py --provider-config configs/providers/openai_gpt5mini.yaml
uv run python scripts/run_rag.py --provider-config configs/providers/openai_gpt5mini.yaml --question "국민연금공단 이러닝 사업 요구사항 정리해줘"
uv run python scripts/run_eval.py --evaluation-path data/eval/<your_eval_file>.jsonl --provider-config configs/providers/openai_gpt5mini.yaml
streamlit run app/main.py
```

## 시나리오 설정

- 시나리오 B provider:
  - `configs/providers/openai_gpt5nano.yaml`
  - `configs/providers/openai_gpt5mini.yaml`
  - `configs/providers/openai_gpt5.yaml`
- 시나리오 A provider:
  - `configs/providers/local_hf.yaml`
  - `configs/providers/local_vllm.yaml`

실험 config:

- `configs/experiments/generation_compare.yaml`
- `configs/experiments/full_rag_compare.yaml`

## 프로젝트 문서

- `docs/project-structure.md`: 저장소 구조와 파일 가이드
- `docs/architecture.md`: 현재 파이프라인 구조
- `docs/decision-log.md`: 실험과 기술 의사결정 기록
- `docs/collaboration/branch-strategy.md`: 팀 브랜치 전략
- `docs/collaboration/git-worktree-workflow.md`: 팀 worktree 워크플로우

## 협업 규칙

기능 개발과 실험은 git worktree 기반으로 진행하고, 원본 RFP 데이터와 실험 산출물은 Git에 올리지 않습니다.
