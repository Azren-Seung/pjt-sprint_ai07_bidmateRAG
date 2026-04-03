# BidMate RAG

공공·기업 제안요청서(RFP)를 빠르게 읽고, 필요한 정보를 추출·요약·질의응답할 수 있도록 만드는 RAG 프로젝트입니다.

> [!NOTE]
> 이 저장소는 현재 **프로젝트 스캐폴드 + 협업 규칙 + 개발 환경 세팅**이 완료된 상태입니다.  
> 다음 구현 우선순위는 `문서 로딩 -> 전처리/청킹 -> retrieval baseline -> generation -> 평가`입니다.

## 프로젝트 개요

이번 프로젝트는 B2G 입찰지원 전문 컨설팅 스타트업 **입찰메이트**의 사내 RAG 시스템을 만든다는 설정으로 진행합니다.

입찰메이트의 컨설턴트는 매일 올라오는 다수의 RFP를 빠르게 훑어야 합니다. 이 프로젝트의 목표는 복잡한 RFP 문서에서 핵심 요구사항, 발주 기관, 예산, 제출 방식, 사업 목적 등을 빠르게 찾고, 사용자의 질문에 문서 근거 기반으로 답변하는 시스템을 구현하는 것입니다.

## 우리가 해결하려는 문제

- 수십 페이지 분량의 RFP를 사람이 직접 모두 읽기 어렵습니다.
- 문서 포맷이 `pdf`, `hwp`, `docx` 등으로 섞여 있습니다.
- 단순 요약이 아니라, **정확한 근거 기반 질의응답**이 필요합니다.
- 여러 문서를 비교하거나 후속 질문을 이어가는 대화형 사용성이 중요합니다.
- 성능뿐 아니라 **속도, 재현성, 협업 과정**까지 함께 관리해야 합니다.

## 핵심 목표

- RFP 문서를 안정적으로 로딩하고 공통 포맷으로 정규화하기
- 문서 메타데이터와 본문을 활용해 검색 성능 높이기
- RAG 파이프라인을 통해 근거 기반 답변 생성하기
- 단일 문서 질의, 다문서 비교, 후속 질문 맥락 유지까지 지원하기
- 시나리오 A/B를 비교 실험하고 결과를 보고서와 발표에 정리하기

## 실험 시나리오

| 시나리오 | 목적 | 예시 스택 |
| --- | --- | --- |
| A. GCP 실행 기반 | 온프레미스/로컬 실행형 RAG 실험 | Hugging Face LLM, Hugging Face Embedding, FAISS/Chroma |
| B. 클라우드 API 기반 | 빠른 베이스라인과 성능 비교 | OpenAI, Gemini, Claude, OpenAI-compatible endpoint, FAISS/Chroma |

현재 저장소 구조는 두 시나리오를 모두 비교하기 쉽도록 `configs/providers/`, `configs/experiments/`, `src/bidmate_rag/providers/` 중심으로 구성되어 있습니다.

## 주요 기능 범위

- `PDF`, `HWP`, `DOCX` 문서 로딩
- 메타데이터 CSV 로딩 및 문서 매핑
- 청킹 전략 실험
- 임베딩 생성 및 벡터 인덱스 구축
- 메타데이터 필터링 기반 retrieval
- 문서 근거 기반 답변 생성
- 평가셋 기반 성능 측정 및 비교 실험
- 발표/보고서/협업 문서화

## 평가 관점

프로젝트 평가는 단순히 "답이 그럴듯한지"가 아니라 아래 관점을 함께 봅니다.

- 사용자가 요청한 내용을 단일 문서에서 정확히 뽑아내는지
- 여러 문서를 잘 종합해서 비교·정리하는지
- 후속 질문의 맥락을 이해하는지
- 문서에 없는 내용은 모른다고 답하는지
- 응답 속도가 너무 느리지 않은지

## 현재 저장소 상태

현재 레포에는 아래 기반 작업이 반영되어 있습니다.

- Python `3.12` + `uv` 개발 환경
- 기본 패키지 구조와 파이프라인 스캐폴드
- `main -> develop -> feat/<initial>/<topic>` 브랜치 전략
- `git worktree` 기반 협업 흐름
- 테스트/린트 기본 설정

## 빠른 시작

### 1. 개발 환경 준비

필수:

- Python `3.12`
- `uv`
- Git

설치 후 아래 명령으로 환경을 준비합니다.

```bash
uv sync --group dev
cp .env.example .env
uv run python main.py
```

## 자주 쓰는 명령어

```bash
make sync
make run
make test
make lint
make format
```

## 권장 협업 흐름

우리 팀은 `develop` 브랜치를 공통 기준선으로 사용하고, 개인 작업은 `feat/<initial>/<topic>` 브랜치에서 진행합니다.

예시:

```bash
git fetch origin
git worktree add .worktrees/feat-dm-loader -b feat/dm/loader-baseline origin/develop
cd .worktrees/feat-dm-loader
uv sync --group dev
uv run pytest
```

자세한 규칙은 아래 문서를 참고하세요.

- [브랜치 전략](docs/collaboration/branch-strategy.md)
- [Git worktree 워크플로우](docs/collaboration/git-worktree-workflow.md)

## 디렉터리 안내

핵심 디렉터리만 빠르게 보면 아래와 같습니다.

- `src/bidmate_rag/`: 실제 RAG 코드
- `configs/`: 모델/실험 설정
- `data/raw/`: 원본 문서와 메타데이터
- `artifacts/`: 벡터 인덱스, 로그, 캐시
- `scripts/`: ingest, index, eval 실행 스크립트
- `docs/`: 구조 문서, 협업 문서, 의사결정 기록

전체 구조는 [프로젝트 구조 문서](docs/project-structure.md)에서 확인할 수 있습니다.

## 문서 목록

- [프로젝트 구조](docs/project-structure.md)
- [아키텍처 메모](docs/architecture.md)
- [기술 의사결정 기록](docs/decision-log.md)
- [브랜치 전략](docs/collaboration/branch-strategy.md)
- [Git worktree 워크플로우](docs/collaboration/git-worktree-workflow.md)

## 다음 구현 우선순위

1. 문서 로딩 baseline 구현
2. 공통 문서 스키마 정의
3. 청킹 및 전처리 baseline 구현
4. retrieval baseline 구축
5. generation baseline 구축
6. 평가셋과 실험 러너 연결

## 데이터 보안 주의사항

> [!IMPORTANT]
> 원본 RFP 문서는 비공개 데이터입니다.  
> `data/raw/rfp/` 아래의 원본 파일은 Git에 올리지 않고, 외부 공유도 금지합니다.  
> 저장소에는 원본 데이터가 아닌 코드, 설정, 문서, 2차 가공 결과만 남깁니다.

## 프로젝트 비전

이 프로젝트의 목표는 단순히 "RAG 한 번 돌려보기"가 아닙니다.  
문서 처리, 검색, 생성, 평가, 협업, 보고서 작성까지 포함한 **실전형 팀 프로젝트 결과물**을 만드는 것이 목표입니다.
