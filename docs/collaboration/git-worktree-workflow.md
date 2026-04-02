# Git Worktree Workflow

이 문서는 우리 팀이 브랜치 작업을 할 때 `git worktree`를 사용하는 기본 규칙을 정리한 문서입니다.

## Why We Use Worktrees

- 기능 개발 중에도 현재 안정적인 작업 디렉터리를 유지할 수 있습니다.
- 여러 브랜치를 동시에 열어 비교하거나 실험하기 쉽습니다.
- 실험 코드와 메인 작업 공간이 섞이지 않아 Git 상태가 깔끔해집니다.

## Team Rule

- 새로운 기능 개발, 리팩토링, 실험 브랜치 작업은 기본적으로 `git worktree`를 사용합니다.
- 메인 작업 디렉터리는 기준선 유지용으로 사용합니다.
- worktree는 프로젝트 루트 아래 숨김 디렉터리인 `.worktrees/`에 생성합니다.
- `.worktrees/`는 Git에서 추적하지 않습니다.

## Recommended Directory Convention

```text
bidmate-rag/
├── .worktrees/
│   ├── feature-retrieval-baseline/
│   ├── feature-hwp-loader/
│   └── experiment-reranker/
└── ...
```

## Branch Naming

- 기능 개발: `feature/<topic>`
- 실험 작업: `experiment/<topic>`
- 버그 수정: `fix/<topic>`
- 문서 작업: `docs/<topic>`

worktree 폴더명은 브랜치명을 파일 경로에 맞게 바꿔 사용하면 됩니다.

예시:

- 브랜치: `feature/hwp-loader`
- worktree 경로: `.worktrees/feature-hwp-loader`

## Basic Workflow

### 1. 새 브랜치용 worktree 생성

```bash
git worktree add .worktrees/feature-hwp-loader -b feature/hwp-loader
```

### 2. 생성된 worktree로 이동

```bash
cd .worktrees/feature-hwp-loader
```

### 3. 필요한 환경 세팅

프로젝트가 Python 기반이면 의존성 설치와 기본 실행 확인을 먼저 진행합니다.

예시:

```bash
uv sync
python3 main.py
```

### 4. 작업 후 커밋 및 푸시

```bash
git add .
git commit -m "feat: add hwp loader baseline"
git push -u origin feature/hwp-loader
```

### 5. 브랜치 정리 후 worktree 제거

작업이 끝나고 더 이상 해당 worktree가 필요 없으면 메인 디렉터리에서 제거합니다.

```bash
git worktree remove .worktrees/feature-hwp-loader
```

필요하면 로컬 브랜치도 정리합니다.

```bash
git branch -d feature/hwp-loader
```

## When Direct Work on `main` Is Allowed

- 프로젝트 초기 스캐폴딩처럼 아주 작은 초기 세팅
- 팀이 합의한 문서 정리 작업
- 긴급한 hotfix로 별도 브랜치 운영보다 즉시 반영이 더 중요한 경우

그 외 대부분의 코드 변경은 worktree + 브랜치 작업을 기본으로 합니다.

## If We Use Codex

Codex로 기능 작업을 시작할 때는 `using-git-worktrees` 스킬을 기준으로 worktree를 준비합니다.

원칙:

- 기존 `.worktrees/` 디렉터리가 있으면 그 위치를 사용합니다.
- project-local worktree 디렉터리는 반드시 `.gitignore`에 들어 있어야 합니다.
- worktree를 만든 뒤 기본 실행 또는 테스트로 깨끗한 시작 상태를 확인합니다.

## Notes

- worktree는 같은 저장소의 여러 브랜치를 동시에 다루기 쉽게 해 주지만, 각 worktree는 사실상 별도 작업 공간처럼 취급해야 합니다.
- 서로 다른 worktree에서 같은 파일을 동시에 크게 수정하면 병합 비용이 커질 수 있으니 작업 범위를 미리 나누는 것이 좋습니다.
