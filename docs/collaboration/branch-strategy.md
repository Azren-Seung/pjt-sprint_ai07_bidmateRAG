# Branch Strategy

이 문서는 우리 팀의 기본 Git 브랜치 전략을 정리한 문서입니다.

## Default Flow

```text
main
└── develop
    ├── feat/jh/hwp-loader
    ├── feat/ay/retrieval-baseline
    ├── feat/mk/eval-dataset
    └── fix/jh/chunker-bug
```

## Branch Roles

- `main`
  배포 또는 제출 기준선입니다. 바로 작업하지 않습니다.
- `develop`
  팀 공통 개발 기준선입니다. 기능 브랜치는 모두 여기서 분기합니다.
- `feat/<initial>/<topic>`
  개인 기능 개발 브랜치입니다.
- `fix/<initial>/<topic>`
  버그 수정 브랜치입니다.
- `docs/<initial>/<topic>`
  문서 작업 브랜치입니다.
- `experiment/<initial>/<topic>`
  실험용 브랜치입니다.

## Naming Convention

- 기능 개발: `feat/<initial>/<topic>`
- 버그 수정: `fix/<initial>/<topic>`
- 문서 작업: `docs/<initial>/<topic>`
- 실험 작업: `experiment/<initial>/<topic>`

예시:

- `feat/ay/hwp-loader`
- `feat/jh/retrieval-hybrid-search`
- `fix/mk/metadata-filter`
- `docs/ay/readme-update`

## Team Rule

- 공통 작업 시작점은 항상 `develop`입니다.
- 개인 작업은 `feat/<initial>/<topic>` 형태의 브랜치에서 진행합니다.
- 작업 브랜치는 가능하면 `git worktree`로 분리된 작업 디렉터리에서 진행합니다.
- `main`에는 직접 기능 작업을 하지 않습니다.
- `develop`에는 개별 기능 브랜치를 병합한 뒤 반영합니다.

## Recommended Commands

### 1. 최신 브랜치 상태 가져오기

```bash
git checkout develop
git pull origin develop
```

### 2. worktree로 새 기능 브랜치 만들기

```bash
git worktree add .worktrees/feat-ay-hwp-loader -b feat/ay/hwp-loader develop
```

### 3. 새 worktree로 이동 후 환경 세팅

```bash
cd .worktrees/feat-ay-hwp-loader
uv sync --group dev
uv run pytest
```

### 4. 작업 후 푸시

```bash
git add .
git commit -m "feat: add hwp loader baseline"
git push -u origin feat/ay/hwp-loader
```

## Merge Rule

- 개인 작업 브랜치는 먼저 `develop` 기준으로 유지합니다.
- 여러 기능이 모이면 `develop`에 병합합니다.
- 발표/제출 기준으로 안정화된 시점에 `develop`에서 `main`으로 반영합니다.

## When To Work Directly On `main`

- 원칙적으로 하지 않습니다.
- 단, 초기 스캐폴딩이나 아주 작은 관리 작업처럼 팀이 명확히 합의한 경우에만 예외적으로 허용합니다.
