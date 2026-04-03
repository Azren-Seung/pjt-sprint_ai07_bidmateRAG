# BidMate RAG

Scaffold for an RFP-focused RAG project with separate areas for app serving, ingestion, retrieval, generation, evaluation, and experiment tracking.

## Quick Start

```bash
uv sync --group dev
cp .env.example .env
uv run python main.py
```

## Development Commands

```bash
make sync
make run
make test
make lint
make format
```

## Python Version

This project is pinned to Python `3.12` through `.python-version` so the team can use a consistent interpreter with `uv`.

## Project Docs

- `docs/project-structure.md`: repository folder and file guide
- `docs/architecture.md`: system architecture notes
- `docs/decision-log.md`: technical decision history
- `docs/collaboration/branch-strategy.md`: team branch strategy
- `docs/collaboration/git-worktree-workflow.md`: team git worktree workflow

## Collaboration Rule

We use git worktrees for feature work and experiments so each branch can run in an isolated workspace without disturbing the main working directory.
