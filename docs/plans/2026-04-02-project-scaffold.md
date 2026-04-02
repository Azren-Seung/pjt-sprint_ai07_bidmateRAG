# Project Scaffold Implementation Plan

> This plan records the initial repository scaffold so the team can understand why the structure was created this way.

**Goal:** Set up a reusable project scaffold for the BidMate RAG team.

**Architecture:** Keep app-serving, reusable RAG modules, experiments, and documentation separate so scenario A and B can share the same retrieval and evaluation workflow.

**Tech Stack:** Python, OpenAI-compatible provider adapters, local vector storage, markdown docs.

---

### Task 1: Directory scaffold

**Files:**
- Create: `app/`, `configs/`, `docs/`, `scripts/`, `src/bidmate_rag/`, `tests/`

- [x] Create the agreed directories for application, pipeline, evaluation, experiments, and reports.
- [x] Move raw RFP files under `data/raw/`.

### Task 2: Placeholder files

**Files:**
- Create: `README.md`
- Create: `pyproject.toml`
- Create: `app/main.py`

- [x] Add placeholder files so teammates can start filling modules without debating paths.
- [x] Keep the root `main.py` as a thin compatibility entrypoint.

### Task 3: Ignore rules

**Files:**
- Modify: `.gitignore`

- [x] Ignore raw data and generated artifacts while keeping tracked placeholders.
