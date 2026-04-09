"""Backwards-compatible shim for the benchmark CLI.

The real implementation lives in :mod:`bidmate_rag.cli.eval`. This file is kept
so that existing invocations like `uv run python scripts/run_eval.py ...`
continue to work; new usage should prefer the `bidmate-eval` console script.
"""

from __future__ import annotations

from bidmate_rag.cli.eval import main

if __name__ == "__main__":
    main()
