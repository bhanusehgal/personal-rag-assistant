"""Quality gate: PASS/STOP banner + exit code against FAITHFULNESS_THRESHOLD.

DESIGN.md section 8. Process rule: re-run and pass this after any change to
prompt version, schema, or retrieval config, before continuing to the next
stage. A failed gate means stop and fix, not proceed and note it.

FAITHFULNESS_THRESHOLD is a placeholder — confirm with the user after the
first real run's numbers (see PROGRESS.md).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FAITHFULNESS_THRESHOLD = 0.80

RUNS_DIR = Path(__file__).resolve().parent / "runs"


def compute_pass_rate(results_path: Path) -> tuple[float, int, int]:
    total = 0
    passed = 0
    with results_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += 1
            if row.get("faithfulness_pass"):
                passed += 1
    rate = passed / total if total else 0.0
    return rate, passed, total


def main() -> None:
    parser = argparse.ArgumentParser(description="Faithfulness quality gate.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--threshold", type=float, default=FAITHFULNESS_THRESHOLD)
    args = parser.parse_args()

    results_path = RUNS_DIR / args.run_id / "results.jsonl"
    if not results_path.exists():
        print(f"[error] No results found at {results_path}")
        sys.exit(2)

    rate, passed, total = compute_pass_rate(results_path)
    pct = rate * 100
    threshold_pct = args.threshold * 100

    if rate >= args.threshold:
        print(f"PASS: faithfulness gate passed ({pct:.1f}% >= {threshold_pct:.1f}% threshold, {passed}/{total} questions)")
        sys.exit(0)
    else:
        print(f"STOP: faithfulness gate FAILED ({pct:.1f}% < {threshold_pct:.1f}% threshold, {passed}/{total} questions) — do not proceed until this is fixed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
