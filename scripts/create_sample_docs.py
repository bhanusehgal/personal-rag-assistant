"""Generate 5 synthetic sample documents into data/raw_docs/ to test the
ingestion pipeline (PDF/DOCX/MD loaders, per-type chunking, metadata) before
you drop in real documents.

All content below is fabricated/placeholder — no real client or exam
material. Run once, inspect the output, then delete or replace with your own
files whenever you're ready.

Usage: python scripts/create_sample_docs.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DOCS_DIR = ROOT / "data" / "raw_docs"
MANIFEST_PATH = ROOT / "data" / "manifest.json"

VALIDATION_REPORT_MD = """# Model Validation Report: NMD Behavioral Engine (SAMPLE)

## Scope
This report documents the independent validation of the Non-Maturity Deposit
(NMD) behavioral engine used to derive repricing and decay assumptions feeding
the ALM model.

## Repricing Beta Methodology
The engine estimates repricing betas using a partial-adjustment model
regressed on a rolling 10-year window of portfolio-level rate and balance
data, segmented by product (checking, savings, MMDA). A dual-regime
specification is used to capture asymmetric repricing behavior in rising vs.
falling rate environments, motivated by the 2022-2023 tightening cycle where
observed betas diverged materially from the prior single-regime estimate.

Key finding: the dual-regime beta for MMDA (0.62 rising / 0.41 falling) is
statistically distinct from the single-regime baseline (0.51) at the 95%
confidence level, supporting the model owner's decision to adopt the
dual-regime specification going forward.

## Decay Rate Assumption
Core/non-core decomposition uses a survival-analysis approach (Kaplan-Meier
curves by vintage and product) rather than a fixed attrition rate. Validation
found the survival-curve approach outperforms the prior fixed-rate approach
in an out-of-sample back-test over 2019-2024, particularly during the 2023
deposit outflow period.

## Findings and Recommendations
1. (Medium) Document the sensitivity of the dual-regime beta to the choice of
   regime-switch threshold; current threshold (25bp cumulative move) is not
   formally justified in the model documentation.
2. (Low) Expand out-of-sample back-testing to include a full rate-cutting
   cycle once sufficient post-2024 data is available.

## Conclusion
The NMD behavioral engine is approved for continued use with the above
findings tracked to closure.
"""

REGULATORY_GUIDANCE_MD = """# SR 11-7 Excerpt: Model Risk Management Framework (SAMPLE)

## Definition of Model Risk
Model risk occurs primarily for two reasons: a model may have fundamental
errors and produce inaccurate outputs when viewed against its design
objective, or a model may be used incorrectly or inappropriately.

## Effective Challenge
Robust model validation is a critical element that helps ensure sound model
risk management. Validation of a model's conceptual soundness, ongoing
monitoring, and outcomes analysis should be performed by staff with
appropriate incentives, competence, and influence — with effective challenge
independent of the model's development and use.

## Model Tiering
Banks are expected to prioritize validation activities based on model risk
tiering, considering the materiality of the model's use, complexity, and the
extent of uncertainty around inputs and assumptions. Higher-tier models
(e.g., those feeding capital, CECL allowance, or ALM/IRRBB risk measures)
warrant more frequent and more rigorous validation.

## Ongoing Monitoring
Ongoing monitoring confirms that a model is appropriately implemented and is
performing as intended, and that assumptions remain suitable for existing
conditions. Ongoing monitoring should include process verification and
benchmarking, and should be conducted at a frequency appropriate to the
nature of the model and its inputs.
"""

PROJECT_DOC_MD = """# TII Project — Design Notes (SAMPLE)

## Background
The TII (Treasury Interest-rate Initiative) project consolidates ALM and
IRRBB assumption governance under a single review cadence, replacing the
prior ad hoc process where NMD, prepayment, and pipeline hedge assumptions
were reviewed on independent schedules.

## Key Design Decisions
- **Unified assumption inventory**: a single source-of-truth spreadsheet (to
  be migrated to a lightweight internal tool in phase 2) listing every ALM
  assumption, its owner, last review date, and next scheduled review.
- **Quarterly assumption review committee**: replaces the previous practice
  of assumption owners updating inputs unilaterally between model runs.
- **Escalation threshold**: any assumption change that moves EVE sensitivity
  by more than 5% of Tier 1 capital in the -/+200bp shock triggers mandatory
  committee review before adoption, rather than routine sign-off.

## Open Items
- Confirm whether prepayment model assumptions fall under TII scope or stay
  with the mortgage model governance track (currently ambiguous — flagged
  for resolution with model risk management).
- Determine reporting cadence to ALCO once the unified inventory is live.

## Status
Phase 1 (inventory build-out) complete. Phase 2 (tooling migration) not yet
started.
"""

CLOSURE_PACK_PARAGRAPHS = [
    "Closure Pack (SAMPLE): IRRBB Assumption Finding Remediation",
    "",
    "Finding ID: MRM-2024-014",
    "Originating Review: NMD Behavioral Engine Validation, 2024 cycle",
    "",
    "Finding Summary",
    "The regime-switch threshold used in the dual-regime repricing beta "
    "specification (25bp cumulative move) was not formally documented or "
    "justified with supporting analysis at the time of the original review.",
    "",
    "Remediation Action",
    "Model owner performed a threshold sensitivity analysis across a 10-50bp "
    "range and documented the rationale for the 25bp threshold, including "
    "back-test results showing stability of the resulting beta estimates "
    "within that range.",
    "",
    "Validation Assessment",
    "Independent validation reviewed the sensitivity analysis and confirmed "
    "the threshold choice is adequately supported. No material change to "
    "the beta estimates or model conclusions resulted from this remediation.",
    "",
    "Closure Decision",
    "Finding MRM-2024-014 is closed as of the date below. No further action "
    "required.",
]

EXAM_RESPONSE_LINES = [
    "Examiner Response (SAMPLE): CECL Loss Forecasting Assumption",
    "",
    "Question: Describe the methodology used to forecast losses under the",
    "reasonable and supportable forecast period, and how the reversion",
    "period to historical loss rates was determined.",
    "",
    "Response: The reasonable and supportable forecast period uses a",
    "12-quarter macroeconomic scenario path (baseline Moody's scenario)",
    "regressed against segment-level loss rates via a vintage-based panel",
    "model. Reversion to long-run historical average loss rates occurs over",
    "a 4-quarter straight-line reversion beginning in quarter 13, consistent",
    "with the institution's documented reversion policy.",
    "",
    "The 4-quarter reversion period length was selected based on backtesting",
    "showing shorter reversion periods (2 quarters) introduced excess",
    "volatility in the allowance estimate without a corresponding accuracy",
    "improvement, while longer periods (8 quarters) understated losses",
    "entering downturn scenarios in the 2020 backtest window.",
]


def write_md(name: str, content: str) -> None:
    (RAW_DOCS_DIR / name).write_text(content, encoding="utf-8")
    print(f"  [ok] {name}")


def write_docx(name: str, paragraphs: list[str]) -> None:
    from docx import Document

    doc = Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    doc.save(str(RAW_DOCS_DIR / name))
    print(f"  [ok] {name}")


def write_pdf(name: str, lines: list[str]) -> None:
    try:
        from fpdf import FPDF
    except ImportError as e:
        raise SystemExit(
            "fpdf2 is required to generate the sample PDF (test-only dependency, "
            "not in requirements.txt). Install it with:\n"
            "    pip install fpdf2"
        ) from e

    from fpdf.enums import XPos, YPos

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    for line in lines:
        if line.strip():
            pdf.multi_cell(0, 6, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
            pdf.ln(6)
    pdf.output(str(RAW_DOCS_DIR / name))
    print(f"  [ok] {name}")


def main() -> None:
    RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing sample documents to {RAW_DOCS_DIR}")

    write_md("sample_validation_report.md", VALIDATION_REPORT_MD)
    write_md("sample_regulatory_guidance.md", REGULATORY_GUIDANCE_MD)
    write_md("sample_project_doc_tii.md", PROJECT_DOC_MD)
    write_docx("sample_closure_pack.docx", CLOSURE_PACK_PARAGRAPHS)
    write_pdf("sample_exam_response.pdf", EXAM_RESPONSE_LINES)

    manifest = {
        "sample_validation_report.md": {
            "doc_type": "validation_report",
            "topic": "ALM",
            "date": "2024-11-15",
        },
        "sample_regulatory_guidance.md": {
            "doc_type": "regulatory_guidance",
            "topic": "other",
            "date": "2011-04-04",
        },
        "sample_project_doc_tii.md": {
            "doc_type": "project_doc",
            "topic": "IRRBB",
            "date": "2025-06-01",
        },
        "sample_closure_pack.docx": {
            "doc_type": "closure_pack",
            "topic": "IRRBB",
            "date": "2025-01-20",
        },
        "sample_exam_response.pdf": {
            "doc_type": "exam_response",
            "topic": "CECL",
            "date": "2024-09-10",
        },
    }

    existing = {}
    if MANIFEST_PATH.exists():
        existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    existing.update(manifest)
    MANIFEST_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"  [ok] manifest.json ({len(manifest)} entries)")

    print("\nDone. Run the indexer next:")
    print("    python -m ingest.indexer --provider fake   # pipeline smoke test, no API key")
    print("    python -m ingest.indexer                   # real embeddings, needs OPENAI_API_KEY in .env")


if __name__ == "__main__":
    main()
