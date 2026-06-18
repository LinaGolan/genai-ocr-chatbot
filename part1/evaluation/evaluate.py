from __future__ import annotations

"""
Offline accuracy evaluation harness for Part 1.

Usage (from project root):
    python -m part1.evaluation.evaluate

For each PDF in phase1_data/ that has a matching ground-truth JSON in
part1/evaluation/ground_truth/, the harness runs the full pipeline and
reports field-level exact-match accuracy and completeness.

Ground-truth files must be hand-labelled.  See ground_truth/README.txt.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_PHASE1_DATA = _PROJECT_ROOT / "phase1_data"
_GROUND_TRUTH_DIR = Path(__file__).parent / "ground_truth"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten(data: Any, prefix: str = "") -> dict[str, str]:
    """Flatten a nested dict to dot-separated keys with string leaf values."""
    if isinstance(data, dict):
        out: dict[str, str] = {}
        for k, v in data.items():
            child_prefix = f"{prefix}.{k}" if prefix else k
            out.update(_flatten(v, child_prefix))
        return out
    return {prefix: str(data) if data is not None else ""}


def _exact_match(predicted: str, expected: str) -> bool:
    return predicted.strip() == expected.strip()


def _evaluate_one(pdf_path: Path, gt_path: Path) -> dict:
    """
    Run the pipeline on *pdf_path* and compare against *gt_path*.
    Returns a per-file metrics dict.
    """
    from part1.backend.ocr_client import analyze_document
    from part1.backend.extractor import extract_fields
    from part1.backend.vision_corrector import correct_and_validate

    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    with open(gt_path, encoding="utf-8") as f:
        ground_truth: dict = json.load(f)

    t0 = time.perf_counter()
    ocr = analyze_document(file_bytes, pdf_path.name)
    extracted = extract_fields(ocr.markdown)
    extracted, val = correct_and_validate(extracted, ocr, file_bytes, pdf_path.name)
    elapsed = time.perf_counter() - t0

    pred_flat = _flatten(extracted)
    gt_flat = _flatten(ground_truth)

    # Only evaluate fields that are in the ground truth
    field_results: dict[str, bool] = {}
    for key, expected_val in gt_flat.items():
        if not expected_val:
            continue  # skip truly empty GT fields (unlabelled)
        predicted_val = pred_flat.get(key, "")
        field_results[key] = _exact_match(predicted_val, expected_val)

    correct = sum(field_results.values())
    total = len(field_results)
    accuracy = correct / total if total > 0 else 0.0

    return {
        "file": pdf_path.name,
        "latency_s": round(elapsed, 1),
        "field_accuracy": round(accuracy, 3),
        "correct_fields": correct,
        "total_labelled_fields": total,
        "completeness": round(val.completeness, 3),
        "accuracy_estimate": val.accuracy_estimate,
        "per_field": {k: ("✓" if v else "✗") for k, v in field_results.items()},
    }


def run_evaluation(
    pdf_dir: Path = _PHASE1_DATA,
    gt_dir: Path = _GROUND_TRUTH_DIR,
) -> list[dict]:
    """Run evaluation on all PDFs with matching ground-truth files."""
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    results = []

    for pdf_path in pdf_paths:
        gt_path = gt_dir / (pdf_path.stem + ".json")
        if not gt_path.exists():
            print(f"  [skip] No ground truth for {pdf_path.name}")
            continue

        # Check if GT is all-empty (not yet labelled)
        with open(gt_path, encoding="utf-8") as f:
            gt_data = json.load(f)
        if not any(_flatten(gt_data).values()):
            print(f"  [skip] Ground truth for {pdf_path.name} is not yet labelled")
            continue

        print(f"  Evaluating {pdf_path.name} ...", end="", flush=True)
        try:
            result = _evaluate_one(pdf_path, gt_path)
            results.append(result)
            print(f"  accuracy={result['field_accuracy']:.1%}  completeness={result['completeness']:.1%}")
        except Exception as exc:
            print(f"  ERROR: {exc}")

    return results


def print_report(results: list[dict]) -> None:
    if not results:
        print("\nNo results to report. Label ground-truth files in part1/evaluation/ground_truth/.")
        return

    print("\n" + "=" * 60)
    print("  BL283 Extraction Evaluation Report")
    print("=" * 60)

    all_acc = [r["field_accuracy"] for r in results]
    all_comp = [r["completeness"] for r in results]

    print(f"\n  Files evaluated: {len(results)}")
    print(f"  Avg field accuracy:  {sum(all_acc)/len(all_acc):.1%}")
    print(f"  Avg completeness:    {sum(all_comp)/len(all_comp):.1%}")

    print("\n  Per-file breakdown:")
    print(f"  {'File':<20} {'Accuracy':>10} {'Completeness':>14} {'Estimate':>10} {'Latency':>9}")
    print("  " + "-" * 66)
    for r in results:
        print(
            f"  {r['file']:<20} {r['field_accuracy']:>9.1%} "
            f"{r['completeness']:>13.1%} {r['accuracy_estimate']:>10} "
            f"{r['latency_s']:>7.1f}s"
        )

    # Per-field breakdown across all files
    all_fields: dict[str, list[bool]] = {}
    for r in results:
        for field, mark in r["per_field"].items():
            all_fields.setdefault(field, []).append(mark == "✓")

    if all_fields:
        print("\n  Per-field accuracy (across all evaluated files):")
        print(f"  {'Field':<40} {'Accuracy':>10}")
        print("  " + "-" * 52)
        for fname, hits in sorted(all_fields.items()):
            fa = sum(hits) / len(hits)
            print(f"  {fname:<40} {fa:>9.1%}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("Running BL283 extraction evaluation …\n")
    evaluation_results = run_evaluation()
    print_report(evaluation_results)

    if not evaluation_results:
        sys.exit(0)

    # Save JSON report next to this file
    report_path = Path(__file__).parent / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
    print(f"\n  Report saved to {report_path}")
