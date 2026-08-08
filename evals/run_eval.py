"""
Evaluation harness for the RAG pipeline.

Runs every question in evals/golden_dataset.jsonl against a real PDF,
using the exact same services the Streamlit app uses (src/services/),
and scores each answer against expected keywords and category-specific
rules (e.g. list completeness, refusal-on-out-of-scope).

Usage:
    python evals/run_eval.py --pdf data/sample_pdfs/your_test_doc.pdf

Writes a timestamped results JSON to evals/results/ and prints a
summary table to the console.
"""

import argparse
import json
import time
import sys
from pathlib import Path
from datetime import datetime, timezone

# Make src/ and utils/ importable regardless of where this is run from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.services.ingestion import ingest_pdf
from src.services.retrieval import build_index, retrieve
from src.services.generation import answer_question


REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DATASET_PATH = REPO_ROOT / "evals" / "golden_dataset.jsonl"
RESULTS_DIR = REPO_ROOT / "evals" / "results"

# Below this confidence, we treat the system as having effectively
# refused/hedged on an answer -- used for the out_of_scope category.
LOW_CONFIDENCE_THRESHOLD = 0.55


def load_golden_dataset(path):
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run_case(index, chunks, case):
    query = case["question"]
    start = time.time()

    results, scores = retrieve(index, chunks, query)
    answer, context, confidence_label = answer_question(results, scores, query)

    latency_ms = round((time.time() - start) * 1000, 1)
    top_score = float(scores[0])

    answer_lower = answer.lower()
    expected_keywords = case.get("expected_keywords", [])
    hits = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    min_hits_required = case.get("min_keyword_hits", len(expected_keywords))

    passed_keywords = len(hits) >= min_hits_required if expected_keywords else True

    if case.get("expect_refusal_or_low_confidence"):
        # For out-of-scope questions, "passing" means the system did NOT
        # confidently assert something specific -- either its confidence
        # label is Low, or the raw top similarity score is low.
        passed = confidence_label == "Low" or top_score < LOW_CONFIDENCE_THRESHOLD
    else:
        passed = passed_keywords

    return {
        "id": case["id"],
        "category": case["category"],
        "question": query,
        "answer": answer,
        "expected_keywords": expected_keywords,
        "keyword_hits": hits,
        "keyword_hit_count": len(hits),
        "min_keyword_hits_required": min_hits_required,
        "confidence_label": confidence_label,
        "top_retrieval_score": round(top_score, 3),
        "latency_ms": latency_ms,
        "passed": passed,
        "notes": case.get("notes", ""),
    }


def summarize(results):
    total = len(results)
    passed = sum(1 for r in results if r["passed"])

    by_category = {}
    for r in results:
        cat = r["category"]
        by_category.setdefault(cat, {"total": 0, "passed": 0})
        by_category[cat]["total"] += 1
        if r["passed"]:
            by_category[cat]["passed"] += 1

    avg_latency = round(sum(r["latency_ms"] for r in results) / total, 1) if total else 0

    return {
        "total_cases": total,
        "passed": passed,
        "pass_rate": round(passed / total, 3) if total else 0,
        "avg_latency_ms": avg_latency,
        "by_category": by_category,
    }


def print_summary_table(summary, results):
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Overall: {summary['passed']}/{summary['total_cases']} passed "
          f"({summary['pass_rate'] * 100:.1f}%)")
    print(f"Avg latency: {summary['avg_latency_ms']} ms\n")

    print(f"{'Category':<15} {'Passed':<10} {'Total':<8}")
    print("-" * 35)
    for cat, stats in summary["by_category"].items():
        print(f"{cat:<15} {stats['passed']:<10} {stats['total']:<8}")

    failures = [r for r in results if not r["passed"]]
    if failures:
        print("\n" + "-" * 60)
        print("FAILED CASES")
        print("-" * 60)
        for r in failures:
            print(f"[{r['id']}] ({r['category']}) {r['question']}")
            print(f"   answer: {r['answer'][:120]}")
            print(f"   keyword hits: {r['keyword_hit_count']}/{r['min_keyword_hits_required']} required")
            print(f"   confidence: {r['confidence_label']} (score={r['top_retrieval_score']})")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run the golden evaluation dataset against a PDF.")
    parser.add_argument("--pdf", required=True, help="Path to a PDF file to evaluate against.")
    parser.add_argument("--dataset", default=str(GOLDEN_DATASET_PATH), help="Path to golden dataset JSONL.")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"ERROR: PDF not found at {pdf_path}")
        sys.exit(1)

    print(f"Loading dataset from {args.dataset} ...")
    cases = load_golden_dataset(args.dataset)
    print(f"Loaded {len(cases)} test cases.")

    print(f"Ingesting {pdf_path.name} ...")
    with open(pdf_path, "rb") as f:
        chunks = ingest_pdf(f)
    print(f"Produced {len(chunks)} chunks.")

    print("Building FAISS index ...")
    index, _ = build_index(chunks)

    results = []
    for case in cases:
        print(f"Running [{case['id']}] {case['question']}")
        results.append(run_case(index, chunks, case))

    summary = summarize(results)
    print_summary_table(summary, results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print(f"Full results written to {out_path}")


if __name__ == "__main__":
    main()
