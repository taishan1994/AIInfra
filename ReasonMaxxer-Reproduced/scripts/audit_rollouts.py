#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load_results(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results", [])
    if not isinstance(rows, list):
        raise ValueError(f"Unsupported rollout file format: {path}")
    return rows


def _print_json(obj: dict[str, Any]) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ReasonMaxxer rollout JSON files for obvious generation-quality issues.")
    parser.add_argument("--input", nargs="+", required=True, help="One or more rollout JSON files")
    parser.add_argument("--long_text_threshold", type=int, default=10000)
    parser.add_argument("--show_examples", type=int, default=5)
    args = parser.parse_args()

    overall_rows = 0
    overall_gens = 0
    overall_correct = 0
    overall_null_extract = 0
    overall_empty = 0
    overall_long = 0
    pass_count_hist = Counter()
    null_examples: list[dict[str, Any]] = []
    long_examples: list[dict[str, Any]] = []

    for item in args.input:
        path = Path(item)
        rows = _load_results(path)
        n_rows = len(rows)
        n_gens = 0
        n_correct = 0
        n_null_extract = 0
        n_empty = 0
        n_long = 0
        lengths: list[int] = []

        for row in rows:
            generations = row.get("generations", [])
            if not isinstance(generations, list):
                continue
            correct_in_row = 0
            for gen in generations:
                text = str(gen.get("text", "") or "")
                extracted = gen.get("extracted_answer")
                is_correct = bool(gen.get("correct", False))
                n_gens += 1
                n_correct += int(is_correct)
                correct_in_row += int(is_correct)
                text_len = len(text)
                lengths.append(text_len)
                if not text.strip():
                    n_empty += 1
                if extracted in {None, ""}:
                    n_null_extract += 1
                    if len(null_examples) < int(args.show_examples):
                        null_examples.append(
                            {
                                "file": path.name,
                                "problem_id": str(row.get("problem_id")),
                                "correct": is_correct,
                                "text_preview": text[:500],
                            }
                        )
                if text_len > int(args.long_text_threshold):
                    n_long += 1
                    if len(long_examples) < int(args.show_examples):
                        long_examples.append(
                            {
                                "file": path.name,
                                "problem_id": str(row.get("problem_id")),
                                "text_len": text_len,
                                "extracted_answer": extracted,
                                "correct": is_correct,
                            }
                        )
            pass_count_hist[correct_in_row] += 1

        summary = {
            "file": path.name,
            "rows": n_rows,
            "generations": n_gens,
            "correct_rate": round(n_correct / max(1, n_gens), 6),
            "null_extract": n_null_extract,
            "empty_text": n_empty,
            "long_text_count": n_long,
            "text_len_mean": round(sum(lengths) / max(1, len(lengths)), 2) if lengths else 0.0,
            "text_len_max": max(lengths) if lengths else 0,
        }
        _print_json(summary)

        overall_rows += n_rows
        overall_gens += n_gens
        overall_correct += n_correct
        overall_null_extract += n_null_extract
        overall_empty += n_empty
        overall_long += n_long

    overall = {
        "rows": overall_rows,
        "generations": overall_gens,
        "correct_rate": round(overall_correct / max(1, overall_gens), 6),
        "null_extract": overall_null_extract,
        "empty_text": overall_empty,
        "long_text_count": overall_long,
        "pass_count_histogram": {str(k): int(v) for k, v in sorted(pass_count_hist.items())},
        "null_extract_examples": null_examples,
        "long_text_examples": long_examples,
    }
    _print_json(overall)


if __name__ == "__main__":
    main()
