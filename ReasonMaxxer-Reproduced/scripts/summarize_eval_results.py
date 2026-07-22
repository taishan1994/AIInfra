#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DATASETS = ["minerva_math", "olympiadbench", "math500", "gsm8k", "aime24", "aime25", "amc23"]


def _pass1(path: Path) -> tuple[float, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results", []) or []
    vals: list[float] = []
    for row in rows:
        gens = row.get("generations", []) or []
        if not gens:
            continue
        correct = sum(1 for gen in gens if bool(gen.get("correct", False)))
        vals.append(float(correct / len(gens)))
    if not vals:
        return float("nan"), len(rows)
    return float(sum(vals) / len(vals)), len(rows)


def _infer_dataset(name: str) -> str:
    for dataset in sorted(DATASETS, key=len, reverse=True):
        if name == dataset or name.endswith(f"_{dataset}"):
            return dataset
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize pass@1 from rollout result JSON files.")
    parser.add_argument("--inputs", nargs="+", required=True, help="JSON result files")
    parser.add_argument("--output_csv", default=None)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for raw in args.inputs:
        path = Path(raw)
        p1, n_rows = _pass1(path)
        rows.append(
            {
                "file": str(path),
                "name": path.stem,
                "dataset": _infer_dataset(path.stem),
                "rows": int(n_rows),
                "pass1": float(p1),
            }
        )

    print("| dataset | name | pass@1 | rows |")
    print("|---|---|---:|---:|")
    for row in rows:
        print(f"| {row['dataset']} | {row['name']} | {float(row['pass1']):.4f} | {int(row['rows'])} |")

    if args.output_csv:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset", "name", "file", "rows", "pass1"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
