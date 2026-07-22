#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _load_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        out: dict[str, dict[str, str]] = {}
        for row in reader:
            key = str(row.get("dataset") or row["name"])
            out[key] = row
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and trained eval summaries.")
    parser.add_argument("--baseline_csv", required=True)
    parser.add_argument("--trained_csv", required=True)
    parser.add_argument("--output_csv", default=None)
    args = parser.parse_args()

    baseline = _load_csv(Path(args.baseline_csv))
    trained = _load_csv(Path(args.trained_csv))
    names = sorted(set(baseline) | set(trained))

    rows: list[dict[str, object]] = []
    for name in names:
        brow = baseline.get(name)
        trow = trained.get(name)
        bpass = float(brow["pass1"]) if brow else float("nan")
        tpass = float(trow["pass1"]) if trow else float("nan")
        brows = int(brow["rows"]) if brow else 0
        trows = int(trow["rows"]) if trow else 0
        rows.append(
            {
                "dataset": name,
                "baseline_pass1": bpass,
                "trained_pass1": tpass,
                "delta_pass1": tpass - bpass,
                "baseline_rows": brows,
                "trained_rows": trows,
            }
        )

    print("| dataset | baseline | trained | delta |")
    print("|---|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['dataset']} | {float(row['baseline_pass1']):.4f} | "
            f"{float(row['trained_pass1']):.4f} | {float(row['delta_pass1']):+.4f} |"
        )

    if args.output_csv:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "dataset",
                    "baseline_pass1",
                    "trained_pass1",
                    "delta_pass1",
                    "baseline_rows",
                    "trained_rows",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
