#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _load_ids(path: Path, keys: list[str]) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in keys:
            values = payload.get(key)
            if isinstance(values, list):
                return [str(x) for x in values]
    raise ValueError(f"Could not find any of keys={keys} in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic holdout split from sampled IDs excluding train-target IDs.")
    parser.add_argument("--sampled_ids_file", required=True, help="JSON containing the full sampled problem_ids list")
    parser.add_argument(
        "--selected_target_ids_file",
        required=True,
        help="JSON containing selected training target_problem_ids",
    )
    parser.add_argument("--holdout_size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    sampled_ids = _load_ids(Path(args.sampled_ids_file), ["problem_ids", "holdout_ids", "train_ids"])
    selected_ids = set(_load_ids(Path(args.selected_target_ids_file), ["target_problem_ids", "problem_ids"]))

    candidates = [pid for pid in sampled_ids if pid not in selected_ids]
    if len(candidates) < int(args.holdout_size):
        raise ValueError(
            f"Not enough candidate holdout IDs after exclusion: have {len(candidates)}, need {int(args.holdout_size)}"
        )

    rng = random.Random(int(args.seed))
    chosen = sorted(rng.sample(candidates, k=int(args.holdout_size)))

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sampled_ids_file": str(args.sampled_ids_file),
        "selected_target_ids_file": str(args.selected_target_ids_file),
        "holdout_size": int(args.holdout_size),
        "seed": int(args.seed),
        "problem_ids": chosen,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {out_path}")
    print(f"holdout_size = {len(chosen)}")


if __name__ == "__main__":
    main()
