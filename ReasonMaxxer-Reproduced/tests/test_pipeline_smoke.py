from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from reasonmaxxer.answer_extraction import extract_boxed_answer, extract_gsm8k_answer, extract_final_number
from reasonmaxxer.answer_verification import answers_match


ROOT = Path(__file__).resolve().parents[1]


class ReasonMaxxerSmokeTests(unittest.TestCase):
    def test_answer_helpers(self) -> None:
        self.assertEqual(extract_boxed_answer(r"Work... \boxed{42}"), "42")
        self.assertEqual(extract_gsm8k_answer("some steps\n#### 1,234"), "1234")
        self.assertEqual(extract_final_number("therefore, the answer is 12.0"), "12")
        self.assertTrue(answers_match("3/4", "0.75"))
        self.assertTrue(answers_match(r"\boxed{5}", "5"))

    def test_sampling_selection_and_prepare(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            parquet_path = tmp / "train.parquet"
            pd.DataFrame(
                [
                    {"question": "q1", "level": 3, "gt_answer": "1"},
                    {"question": "q2", "level": 3, "gt_answer": "2"},
                    {"question": "q3", "level": 4, "gt_answer": "3"},
                    {"question": "q4", "level": 4, "gt_answer": "4"},
                    {"question": "q5", "level": 5, "gt_answer": "5"},
                    {"question": "q6", "level": 5, "gt_answer": "6"},
                ]
            ).to_parquet(parquet_path)

            sampled_dir = tmp / "sampled"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/sample_simplerl_records.py"),
                    "--train_parquet",
                    str(parquet_path),
                    "--levels",
                    "3,4,5",
                    "--per_level",
                    "1",
                    "--shard_size",
                    "1",
                    "--merged_shard_size",
                    "2",
                    "--seed",
                    "42",
                    "--out_dir",
                    str(sampled_dir),
                ],
                check=True,
                cwd=ROOT,
            )
            manifest = json.loads((sampled_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["files"]["merged"]["num_records"], 3)

            rollouts = {
                "rollouts": [
                    {
                        "problem_id": "p1",
                        "problem_text": "prob1",
                        "ground_truth": "1",
                        "category": "math",
                        "gen_index": 0,
                        "correct": True,
                        "truncated": False,
                        "prompt_length": 2,
                        "num_completion_tokens": 2,
                        "input_ids": [1, 2, 3, 4],
                        "entropies": [1.8, 0.2],
                    },
                    {
                        "problem_id": "p1",
                        "problem_text": "prob1",
                        "ground_truth": "1",
                        "category": "math",
                        "gen_index": 1,
                        "correct": False,
                        "truncated": False,
                        "prompt_length": 2,
                        "num_completion_tokens": 2,
                        "input_ids": [1, 2, 4, 5],
                        "entropies": [1.9, 0.1],
                    },
                    {
                        "problem_id": "p2",
                        "problem_text": "prob2",
                        "ground_truth": "2",
                        "category": "math",
                        "gen_index": 0,
                        "correct": True,
                        "truncated": False,
                        "prompt_length": 2,
                        "num_completion_tokens": 2,
                        "input_ids": [1, 2, 6, 7],
                        "entropies": [1.7, 0.1],
                    },
                    {
                        "problem_id": "p2",
                        "problem_text": "prob2",
                        "ground_truth": "2",
                        "category": "math",
                        "gen_index": 1,
                        "correct": False,
                        "truncated": True,
                        "prompt_length": 2,
                        "num_completion_tokens": 3,
                        "input_ids": [1, 2, 6, 8, 9],
                        "entropies": [1.6, 0.4, 0.2],
                    },
                ]
            }
            rollouts_path = tmp / "rollouts.json"
            rollouts_path.write_text(json.dumps(rollouts, indent=2) + "\n", encoding="utf-8")

            selection_dir = tmp / "selection"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/select_mid_pool.py"),
                    "--input",
                    str(rollouts_path),
                    "--output_dir",
                    str(selection_dir),
                    "--max_target_problems",
                    "2",
                    "--require_both_signs",
                    "--trim_fraction",
                    "0.75",
                ],
                check=True,
                cwd=ROOT,
            )
            selected_ids = json.loads((selection_dir / "selected_problem_ids.json").read_text(encoding="utf-8"))
            self.assertEqual(len(selected_ids["target_problem_ids"]), 2)

            train_dir = tmp / "train"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/prepare_training_data.py"),
                    "--rollouts_file",
                    str(selection_dir / "selected_rollouts_trim80_entropy.json"),
                    "--tau_pos",
                    "1.0",
                    "--tau_neg",
                    "1.0",
                    "--min_pass",
                    "0.0",
                    "--max_pass",
                    "1.0",
                    "--max_target_problems",
                    "10",
                    "--target_ids_output",
                    str(train_dir / "target_ids.json"),
                    "--processed_output",
                    str(train_dir / "processed.json"),
                    "--training_examples_output",
                    str(train_dir / "examples.json"),
                    "--stats_output",
                    str(train_dir / "stats.json"),
                    "--selected_records_output",
                    str(train_dir / "records.json"),
                ],
                check=True,
                cwd=ROOT,
            )
            processed = json.loads((train_dir / "processed.json").read_text(encoding="utf-8"))
            self.assertGreater(len(processed["rollouts"]), 0)
            self.assertTrue(any(row["decision_positions"] for row in processed["rollouts"]))


if __name__ == "__main__":
    unittest.main()
