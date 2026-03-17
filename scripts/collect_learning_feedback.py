#!/usr/bin/env python3
"""Collect bootstrap learning feedback and export CSV summaries."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def run_learning_command():
    cmd = (
        "rm -f neurofed.db detail.log && cargo build && timeout 180 target/debug/neuro-fed-node 2>&1 | tee output.log"
        + " && cat detail.log"
    )
    completed = subprocess.run(cmd, shell=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError("Learning command failed")


def parse_detail_log(log_path: Path):
    raw = log_path.read_text()
    entries = raw.split("---\n")
    results = []
    interesting = {"HumanEval/48", "HumanEval/72"}
    for entry in entries:
        if "Bootstrap learning" not in entry:
            continue
        task_id = None
        loss = None
        trajectory = None
        decoded_output = None
        guided_replay = None
        guided_loss = None
        guided_plan = None
        canonical_solution = None
        question_block = None
        for line in entry.splitlines():
            if line.startswith("Question:"):
                question_block = line[len("Question:"):].strip()
                try:
                    question_json = json.loads(question_block)
                    task_id = question_json.get("task_id")
                    canonical_solution = question_json.get("canonical_solution")
                except json.JSONDecodeError:
                    pass
            elif line.strip().startswith("Loss:"):
                loss = float(line.split("Loss:")[-1].strip())
            elif line.strip().startswith("Trajectory:"):
                trajectory = line.split("Trajectory:")[-1].strip()
            elif line.strip().startswith("Decoded Output:"):
                decoded_output = line.split("Decoded Output:")[-1].strip()
            elif line.strip().startswith("Guided Replay:"):
                guided_replay = line.split("Guided Replay:")[-1].strip()
            elif line.strip().startswith("Guided Replay Loss:"):
                guided_loss = line.split("Guided Replay Loss:")[-1].strip()
            elif line.strip().startswith("Guided Replay Plan:"):
                guided_plan = line.split("Guided Replay Plan:")[-1].strip()
        if task_id in interesting and loss and loss > 150:
            results.append({
                "task_id": task_id,
                "loss": str(loss),
                "trajectory": trajectory or "",
                "decoded_output": decoded_output or "",
                "canonical_solution": canonical_solution or "",
                "question": question_block or "",
                "guided_replay": guided_replay or "no",
                "guided_loss": guided_loss or "",
                "guided_plan": guided_plan or "",
            })
    return results


def export_csv(data, output_path: Path):
    import csv

    with output_path.open("w", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "task_id",
                "loss",
                "trajectory",
                "decoded_output",
                "canonical_solution",
                "guided_replay",
                "guided_loss",
                "guided_plan",
            ],
        )
        writer.writeheader()
        for row in data:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Run bootstrap learning collector and export feedback CSV.")
    parser.add_argument("--output", type=Path, default=Path("learning_feedback.csv"))
    parser.add_argument("--log", type=Path, default=Path("detail.log"))
    parser.add_argument("--skip-run", action="store_true", help="Skip rerunning the node and reuse existing logs.")
    args = parser.parse_args()

    if not args.skip_run:
        run_learning_command()

    if not args.log.exists():
        print(f"Log missing: {args.log}", file=sys.stderr)
        sys.exit(1)

    data = parse_detail_log(args.log)
    if not data:
        print("No learning entries found.")
        sys.exit(0)

    export_csv(data, args.output)
    print(f"Exported {len(data)} entries to {args.output}")


if __name__ == "__main__":
    main()
