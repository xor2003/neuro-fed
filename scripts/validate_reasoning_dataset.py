#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path

from prepare_reasoning_dataset import prepare_row, reasoning_score


def load_rows(path: Path):
    with path.open("r", encoding="utf-8") as inp:
        for line in inp:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def pct(part, total):
    if not total:
        return 0.0
    return round((part * 100.0) / total, 2)


def report(path: Path, score_threshold: int, apply_prepare: bool):
    total = 0
    source_types = Counter()
    thought_rows = 0
    reasoning_ready_rows = 0
    code_with_tests = 0
    generic_assistant_rows = 0
    long_rows = 0
    score_hist = Counter()

    for row in load_rows(path):
        total += 1
        source_type = row.get("metadata", {}).get("source_type", "unknown")
        source_types[source_type] += 1

        effective = prepare_row(row) if apply_prepare else row
        if effective:
            score = effective.get("metadata", {}).get("reasoning_score")
            if score is None:
                score = reasoning_score(effective)
            try:
                score = int(score)
            except Exception:
                score = 0
            score_hist[min(score, 10)] += 1
            if score >= score_threshold:
                reasoning_ready_rows += 1
            if effective.get("thought"):
                thought_rows += 1

        if source_type == "code" and (row.get("tests") or "").strip():
            code_with_tests += 1

        if source_type == "assistant":
            has_thought = bool(row.get("thought"))
            user = (row.get("user") or "").lower()
            looks_reasoning = any(
                cue in user
                for cue in (
                    "why",
                    "how",
                    "explain",
                    "reason",
                    "analyze",
                    "analyse",
                    "compare",
                    "derive",
                    "prove",
                    "step by step",
                    "investigate",
                    "plan",
                )
            )
            if not has_thought and not looks_reasoning:
                generic_assistant_rows += 1

        combined_len = sum(
            len(row.get(key, "") or "")
            for key in ("user", "assistant", "action", "observation", "code", "tests")
        ) + sum(len(t or "") for t in (row.get("thought") or []))
        if combined_len > 3500:
            long_rows += 1

    print(f"Dataset: {path}")
    print(f"Total rows: {total}")
    for key in sorted(source_types):
        print(f"source_type.{key}: {source_types[key]} ({pct(source_types[key], total)}%)")
    print(f"thought_rows: {thought_rows} ({pct(thought_rows, total)}%)")
    print(
        f"reasoning_ready_rows(score>={score_threshold}): {reasoning_ready_rows} ({pct(reasoning_ready_rows, total)}%)"
    )
    print(f"code_rows_with_tests: {code_with_tests} ({pct(code_with_tests, total)}%)")
    print(f"generic_assistant_rows: {generic_assistant_rows} ({pct(generic_assistant_rows, total)}%)")
    print(f"oversized_rows_gt_3500_chars: {long_rows} ({pct(long_rows, total)}%)")
    for bucket in range(0, 11):
        print(f"reasoning_score.{bucket}: {score_hist[bucket]}")

    print("Assessment:")
    if reasoning_ready_rows < max(100, total * 0.4):
        print("- reasoning coverage is weak for a reasoning-focused training run")
    else:
        print("- reasoning coverage is substantial enough for a focused run")
    if generic_assistant_rows > total * 0.25:
        print("- too much generic assistant data remains; filter harder before training")
    else:
        print("- generic assistant contamination is controlled")
    if code_with_tests < max(25, total * 0.05):
        print("- code verification signal is still thin")
    else:
        print("- code verification signal is present")


def main():
    parser = argparse.ArgumentParser(description="Validate whether a normalized JSONL dataset is suitable for reasoning training")
    parser.add_argument("--input", required=True, help="Input JSONL")
    parser.add_argument("--score-threshold", type=int, default=3, help="Minimum reasoning score for readiness")
    parser.add_argument(
        "--apply-prepare",
        action="store_true",
        help="Run the same heuristic preparation logic before scoring, for apples-to-apples validation",
    )
    args = parser.parse_args()

    report(Path(args.input), args.score_threshold, args.apply_prepare)


if __name__ == "__main__":
    main()
