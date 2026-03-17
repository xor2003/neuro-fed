#!/usr/bin/env python3
import argparse
import json
import random
import re
from pathlib import Path


def load_rows(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def text_len(row):
    total = 0
    for key in ["user", "assistant", "action", "observation", "code", "tests"]:
        total += len(row.get(key, "") or "")
    for t in row.get("thought", []) or []:
        total += len(t or "")
    return total


def main():
    parser = argparse.ArgumentParser(description="Query/Filter normalized learning JSONL")
    parser.add_argument("--input", required=True, help="Input JSONL")
    parser.add_argument("--type", help="Filter by metadata.source_type")
    parser.add_argument("--contains", help="Regex to match any field")
    parser.add_argument("--max-chars", type=int, help="Max combined char length")
    parser.add_argument("--min-score", type=float, help="Min metadata.score if present")
    parser.add_argument("--preset", choices=["alpaca", "dolly", "openassistant"])
    parser.add_argument("--sample", type=int, default=0, help="Sample N rows")
    parser.add_argument("--output", help="Write filtered rows to JSONL")
    parser.add_argument("--stats", action="store_true", help="Print dataset stats")
    args = parser.parse_args()

    rows = list(load_rows(Path(args.input)))
    if args.type:
        rows = [r for r in rows if r.get("metadata", {}).get("source_type") == args.type]
    if args.contains:
        pattern = re.compile(args.contains, re.IGNORECASE)
        def matches(row):
            blob = json.dumps(row, ensure_ascii=False)
            return bool(pattern.search(blob))
        rows = [r for r in rows if matches(r)]
    if args.max_chars:
        rows = [r for r in rows if text_len(r) <= args.max_chars]
    if args.min_score is not None:
        def score_ok(row):
            score = row.get("metadata", {}).get("score")
            if score is None:
                return True
            try:
                return float(score) >= args.min_score
            except ValueError:
                return True
        rows = [r for r in rows if score_ok(r)]

    if args.preset:
        banned_patterns = []
        max_chars = args.max_chars
        if args.preset == "alpaca":
            banned_patterns = [
                r"\broleplay\b",
                r"\bpretend\b",
                r"\bpoem\b",
                r"\bpoetry\b",
                r"\bstory\b",
                r"\bfiction\b",
                r"\blyrics\b",
                r"\bsong\b",
                r"\bcharacter\b",
            ]
            max_chars = max_chars or 1500
        elif args.preset == "dolly":
            max_chars = max_chars or 2000
        elif args.preset == "openassistant":
            banned_patterns = [
                r"\btoxicity\b",
                r"\babuse\b",
                r"\bhate\b",
            ]
            max_chars = max_chars or 2500

        if banned_patterns:
            pattern = re.compile("|".join(banned_patterns), re.IGNORECASE)

            def allowed(row):
                blob = json.dumps(row, ensure_ascii=False)
                return not pattern.search(blob)

            rows = [r for r in rows if allowed(r)]

        if max_chars:
            rows = [r for r in rows if text_len(r) <= max_chars]

    if args.stats:
        counts = {}
        for r in rows:
            t = r.get("metadata", {}).get("source_type", "unknown")
            counts[t] = counts.get(t, 0) + 1
        print("Rows:", len(rows))
        for k, v in sorted(counts.items()):
            print(f"{k}: {v}")

    if args.sample and rows:
        rows = random.sample(rows, min(args.sample, len(rows)))

    if args.output:
        out_path = Path(args.output)
        with out_path.open("w", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows)} rows to {out_path}")
    elif not args.stats:
        for row in rows[:5]:
            print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
