#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


ARITH_RE = re.compile(r"\b(\d+)\s*([\+\-\*])\s*(\d+)\b")


def augment_row(row):
    if row.get("thought"):
        return row, False
    if row.get("metadata", {}).get("source_type") != "assistant":
        return row, False

    user = row.get("user", "")
    match = ARITH_RE.search(user)
    if not match:
        return row, False

    a = int(match.group(1))
    op = match.group(2)
    b = int(match.group(3))
    if op == "+":
        result = a + b
        thoughts = [f"add {a} and {b}", f"result = {result}"]
    elif op == "-":
        result = a - b
        thoughts = [f"subtract {b} from {a}", f"result = {result}"]
    else:
        result = a * b
        thoughts = [f"decompose {a} * {b}", f"result = {result}"]

    row["thought"] = thoughts
    meta = row.get("metadata", {})
    meta["augmented_reasoning"] = True
    row["metadata"] = meta
    return row, True


def main():
    parser = argparse.ArgumentParser(description="Add simple reasoning traces to assistant rows")
    parser.add_argument("--input", required=True, help="Input normalized JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL with augmentations")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    augmented = 0
    total = 0
    with output_path.open("w", encoding="utf-8") as out:
        for line in input_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            row, changed = augment_row(row)
            if changed:
                augmented += 1
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Augmented {augmented} / {total} rows -> {output_path}")


if __name__ == "__main__":
    main()
