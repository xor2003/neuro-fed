#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
from pathlib import Path


ARITH_RE = re.compile(r"\b(\d+)\s*([\+\-\*])\s*(\d+)\b")
REASONING_CUE_RE = re.compile(
    r"\b(why|how|explain|reason|analy[sz]e|compare|derive|prove|step by step|investigate|plan)\b",
    re.IGNORECASE,
)


def stable_id(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def compact_steps(text: str, limit: int = 4):
    steps = []
    for part in re.split(r"[\n;]+", text or ""):
        part = part.strip()
        if part:
            steps.append(part)
        if len(steps) >= limit:
            break
    return steps


def arithmetic_thoughts(user: str):
    match = ARITH_RE.search(user or "")
    if not match:
        return []
    a = int(match.group(1))
    op = match.group(2)
    b = int(match.group(3))
    if op == "+":
        result = a + b
        return [f"identify operands {a} and {b}", f"add {a} and {b}", f"result = {result}"]
    if op == "-":
        result = a - b
        return [f"identify operands {a} and {b}", f"subtract {b} from {a}", f"result = {result}"]
    result = a * b
    return [
        f"identify operands {a} and {b}",
        f"multiply {a} by {b}",
        f"result = {result}",
    ]


def reasoning_score(row):
    score = 0
    source_type = row.get("metadata", {}).get("source_type", "")
    thought = [t for t in (row.get("thought") or []) if str(t).strip()]
    user = row.get("user", "") or ""
    assistant = row.get("assistant", "") or ""
    code = row.get("code", "") or ""
    tests = row.get("tests", "") or ""
    action = row.get("action", "") or ""
    observation = row.get("observation", "") or ""

    if source_type == "reasoning":
        score += 5
    elif source_type == "code":
        score += 3
    elif source_type == "agent":
        score += 3
    elif source_type == "assistant":
        score += 1

    if thought:
        score += min(len(thought), 4)
    if REASONING_CUE_RE.search(user):
        score += 2
    if ARITH_RE.search(user):
        score += 2
    if tests.strip():
        score += 2
    if code.strip():
        score += 1
    if action.strip() or observation.strip():
        score += 1
    if len(assistant) > 40:
        score += 1
    return score


def prepare_row(row):
    out = dict(row)
    out["thought"] = [str(t).strip() for t in (row.get("thought") or []) if str(t).strip()]
    if not out["thought"]:
        out["thought"] = arithmetic_thoughts(out.get("user", ""))

    source_type = out.get("metadata", {}).get("source_type", "")
    keep = False
    if source_type == "reasoning":
        keep = bool(out["thought"]) and bool(out.get("assistant", "").strip())
    elif source_type == "code":
        keep = bool(out.get("user", "").strip()) and (
            bool(out.get("tests", "").strip()) or bool(out.get("code", "").strip())
        )
        if keep and not out["thought"]:
            out["thought"] = compact_steps(
                "inspect the prompt; infer the required behavior; implement the function; verify against tests"
            )
    elif source_type == "agent":
        keep = bool(out.get("user", "").strip()) and (
            bool(out.get("action", "").strip()) or bool(out.get("observation", "").strip())
        )
        if keep and not out["thought"]:
            out["thought"] = compact_steps(
                "understand the goal; choose a tool action; observe the result; decide the next action"
            )
    elif source_type == "assistant":
        cue_match = REASONING_CUE_RE.search(out.get("user", "")) or ARITH_RE.search(out.get("user", ""))
        keep = bool(cue_match)
        if keep and not out["thought"]:
            out["thought"] = compact_steps(
                "extract the task; reason through the answer; state the conclusion"
            )

    if not keep:
        return None

    metadata = dict(out.get("metadata", {}))
    metadata["reasoning_score"] = reasoning_score(out)
    metadata["reasoning_ready"] = True
    metadata["reasoning_profile"] = source_type or "unknown"
    out["metadata"] = metadata
    out["id"] = out.get("id") or stable_id(json.dumps(out, sort_keys=True, ensure_ascii=False))
    return out


def main():
    parser = argparse.ArgumentParser(description="Prepare a reasoning-focused normalized JSONL dataset")
    parser.add_argument("--input", required=True, help="Input normalized JSONL")
    parser.add_argument("--output", required=True, help="Output reasoning-focused JSONL")
    parser.add_argument("--min-score", type=int, default=3, help="Minimum reasoning score to keep")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    seen = set()
    written = 0
    kept_by_type = {}
    with input_path.open("r", encoding="utf-8") as inp, output_path.open("w", encoding="utf-8") as out:
        for line in inp:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            prepared = prepare_row(row)
            if not prepared:
                continue
            if prepared.get("metadata", {}).get("reasoning_score", 0) < args.min_score:
                continue
            dedupe_key = (
                prepared.get("user", ""),
                json.dumps(prepared.get("thought", []), ensure_ascii=False),
                prepared.get("assistant", ""),
                prepared.get("code", ""),
                prepared.get("tests", ""),
                prepared.get("action", ""),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            out.write(json.dumps(prepared, ensure_ascii=False) + "\n")
            written += 1
            source_type = prepared.get("metadata", {}).get("source_type", "unknown")
            kept_by_type[source_type] = kept_by_type.get(source_type, 0) + 1

    print(f"Wrote {written} reasoning-focused rows to {output_path}")
    for key in sorted(kept_by_type):
        print(f"{key}: {kept_by_type[key]}")


if __name__ == "__main__":
    main()
