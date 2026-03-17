#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
from pathlib import Path


TAG_RE = re.compile(r"<[^>]+>")


def strip_html(text: str) -> str:
    return TAG_RE.sub("", text or "")


def stable_id(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def normalize_record(record, system_prompt, max_chars):
    rtype = record.get("type", "").strip().lower()
    rid = record.get("id") or stable_id(json.dumps(record, sort_keys=True))

    out = {
        "id": rid,
        "system": system_prompt,
        "user": "",
        "thought": [],
        "action": "",
        "observation": "",
        "assistant": "",
        "code": "",
        "tests": "",
        "metadata": {"source_type": rtype},
    }

    if rtype == "assistant":
        out["user"] = record.get("user", "")
        out["assistant"] = record.get("assistant", "")
    elif rtype == "reasoning":
        out["user"] = record.get("problem", "")
        out["thought"] = record.get("thoughts", [])
        out["assistant"] = record.get("solution", "")
        out["metadata"]["task"] = record.get("task")
        out["metadata"]["operation"] = record.get("operation")
    elif rtype == "code":
        out["user"] = record.get("instruction", "")
        out["code"] = record.get("code", "")
        out["tests"] = record.get("tests", "")
        out["assistant"] = record.get("final", "") or out["code"]
    elif rtype == "agent":
        out["user"] = record.get("goal", "")
        out["action"] = record.get("tool_call", "")
        out["observation"] = record.get("observation", "")
        out["assistant"] = record.get("next_action", "")
    else:
        return None

    # Cleanup
    out["user"] = strip_html(out["user"]).strip()
    out["assistant"] = strip_html(out["assistant"]).strip()
    out["action"] = strip_html(out["action"]).strip()
    out["observation"] = strip_html(out["observation"]).strip()
    out["code"] = strip_html(out["code"]).strip()
    out["tests"] = strip_html(out["tests"]).strip()
    out["thought"] = [strip_html(t).strip() for t in out["thought"] if t]

    # Filter oversized rows
    for key in ["user", "assistant", "action", "observation", "code", "tests"]:
        if len(out[key]) > max_chars:
            return None
    for t in out["thought"]:
        if len(t) > max_chars:
            return None

    if not out["user"]:
        return None
    return out


def main():
    parser = argparse.ArgumentParser(description="Normalize multi-type learning datasets")
    parser.add_argument("--input", required=True, help="Comma-separated JSONL paths")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--system", default="You are an intelligent assistant.")
    parser.add_argument("--max-chars", type=int, default=4000)
    args = parser.parse_args()

    inputs = [Path(p.strip()) for p in args.input.split(",") if p.strip()]
    out_path = Path(args.output)

    written = 0
    with out_path.open("w", encoding="utf-8") as out:
        for path in inputs:
            for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                normalized = normalize_record(record, args.system, args.max_chars)
                if not normalized:
                    continue
                out.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                written += 1

    print(f"Wrote {written} rows to {out_path}")


if __name__ == "__main__":
    main()
