#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from prepare_reasoning_dataset import prepare_row, reasoning_score, stable_id


SYSTEM_PROMPT = """You are preparing training data for a reasoning-focused local assistant.
Return strict JSON only.

Decide whether the row should be kept for reasoning training.
Prefer rows that teach step-by-step reasoning, planning, code synthesis with tests, tool-use reasoning, or evidence-based investigation.
Reject rows that are generic chat, roleplay, vague opinion-only text, or unstructured filler.

Output JSON schema:
{
  "include": true,
  "reasoning_score": 0,
  "reasoning_profile": "reasoning|code|agent|assistant",
  "thought": ["short step", "short step"],
  "assistant": "optional cleaned answer",
  "notes": "short rationale"
}

Rules:
- JSON only, no markdown.
- reasoning_score must be 0..10.
- thought should have at most 6 short steps.
- Preserve original meaning. Do not invent facts.
- For code rows, thoughts should reflect inspect -> implement -> verify.
- For agent rows, thoughts should reflect goal -> action -> observation -> next action.
- For reasoning rows, keep solution concise and preserve answer correctness.
"""


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


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def post_chat_completion(base_url: str, api_key: str, model: str, system_prompt: str, user_prompt: str, timeout: float):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    return json.loads(content)


def clamp_score(value):
    try:
        score = int(value)
    except Exception:
        return 0
    return max(0, min(10, score))


def normalize_llm_output(obj):
    thought = obj.get("thought") or []
    if isinstance(thought, str):
        thought = [thought]
    thought = [str(item).strip() for item in thought if str(item).strip()][:6]
    return {
        "include": bool(obj.get("include")),
        "reasoning_score": clamp_score(obj.get("reasoning_score", 0)),
        "reasoning_profile": str(obj.get("reasoning_profile") or "assistant").strip() or "assistant",
        "thought": thought,
        "assistant": str(obj.get("assistant") or "").strip(),
        "notes": str(obj.get("notes") or "").strip(),
    }


def build_user_prompt(row):
    return json.dumps(
        {
            "id": row.get("id"),
            "metadata": row.get("metadata", {}),
            "user": row.get("user", ""),
            "thought": row.get("thought", []),
            "assistant": row.get("assistant", ""),
            "code": row.get("code", ""),
            "tests": row.get("tests", ""),
            "action": row.get("action", ""),
            "observation": row.get("observation", ""),
        },
        ensure_ascii=False,
    )


def merge_llm_decision(row, decision, overwrite_thought, overwrite_assistant):
    out = dict(row)
    metadata = dict(out.get("metadata", {}))
    metadata["reasoning_score"] = decision["reasoning_score"]
    metadata["reasoning_ready"] = decision["include"]
    metadata["reasoning_profile"] = decision["reasoning_profile"]
    metadata["llm_reasoning_notes"] = decision["notes"]
    metadata["llm_preprocessed"] = True
    out["metadata"] = metadata

    if decision["thought"] and (overwrite_thought or not out.get("thought")):
        out["thought"] = decision["thought"]
    if decision["assistant"] and (overwrite_assistant or not out.get("assistant", "").strip()):
        out["assistant"] = decision["assistant"]
    out["id"] = out.get("id") or stable_id(json.dumps(out, sort_keys=True, ensure_ascii=False))
    return out


def main():
    parser = argparse.ArgumentParser(description="Use an OpenAI-compatible LLM to prepare a reasoning-focused dataset")
    parser.add_argument("--input", required=True, help="Input normalized JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "").strip(), help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "").strip(), help="API key")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "").strip() or "gpt-4o-mini", help="Remote model name")
    parser.add_argument("--max-rows", type=int, default=0, help="Only process the first N heuristic candidate rows")
    parser.add_argument("--min-heuristic-score", type=int, default=2, help="Only send rows above this heuristic score to the LLM")
    parser.add_argument("--overwrite-thought", action="store_true", help="Allow LLM to replace existing thought traces")
    parser.add_argument("--overwrite-assistant", action="store_true", help="Allow LLM to replace existing assistant text")
    parser.add_argument("--resume", action="store_true", help="Start from existing output rows and skip matching ids")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds")
    parser.add_argument("--sleep-ms", type=int, default=0, help="Sleep between requests")
    parser.add_argument("--dry-run", action="store_true", help="Do not call the LLM; keep heuristic-prepared rows only")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")
    if not args.dry_run and (not args.base_url or not args.api_key):
        raise SystemExit("Set --base-url and --api-key, or use OPENAI_BASE_URL and OPENAI_API_KEY.")

    existing = {}
    if args.resume and output_path.exists():
        for row in load_rows(output_path):
            row_id = row.get("id")
            if row_id:
                existing[row_id] = row

    written = []
    kept = 0
    sent = 0
    skipped_existing = 0
    failed = 0

    for row in load_rows(input_path):
        prepared = prepare_row(row)
        if not prepared:
            continue
        heuristic = reasoning_score(prepared)
        if heuristic < args.min_heuristic_score:
            continue

        row_id = prepared.get("id") or stable_id(json.dumps(prepared, sort_keys=True, ensure_ascii=False))
        prepared["id"] = row_id
        if row_id in existing:
            written.append(existing[row_id])
            skipped_existing += 1
            continue

        if args.max_rows and sent >= args.max_rows:
            break

        if args.dry_run:
            metadata = dict(prepared.get("metadata", {}))
            metadata["llm_preprocessed"] = False
            metadata["reasoning_score"] = heuristic
            metadata["reasoning_ready"] = True
            prepared["metadata"] = metadata
            written.append(prepared)
            kept += 1
            sent += 1
            continue

        user_prompt = build_user_prompt(prepared)
        try:
            raw_decision = post_chat_completion(
                args.base_url,
                args.api_key,
                args.model,
                SYSTEM_PROMPT,
                user_prompt,
                args.timeout,
            )
            decision = normalize_llm_output(raw_decision)
            sent += 1
            if decision["include"]:
                merged = merge_llm_decision(
                    prepared,
                    decision,
                    overwrite_thought=args.overwrite_thought,
                    overwrite_assistant=args.overwrite_assistant,
                )
                written.append(merged)
                kept += 1
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError) as exc:
            failed += 1
            fallback = dict(prepared)
            metadata = dict(fallback.get("metadata", {}))
            metadata["llm_preprocessed"] = False
            metadata["llm_preprocess_error"] = str(exc)
            metadata["reasoning_score"] = heuristic
            fallback["metadata"] = metadata
            written.append(fallback)
        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    write_jsonl(output_path, written)
    print(f"Wrote {len(written)} rows to {output_path}")
    print(f"kept={kept} sent={sent} skipped_existing={skipped_existing} failed={failed}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
