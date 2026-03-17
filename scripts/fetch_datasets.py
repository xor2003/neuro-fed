#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path


def require_datasets():
    try:
        import datasets  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "Missing python package 'datasets'. Install with:\n"
            "  python -m pip install datasets huggingface_hub\n"
            f"Error: {exc}"
        )


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def alpaca_rows(ds):
    for row in ds:
        instruction = row.get("instruction", "").strip()
        inp = row.get("input", "").strip()
        output = row.get("output", "").strip()
        if not instruction or not output:
            continue
        user = instruction
        if inp:
            user = f"{instruction}\n\nInput: {inp}"
        yield {"type": "assistant", "user": user, "assistant": output}


def dolly_rows(ds):
    for row in ds:
        instruction = row.get("instruction", "").strip()
        context = row.get("context", "").strip()
        response = row.get("response", "").strip()
        if not instruction or not response:
            continue
        user = instruction
        if context:
            user = f"{instruction}\n\nContext: {context}"
        yield {"type": "assistant", "user": user, "assistant": response}


def openassistant_rows(ds):
    prompter_cache = {}
    for row in ds:
        if "messages" in row:
            messages = row.get("messages") or []
            for i in range(1, len(messages)):
                if messages[i].get("role") == "assistant":
                    user = messages[i - 1].get("content") or ""
                    assistant = messages[i].get("content") or ""
                    if user.strip() and assistant.strip():
                        yield {"type": "assistant", "user": user, "assistant": assistant}
            continue

        role = row.get("role")
        text = (row.get("text") or "").strip()
        if not text:
            continue
        if role == "prompter":
            msg_id = row.get("message_id")
            if msg_id:
                prompter_cache[msg_id] = text
        elif role == "assistant":
            parent_id = row.get("parent_id")
            user = prompter_cache.get(parent_id, "")
            if user.strip() and text.strip():
                yield {"type": "assistant", "user": user, "assistant": text}


def gsm8k_rows(ds):
    for row in ds:
        question = (row.get("question") or "").strip()
        answer = (row.get("answer") or "").strip()
        if not question or not answer:
            continue
        parts = answer.split("####")
        solution = parts[-1].strip() if parts else answer.strip()
        reasoning = parts[0].strip() if len(parts) > 1 else ""
        thoughts = [line.strip() for line in reasoning.splitlines() if line.strip()]
        yield {
            "type": "reasoning",
            "problem": question,
            "thoughts": thoughts,
            "solution": solution,
        }


def strategyqa_rows(ds):
    for row in ds:
        question = (row.get("question") or "").strip()
        answer = row.get("answer")
        if answer is True:
            solution = "yes"
        elif answer is False:
            solution = "no"
        else:
            solution = str(answer).strip()
        if not question or not solution:
            continue
        facts = row.get("facts") or row.get("explanation") or []
        if isinstance(facts, str):
            thoughts = [facts.strip()] if facts.strip() else []
        else:
            thoughts = [str(f).strip() for f in facts if str(f).strip()]
        yield {
            "type": "reasoning",
            "problem": question,
            "thoughts": thoughts,
            "solution": solution,
        }


def hotpotqa_rows(ds):
    for row in ds:
        question = (row.get("question") or "").strip()
        answer = (row.get("answer") or "").strip()
        if not question or not answer:
            continue
        supporting = row.get("supporting_facts") or []
        thoughts = []
        if isinstance(supporting, list):
            for item in supporting:
                if isinstance(item, list) and item:
                    thoughts.append(str(item[0]).strip())
                elif isinstance(item, str):
                    thoughts.append(item.strip())
        yield {
            "type": "reasoning",
            "problem": question,
            "thoughts": [t for t in thoughts if t],
            "solution": answer,
        }


def codesearchnet_rows(ds):
    for row in ds:
        code = (row.get("func_code_string") or row.get("code") or "").strip()
        doc = (row.get("func_documentation_string") or row.get("docstring") or "").strip()
        if not code:
            continue
        yield {
            "type": "code",
            "instruction": doc or "Write a function that matches the following code.",
            "code": code,
            "tests": "",
        }


def humaneval_rows(ds):
    for row in ds:
        prompt = (row.get("prompt") or "").strip()
        test = (row.get("test") or "").strip()
        if not prompt:
            continue
        yield {
            "type": "code",
            "instruction": prompt,
            "code": "",
            "tests": test,
        }


def the_stack_rows(ds):
    for row in ds:
        content = (row.get("content") or "").strip()
        if not content:
            continue
        yield {
            "type": "code",
            "instruction": "Use the following code snippet.",
            "code": content,
            "tests": "",
        }


def toolbench_rows(ds):
    for row in ds:
        conversations = row.get("conversations") or []
        if isinstance(conversations, str):
            try:
                conversations = json.loads(conversations)
            except Exception:
                conversations = []

        if conversations:
            user_text = ""
            tool_call = ""
            observation = ""
            next_action = ""

            for msg in conversations:
                if isinstance(msg, str):
                    continue
                role = msg.get("from") or msg.get("role") or ""
                value = (msg.get("value") or msg.get("content") or "").strip()
                if not value:
                    continue
                if role == "user" and not user_text:
                    user_text = value
                elif role in ("assistant", "tool") and not tool_call and user_text:
                    tool_call = value
                elif role in ("function", "tool") and tool_call and not observation:
                    observation = value
                elif role == "assistant":
                    next_action = value

            if not user_text:
                continue
            yield {
                "type": "agent",
                "goal": user_text,
                "tool_call": tool_call,
                "observation": observation,
                "next_action": next_action or tool_call or "COMPLETE",
            }
            continue

        # Fallback schema: query + api list (no conversation transcript).
        goal = (row.get("query") or row.get("instruction") or row.get("question") or "").strip()
        if not goal:
            continue
        api_list = row.get("relevant_apis") or row.get("api_list") or row.get("apis") or []
        tool_call = ""
        observation = ""
        if isinstance(api_list, list) and api_list:
            names = []
            for item in api_list[:3]:
                if isinstance(item, str):
                    names.append(item)
                elif isinstance(item, dict):
                    name = (
                        item.get("api_name")
                        or item.get("name")
                        or item.get("tool_name")
                        or item.get("api")
                    )
                    if name:
                        names.append(str(name))
                    else:
                        names.append(json.dumps(item, ensure_ascii=False))
                else:
                    names.append(str(item))
            tool_call = "CALL_TOOL " + "; ".join(names)
            observation = json.dumps(api_list, ensure_ascii=False)
        elif isinstance(api_list, dict):
            tool_call = "CALL_TOOL " + json.dumps(api_list, ensure_ascii=False)
            observation = json.dumps(api_list, ensure_ascii=False)
        elif isinstance(api_list, str) and api_list.strip():
            tool_call = "CALL_TOOL " + api_list.strip()
            observation = api_list.strip()
        yield {
            "type": "agent",
            "goal": goal,
            "tool_call": tool_call,
            "observation": observation,
            "next_action": "COMPLETE",
        }


def webarena_rows(ds):
    for row in ds:
        intent = (row.get("intent") or "").strip()
        if not intent:
            continue
        start_urls = row.get("start_urls") or []
        start = ", ".join(start_urls) if isinstance(start_urls, list) else str(start_urls)
        eval_info = row.get("eval") or ""
        tool_call = f"NAVIGATE {start}".strip()
        yield {
            "type": "agent",
            "goal": intent,
            "tool_call": tool_call,
            "observation": str(eval_info),
            "next_action": "COMPLETE",
        }


DATASET_MAP = {
    "alpaca": ("tatsu-lab/alpaca", None, "train", alpaca_rows),
    "dolly": ("databricks/databricks-dolly-15k", None, "train", dolly_rows),
    "openassistant": ("OpenAssistant/oasst1", None, "train", openassistant_rows),
    "gsm8k": ("gsm8k", "main", "train", gsm8k_rows),
    "strategyqa": ("tasksource/strategy-qa", None, "train", strategyqa_rows),
    "hotpotqa": ("hotpot_qa", "fullwiki", "train", hotpotqa_rows),
    "codesearchnet": ("code_search_net", None, "train", codesearchnet_rows),
    "humaneval": ("openai_humaneval", None, "test", humaneval_rows),
    "the_stack": ("bigcode/the-stack", None, "train", the_stack_rows),
    "toolbench": ("tuandunghcmut/toolbench-v1", "benchmark", "g1_instruction", toolbench_rows),
    "webarena": ("AmineHA/WebArena-Verified", None, "full", webarena_rows),
}


def main():
    parser = argparse.ArgumentParser(description="Fetch datasets and emit raw JSONL")
    parser.add_argument("--out-dir", default="data/raw", help="Output directory")
    parser.add_argument(
        "--datasets",
        default="alpaca,dolly,openassistant,gsm8k,strategyqa,hotpotqa,codesearchnet,humaneval",
        help="Comma-separated dataset keys",
    )
    parser.add_argument("--limit", type=int, default=5000, help="Max rows per dataset")
    parser.add_argument("--streaming", action="store_true", help="Use HF streaming")
    parser.add_argument("--language", help="Language filter for the-stack/codesearchnet")
    args = parser.parse_args()

    require_datasets()
    from datasets import load_dataset

    out_dir = Path(args.out_dir)
    keys = [k.strip() for k in args.datasets.split(",") if k.strip()]
    for key in keys:
        if key not in DATASET_MAP:
            print(f"Unknown dataset key: {key}")
            continue
        name, config_name, split, row_fn = DATASET_MAP[key]
        dataset_kwargs = {}
        if key in ("the_stack", "codesearchnet") and args.language:
            dataset_kwargs["name"] = args.language
        if config_name:
            dataset_kwargs["name"] = config_name
        ds = load_dataset(name, split=split, streaming=args.streaming, **dataset_kwargs)
        rows = []
        count = 0
        for item in row_fn(ds):
            rows.append(item)
            count += 1
            if count >= args.limit:
                break
        out_path = out_dir / f"{key}.jsonl"
        write_jsonl(out_path, rows)
        print(f"{key}: wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
