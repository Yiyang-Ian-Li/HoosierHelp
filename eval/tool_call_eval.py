from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.llm_user import LLMSimulatedUser
from eval.tool_call_backends import backend_metadata, make_backend
from eval.tool_call_parsers import clean_tool_call_text
from eval.tool_call_schema import USER_BEHAVIORS, normalize_tool_args, tool_arg_scores
from tools.indiana211 import load_resource_index, search_resources_tool_schema


DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_EXTRA_USER_BEHAVIORS = ("rambling", "impatience", "self_contradictory", "unsupported_request")


def run(args: argparse.Namespace) -> Path:
    args.output_dir = resolve_output_dir(args)
    specs = read_specs(args.specs)
    tool_schema = search_resources_tool_schema(load_resource_index(args.resources))
    expanded = expand_specs(specs, args.limit_conversations, selected_user_behaviors(args))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "args.json", serializable_args(args))
    records_path = args.output_dir / "records.jsonl"
    records_path.write_text("", encoding="utf-8")
    records = evaluate_all(expanded, tool_schema, args, records_path)
    summary = summarize(records)
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return args.output_dir


def evaluate_all(
    expanded: list[tuple[dict[str, Any], str]],
    tool_schema: dict[str, Any],
    args: argparse.Namespace,
    records_path: Path,
) -> list[dict[str, Any]]:
    if not expanded:
        return []
    records = []
    if args.backend == "local" or args.jobs == 1:
        backend = make_backend(args)
        with records_path.open("a", encoding="utf-8") as handle:
            for spec, user_behavior in tqdm(expanded, desc="tool-call eval"):
                record = evaluate_conversation(spec, user_behavior, tool_schema, backend, args)
                records.append(record)
                write_jsonl_row(handle, record)
        return records
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(evaluate_conversation, spec, user_behavior, tool_schema, make_backend(args), args): (spec, user_behavior)
            for spec, user_behavior in expanded
        }
        with records_path.open("a", encoding="utf-8") as handle:
            for future in tqdm(as_completed(futures), total=len(futures), desc="tool-call eval"):
                record = future.result()
                records.append(record)
                write_jsonl_row(handle, record)
    records.sort(key=lambda record: record["user_id"])
    return records


def evaluate_conversation(
    spec: dict[str, Any],
    user_behavior: str,
    tool_schema: dict[str, Any],
    backend,
    args: argparse.Namespace,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    user = LLMSimulatedUser(
        spec=spec,
        user_behavior=user_behavior,
        provider=args.user_provider,
        model=args.user_model,
        seed=args.user_seed,
        temperature=args.user_temperature,
        max_output_tokens=args.user_max_output_tokens,
    )
    messages.append({"role": "user", "content": user.opening()})
    raw_agent_outputs = []
    predicted_tool_call = None
    parse_mode = "none"
    token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    for _ in range(args.max_agent_turns):
        output = backend.generate(messages, tool_schema)
        raw_agent_outputs.append(output.text)
        add_token_usage(token_usage, output.token_usage or {})
        if output.tool_call is not None:
            predicted_tool_call = output.tool_call.arguments
            parse_mode = output.tool_call.parse_mode
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": output.tool_call.name,
                                "arguments": predicted_tool_call,
                            },
                        }
                    ],
                }
            )
            break
        assistant_text = clean_tool_call_text(output.text)
        messages.append({"role": "assistant", "content": assistant_text})
        user_reply = user.respond(messages, assistant_text)
        if not user_reply:
            break
        messages.append({"role": "user", "content": user_reply})

    expected = expected_tool_args(spec)
    score = tool_arg_scores(predicted_tool_call, expected)
    return {
        "user_spec_id": user_spec_id(spec),
        "user_id": f"{user_spec_id(spec)}__{user_behavior}",
        "user_behavior": user_behavior,
        "backend": backend_metadata(args),
        "expected_tool_call": expected,
        "predicted_tool_call": normalize_tool_args(predicted_tool_call or {}) if predicted_tool_call else None,
        "parse_mode": parse_mode,
        "score": score,
        "messages": messages,
        "raw_agent_outputs": raw_agent_outputs,
        "user_simulator": "llm",
        "user_simulator_state": {
            "contradiction_area": user.contradiction_area,
            "contradiction_used": user.contradiction_used,
        },
        "user_backend": {
            "provider": args.user_provider,
            "model": args.user_model,
            "temperature": args.user_temperature,
            "max_output_tokens": args.user_max_output_tokens,
        },
        "token_usage": token_usage,
    }


def expected_tool_args(spec: dict[str, Any]) -> dict[str, Any]:
    args = {
        "service_categories": [spec["service_category"]],
        "schedule": spec.get("schedule") or {},
        "intake_methods": spec.get("intake_methods") or [],
        "available_documents": spec.get("available_documents") or [],
        "eligibility": spec.get("eligibility") or [],
    }
    location = spec.get("location") or {}
    for key in ("counties", "cities", "zipcodes"):
        args[key] = location.get(key) or []
    return normalize_tool_args(args)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "valid_tool_call",
        "service_match",
        "location_match",
        "schedule_match",
        "intake_match",
        "documents_match",
        "eligibility_match",
        "all_match",
    ]
    parse_modes = Counter(record["parse_mode"] for record in records)
    return {
        "cases": len(records),
        "overall": aggregate([record["score"] for record in records], keys),
        "by_user_behavior": {
            user_behavior: aggregate([record["score"] for record in records if record["user_behavior"] == user_behavior], keys)
            for user_behavior in sorted({record["user_behavior"] for record in records})
        },
        "parse_modes": dict(sorted(parse_modes.items())),
        "parse_none_rate": parse_modes.get("none", 0) / len(records) if records else 0.0,
        "score_by_parse_mode": {
            mode: aggregate([record["score"] for record in records if record["parse_mode"] == mode], keys)
            for mode in sorted(parse_modes)
        },
        "turn_stats": turn_stats(records),
        "token_usage": token_usage_summary(records),
        "failure_counts": failure_counts(records),
        "by_user_behavior_failure_counts": {
            user_behavior: failure_counts([record for record in records if record["user_behavior"] == user_behavior])
            for user_behavior in sorted({record["user_behavior"] for record in records})
        },
    }


def aggregate(scores: list[dict[str, bool]], keys: list[str]) -> dict[str, float]:
    if not scores:
        return {"n": 0}
    total = len(scores)
    return {f"{key}_rate": sum(bool(score.get(key)) for score in scores) / total for key in keys} | {"n": total}


def turn_stats(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {"agent_turns_avg": 0.0, "agent_turns_median": 0.0, "agent_turns_max": 0.0, "user_messages_avg": 0.0, "user_messages_median": 0.0, "user_messages_max": 0.0}
    agent_turns = [len(record.get("raw_agent_outputs") or []) for record in records]
    user_messages = [
        sum(1 for message in record.get("messages", []) if message.get("role") == "user")
        for record in records
    ]
    return {
        "agent_turns_avg": mean(agent_turns),
        "agent_turns_median": median(agent_turns),
        "agent_turns_max": max(agent_turns),
        "user_messages_avg": mean(user_messages),
        "user_messages_median": median(user_messages),
        "user_messages_max": max(user_messages),
    }


def token_usage_summary(records: list[dict[str, Any]]) -> dict[str, int | float]:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for record in records:
        usage = record.get("token_usage") or {}
        for key in totals:
            totals[key] += int(usage.get(key) or 0)
    count = len(records)
    return {
        **totals,
        "avg_input_tokens": totals["input_tokens"] / count if count else 0.0,
        "avg_output_tokens": totals["output_tokens"] / count if count else 0.0,
        "avg_total_tokens": totals["total_tokens"] / count if count else 0.0,
    }


def failure_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    fields = {
        "invalid_tool_call": "valid_tool_call",
        "service": "service_match",
        "location": "location_match",
        "schedule": "schedule_match",
        "intake": "intake_match",
        "documents": "documents_match",
        "eligibility": "eligibility_match",
        "not_all_match": "all_match",
    }
    return {
        label: sum(not bool(record.get("score", {}).get(score_key)) for record in records)
        for label, score_key in fields.items()
    } | {"n": len(records)}
    return {
        field: sum(not bool(record.get("score", {}).get(field)) for record in records)
        for field in fields
    } | {"n": len(records)}


def selected_user_behaviors(args: argparse.Namespace) -> tuple[str, ...]:
    user_behaviors = ["normal", *(args.user_behaviors or DEFAULT_EXTRA_USER_BEHAVIORS)]
    return tuple(dict.fromkeys(user_behaviors))


def expand_specs(specs: list[dict[str, Any]], limit: int, user_behaviors: tuple[str, ...]) -> list[tuple[dict[str, Any], str]]:
    rows = [(spec, user_behavior) for spec in specs for user_behavior in user_behaviors]
    return rows[:limit] if limit else rows


def user_spec_id(spec: dict[str, Any]) -> str:
    return str(spec["user_spec_id"])


def add_token_usage(total: dict[str, int], item: dict[str, int]) -> None:
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        total[key] = total.get(key, 0) + int(item.get(key, 0) or 0)


def read_specs(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            write_jsonl_row(handle, row)


def write_jsonl_row(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    explicit = getattr(args, "output_dir", None)
    if explicit is not None:
        return Path(explicit)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    spec_name = slug(Path(args.specs).stem)
    model_name = slug(str(args.model).split("/")[-1])
    adapter = getattr(args, "adapter", None)
    adapter_name = f"adapter-{slug(Path(adapter).parent.name if Path(adapter).name == 'adapter' else Path(adapter).name)}" if adapter else "base"
    limit = int(getattr(args, "limit_conversations", 0) or 0)
    limit_name = f"n{limit}" if limit else "all"
    run_name = "__".join([spec_name, slug(args.backend), model_name, adapter_name, limit_name, timestamp])
    return Path("experiments/tool_call_eval") / run_name


def slug(value: str) -> str:
    chars = []
    for char in value.lower():
        if char.isalnum():
            chars.append(char)
        elif char in {"-", "_", "."}:
            chars.append(char)
        else:
            chars.append("-")
    text = "".join(chars).strip("-._")
    while "--" in text:
        text = text.replace("--", "-")
    return text or "run"


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    values = vars(args).copy()
    for key, value in values.items():
        if isinstance(value, Path):
            values[key] = str(value)
        elif isinstance(value, tuple):
            values[key] = list(value)
    return {"command": sys.argv, "args": values}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate search_resources tool-call argument correctness.")
    parser.add_argument("--specs", type=Path, default=Path("data/benchmark/case_specs.json"))
    parser.add_argument("--resources", type=Path, default=Path("data/benchmark/filtered_resources_tagged.csv"))
    parser.add_argument("--output-dir", type=Path, help="Directory for args.json, records.jsonl, and summary.json. Defaults to a timestamped directory under experiments/tool_call_eval/.")
    parser.add_argument("--backend", choices=["local", "responses"], default="local")
    parser.add_argument("--provider", choices=["openai", "openrouter"], default="openai")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--limit-conversations", type=int, default=0)
    parser.add_argument("--max-agent-turns", type=int, default=8)
    parser.add_argument("--agent-max-new-tokens", type=int, default=256)
    parser.add_argument("--agent-temperature", type=float, default=0.0)
    parser.add_argument("--user-provider", choices=["openai", "openrouter"], default="openai")
    parser.add_argument("--user-model", default="gpt-4.1-mini")
    parser.add_argument("--user-temperature", type=float, default=0.7)
    parser.add_argument("--user-max-output-tokens", type=int, default=180)
    parser.add_argument("--user-behaviors", nargs="+", choices=[behavior for behavior in USER_BEHAVIORS if behavior != "normal"], default=list(DEFAULT_EXTRA_USER_BEHAVIORS))
    parser.add_argument("--user-seed", type=int, default=7)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    return parser.parse_args(argv)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
