from __future__ import annotations

import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

from tqdm import tqdm

from agent.llm import is_llama_cpp_provider
from eval.llm_user import (
    DEFAULT_USER_ENABLE_THINKING,
    DEFAULT_USER_GENERATION_TOKEN_LIMIT,
    DEFAULT_USER_THINKING_BUDGET_TOKENS,
    LLMSimulatedUser,
)
from eval.spec_generation import build_user_specs
from eval.tool_call_backends import (
    DEFAULT_AGENT_ENABLE_THINKING,
    DEFAULT_AGENT_GENERATION_TOKEN_LIMIT,
    DEFAULT_AGENT_THINKING_BUDGET_TOKENS,
    backend_metadata,
    make_backend,
)
from eval.tool_call_parsers import clean_tool_call_text
from eval.tool_call_schema import (
    normalize_tool_args,
    parse_selected_resource_ids,
    score_resource_selection_by_need,
    score_tool_calls,
)
from tools.indiana211 import (
    DEFAULT_RESULT_LIMIT,
    execute_search_resources,
    load_resource_index,
    search_resources_tool_schema,
)


DEFAULT_MODEL = "qwen3.6-35b-a3b"
DEFAULT_OPENAI_USER_MODEL = "gpt-4.1-mini"
DEFAULT_OPENROUTER_USER_MODEL = "openai/gpt-4.1-mini"
DEFAULT_USER_BEHAVIORS = ("normal", "rambling", "impatience", "self_contradictory", "unsupported_request")


@dataclass
class EvalConfig:
    specs: Path | None = None
    sample_count: int = 64
    sample_seed: int = 20260611
    sample_progress_every: int = 0
    resources: Path = Path("data/benchmark/filtered_resources_tagged.csv")
    output_dir: Path | None = None
    backend: str = "llama_cpp"
    provider: str = "openai"
    model: str = DEFAULT_MODEL
    adapter: Path | None = None
    max_agent_turns: int = 8
    tool_result_limit: int = DEFAULT_RESULT_LIMIT
    agent_generation_token_limit: int = DEFAULT_AGENT_GENERATION_TOKEN_LIMIT
    agent_enable_thinking: bool = DEFAULT_AGENT_ENABLE_THINKING
    agent_thinking_budget_tokens: int | None = DEFAULT_AGENT_THINKING_BUDGET_TOKENS
    agent_temperature: float = 0.0
    user_provider: str = "llama_cpp"
    user_model: str | None = None
    user_generation_token_limit: int = DEFAULT_USER_GENERATION_TOKEN_LIMIT
    user_enable_thinking: bool = DEFAULT_USER_ENABLE_THINKING
    user_thinking_budget_tokens: int | None = DEFAULT_USER_THINKING_BUDGET_TOKENS
    user_temperature: float = 0.0
    user_behaviors: list[str] = field(default_factory=lambda: list(DEFAULT_USER_BEHAVIORS))
    user_seed: int = 7
    jobs: int = 1
    load_in_4bit: bool = True


def run(config: EvalConfig) -> Path:
    resolve_user_model(config)
    config.output_dir = resolve_output_dir(config)
    resource_index = load_resource_index(config.resources)
    specs = load_or_generate_specs(config, resource_index)
    tool_schema = search_resources_tool_schema(resource_index)
    expanded = expand_specs(specs, selected_user_behaviors(config))
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.output_dir / "args.json", serializable_run_config(config))
    write_json(config.output_dir / "generated_specs.json", specs)
    records_path = config.output_dir / "records.jsonl"
    records_path.write_text("", encoding="utf-8")
    records = evaluate_all(expanded, tool_schema, resource_index, config, records_path)
    summary = summarize(records)
    write_json(config.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return config.output_dir


def evaluate_all(
    expanded: list[tuple[dict[str, Any], str]],
    tool_schema: dict[str, Any],
    resource_index,
    config: EvalConfig,
    records_path: Path,
) -> list[dict[str, Any]]:
    if not expanded:
        return []
    records = []
    if config.backend == "local" or config.jobs == 1:
        backend = make_backend(config)
        with records_path.open("a", encoding="utf-8") as handle:
            for spec, user_behavior in tqdm(expanded, desc="tool-call eval"):
                record = evaluate_conversation(spec, user_behavior, tool_schema, resource_index, backend, config)
                records.append(record)
                write_jsonl_row(handle, record)
        return records
    with ThreadPoolExecutor(max_workers=config.jobs) as executor:
        futures = {
            executor.submit(evaluate_conversation, spec, user_behavior, tool_schema, resource_index, make_backend(config), config): (spec, user_behavior)
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
    resource_index,
    backend,
    config: EvalConfig,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    user = LLMSimulatedUser(
        spec=spec,
        user_behavior=user_behavior,
        provider=config.user_provider,
        model=config.user_model,
        seed=config.user_seed,
        temperature=config.user_temperature,
        max_output_tokens=config.user_generation_token_limit,
        enable_thinking=config.user_enable_thinking,
        thinking_budget_tokens=config.user_thinking_budget_tokens,
    )
    messages.append({"role": "user", "content": user.opening()})
    raw_agent_outputs = []
    predicted_tool_calls = []
    parse_modes = []
    executed_tool_results = []
    final_text = ""
    token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    for _ in range(config.max_agent_turns):
        output = backend.generate(messages, tool_schema)
        raw_agent_outputs.append(output.text)
        add_token_usage(token_usage, output.token_usage or {})
        if output.tool_calls:
            messages.append({"role": "assistant", "content": output.text})
            for call_index, tool_call in enumerate(output.tool_calls, start=1):
                predicted_tool_calls.append(tool_call.arguments)
                parse_modes.append(tool_call.parse_mode)
                result = execute_search_resources(resource_index, tool_call.arguments, limit=config.tool_result_limit)
                executed_tool_results.append(
                    {
                        "call_index": len(predicted_tool_calls),
                        "arguments": normalize_tool_args(tool_call.arguments),
                        "result": result,
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Tool result for search_resources call {call_index}:\n"
                            f"{json.dumps(result, ensure_ascii=False)}\n"
                            "Now choose the best returned resource_id or resource_ids for the original user. "
                            "Do not call search_resources again unless these results are empty or the user has provided new constraints."
                        ),
                    }
                )
            continue
        if predicted_tool_calls:
            final_text = clean_tool_call_text(output.text)
            messages.append({"role": "assistant", "content": final_text})
            break
        assistant_text = clean_tool_call_text(output.text)
        messages.append({"role": "assistant", "content": assistant_text})
        user_reply = user.respond(messages, assistant_text)
        if not user_reply:
            break
        messages.append({"role": "user", "content": user_reply})

    expected = expected_tool_calls(spec)
    expected_resource_ids = expected_resource_ids_from_spec(spec)
    acceptable_resource_ids_by_need = acceptable_resource_ids_for_expected_calls(resource_index, expected, config.tool_result_limit)
    predicted_resource_ids = normalize_selected_resource_ids(
        parse_selected_resource_ids(final_text),
        executed_tool_results,
    )
    tool_score = score_tool_calls(predicted_tool_calls, expected)
    resource_score = score_resource_selection_by_need(predicted_resource_ids, acceptable_resource_ids_by_need)
    score = {
        **tool_score,
        **resource_score,
        "end_to_end_match": bool(tool_score["all_match"]) and bool(resource_score["resource_exact_match"]),
    }
    return {
        "user_spec_id": user_spec_id(spec),
        "user_id": f"{user_spec_id(spec)}__{user_behavior}",
        "user_behavior": user_behavior,
        "backend": backend_metadata(config),
        "case_type": spec.get("case_type") or ("composite" if len(spec.get("needs") or []) > 1 else "single"),
        "constraint_profile": spec.get("constraint_profile") or "all_hard",
        "expected_tool_calls": expected,
        "predicted_tool_calls": [normalize_tool_args(call) for call in predicted_tool_calls],
        "expected_resource_ids": expected_resource_ids,
        "acceptable_resource_ids_by_need": acceptable_resource_ids_by_need,
        "predicted_resource_ids": predicted_resource_ids,
        "final_text": final_text,
        "executed_tool_results": executed_tool_results,
        "parse_mode": parse_modes[0] if parse_modes else "none",
        "parse_modes": parse_modes,
        "score": score,
        "messages": messages,
        "raw_agent_outputs": raw_agent_outputs,
        "user_simulator": "llm",
        "user_simulator_state": {
            "contradiction_area": user.contradiction_area,
            "contradiction_used": user.contradiction_used,
        },
        "user_backend": {
            "provider": config.user_provider,
            "model": config.user_model,
            "temperature": config.user_temperature,
            "generation_token_limit": config.user_generation_token_limit,
            "enable_thinking": config.user_enable_thinking,
            "thinking_budget_tokens": config.user_thinking_budget_tokens,
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


def expected_tool_calls(spec: dict[str, Any]) -> list[dict[str, Any]]:
    needs = spec.get("needs")
    if isinstance(needs, list) and needs:
        return [expected_tool_args_for_need(need) for need in needs if isinstance(need, dict)]
    return [expected_tool_args(spec)]


def expected_tool_args_for_need(need: dict[str, Any]) -> dict[str, Any]:
    args = {
        "service_categories": need.get("service_categories") or [],
        "schedule": need.get("schedule") or {},
        "intake_methods": need.get("intake_methods") or [],
        "available_documents": need.get("available_documents") or [],
        "eligibility": need.get("eligibility") or [],
    }
    location = need.get("location") or {}
    for key in ("counties", "cities", "zipcodes"):
        args[key] = location.get(key) or []
    return normalize_tool_args(args)


def expected_resource_ids_from_spec(spec: dict[str, Any]) -> list[str]:
    resources = spec.get("ground_truth_resources")
    if isinstance(resources, list) and resources:
        return [str(item.get("resource_id")) for item in resources if isinstance(item, dict) and item.get("resource_id")]
    if spec.get("source_resource_id"):
        return [str(spec["source_resource_id"])]
    return []


def acceptable_resource_ids_for_expected_calls(resource_index, expected_calls: list[dict[str, Any]], limit: int) -> list[list[str]]:
    acceptable = []
    for call in expected_calls:
        result = execute_search_resources(resource_index, call, limit=limit)
        acceptable.append([str(item["resource_id"]) for item in result.get("resources", []) if item.get("resource_id")])
    return acceptable


def normalize_selected_resource_ids(predicted_ids: list[str], executed_tool_results: list[dict[str, Any]]) -> list[str]:
    candidates = []
    for item in executed_tool_results:
        result = item.get("result") or {}
        for resource in result.get("resources") or []:
            resource_id = str(resource.get("resource_id") or "")
            if resource_id and resource_id not in candidates:
                candidates.append(resource_id)
    normalized = []
    for predicted in predicted_ids:
        resolved = predicted
        if predicted not in candidates:
            matches = [candidate for candidate in candidates if candidate.startswith(predicted)]
            if len(matches) == 1:
                resolved = matches[0]
        if resolved not in normalized:
            normalized.append(resolved)
    return normalized


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "valid_tool_call",
        "tool_call_count_match",
        "service_match",
        "location_match",
        "schedule_match",
        "intake_match",
        "documents_match",
        "eligibility_match",
        "all_match",
        "resource_exact_match",
        "end_to_end_match",
    ]
    parse_modes = Counter(record["parse_mode"] for record in records)
    return {
        "cases": len(records),
        "overall": aggregate([record["score"] for record in records], keys),
        "by_user_behavior": {
            user_behavior: aggregate([record["score"] for record in records if record["user_behavior"] == user_behavior], keys)
            for user_behavior in sorted({record["user_behavior"] for record in records})
        },
        "by_case_type": {
            case_type: aggregate([record["score"] for record in records if record.get("case_type") == case_type], keys)
            for case_type in sorted({record.get("case_type") for record in records})
        },
        "by_constraint_profile": {
            profile: aggregate([record["score"] for record in records if record.get("constraint_profile") == profile], keys)
            for profile in sorted({record.get("constraint_profile") for record in records})
        },
        "by_case_type_constraint_behavior": {
            f"{case_type}__{profile}__{behavior}": aggregate(
                [
                    record["score"]
                    for record in records
                    if record.get("case_type") == case_type
                    and record.get("constraint_profile") == profile
                    and record.get("user_behavior") == behavior
                ],
                keys,
            )
            for case_type in sorted({record.get("case_type") for record in records})
            for profile in sorted({record.get("constraint_profile") for record in records})
            for behavior in sorted({record.get("user_behavior") for record in records})
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
    result = {f"{key}_rate": sum(bool(score.get(key)) for score in scores) / total for key in keys} | {"n": total}
    result["resource_precision_avg"] = mean(float(score.get("resource_precision") or 0.0) for score in scores)
    result["resource_recall_avg"] = mean(float(score.get("resource_recall") or 0.0) for score in scores)
    result["predicted_tool_call_count_avg"] = mean(float(score.get("predicted_call_count") or 0.0) for score in scores)
    return result


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
        "tool_call_count": "tool_call_count_match",
        "service": "service_match",
        "location": "location_match",
        "schedule": "schedule_match",
        "intake": "intake_match",
        "documents": "documents_match",
        "eligibility": "eligibility_match",
        "resource_selection": "resource_exact_match",
        "end_to_end": "end_to_end_match",
        "not_all_match": "all_match",
    }
    return {
        label: sum(not bool(record.get("score", {}).get(score_key)) for record in records)
        for label, score_key in fields.items()
    } | {"n": len(records)}


def selected_user_behaviors(config: EvalConfig) -> tuple[str, ...]:
    user_behaviors = [*(config.user_behaviors or DEFAULT_USER_BEHAVIORS)]
    return tuple(dict.fromkeys(user_behaviors))


def resolve_user_model(config: EvalConfig) -> None:
    if config.user_model:
        return
    if is_llama_cpp_provider(config.user_provider):
        config.user_model = config.model
    elif config.user_provider == "openrouter":
        config.user_model = DEFAULT_OPENROUTER_USER_MODEL
    else:
        config.user_model = DEFAULT_OPENAI_USER_MODEL


def load_or_generate_specs(config: EvalConfig, resource_index) -> list[dict[str, Any]]:
    if config.specs is not None:
        return read_specs(config.specs)
    return build_user_specs(
        resource_index.resources,
        count=config.sample_count,
        seed=config.sample_seed,
        progress_every=config.sample_progress_every,
    )


def expand_specs(specs: list[dict[str, Any]], user_behaviors: tuple[str, ...]) -> list[tuple[dict[str, Any], str]]:
    return [(spec, user_behavior) for spec in specs for user_behavior in user_behaviors]


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


def resolve_output_dir(config: EvalConfig) -> Path:
    explicit = getattr(config, "output_dir", None)
    if explicit is not None:
        return Path(explicit)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    spec_name = slug(Path(config.specs).stem) if getattr(config, "specs", None) else f"samples{int(config.sample_count)}-seed{int(config.sample_seed)}"
    model_name = slug(str(config.model).split("/")[-1])
    adapter = getattr(config, "adapter", None)
    adapter_name = f"adapter-{slug(Path(adapter).parent.name if Path(adapter).name == 'adapter' else Path(adapter).name)}" if adapter else "base"
    behavior_name = f"behaviors{len(selected_user_behaviors(config))}"
    thinking_name = "think-off"
    if config.agent_enable_thinking:
        budget = config.agent_thinking_budget_tokens
        thinking_name = f"think-{budget}" if budget is not None else "think-unlimited"
    run_name = "__".join([spec_name, slug(config.backend), model_name, adapter_name, behavior_name, thinking_name, timestamp])
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


def serializable_run_config(config: EvalConfig) -> dict[str, Any]:
    values = vars(config).copy()
    for key, value in values.items():
        if isinstance(value, Path):
            values[key] = str(value)
        elif isinstance(value, tuple):
            values[key] = list(value)
    values["agent_generation_token_limit"] = backend_metadata(config)["agent_generation_token_limit"]
    values["user_generation_token_limit"] = config.user_generation_token_limit
    return {"command": sys.argv, "config": values}


def main() -> None:
    raise SystemExit("Use `uv run python main.py` as the evaluation entrypoint.")


if __name__ == "__main__":
    main()
