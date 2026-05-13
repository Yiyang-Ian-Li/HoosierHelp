from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import aggregate, aggregate_breakdown, parse_tool_output, score_case


def main() -> None:
    args = parse_args()
    analyze_run_dir(args.run_dir)


def analyze_run_dir(run_dir: Path) -> dict:
    case_paths = [
        path
        for path in sorted(case_dir(run_dir).glob("*.json"))
        if path.name != "summary.json"
    ]
    cases = [json.loads(path.read_text(encoding="utf-8")) for path in case_paths]
    scores = [
        score_case(
            case["card"],
            case["ground_truth"],
            case["transcript"],
            case["response"],
        )
        for case in cases
    ]
    details = analyze_cases(cases, scores)
    summary_path = run_dir / "summary.json"
    run_metadata = load_run_metadata(run_dir)
    breakdown = aggregate_breakdown(cases, scores)
    summary = {
        "run": {
            **run_metadata,
            "completed_cases": sum(1 for case in cases if case.get("completed")),
            "stop_reasons": {
                reason: sum(1 for case in cases if case.get("stop_reason") == reason)
                for reason in sorted({case.get("stop_reason") for case in cases})
            },
        },
        "token_usage": aggregate_token_usage(cases),
        "metrics": aggregate(scores),
        "by_case_type": breakdown["by_case_type"],
        "by_trait": breakdown["by_trait"],
        "simulated_user_diagnostics": aggregate_simulated_user_diagnostics(cases),
        "analysis": details,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Analyzed {len(cases)} cases")
    print(f"Wrote {summary_path}")
    return summary


def load_run_metadata(run_dir: Path) -> dict:
    run_path = run_dir / "run.json"
    if run_path.exists():
        return json.loads(run_path.read_text(encoding="utf-8"))
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {}
    old = json.loads(summary_path.read_text(encoding="utf-8"))
    if isinstance(old.get("run"), dict):
        return old["run"]
    keys = [
        "provider",
        "agent_type",
        "agent_model",
        "user_model",
        "case_type",
        "users",
        "index_path",
        "max_turns",
        "jobs",
    ]
    return {key: old[key] for key in keys if key in old}


def case_dir(run_dir: Path) -> Path:
    conversations = run_dir / "conversations"
    return conversations if conversations.exists() else run_dir


def analyze_cases(cases: list[dict], scores: list[dict]) -> dict:
    total_function_calls = 0
    empty_outputs = 0
    filter_counts = Counter()
    service_category_counts = Counter()
    no_tool_case_count = 0
    general_advice_after_empty_count = 0
    failure_counts = Counter()
    retrieval_field_errors = new_retrieval_field_error_stats()
    score_by_user = {score["user_id"]: score for score in scores}
    for case in cases:
        user_id = case["card"]["user_id"]
        score = score_by_user[user_id]
        if not score["id_hit"]:
            if not score.get("final_json_valid"):
                failure_counts["invalid_final_json"] += 1
            if not score["retrieval_hit"]:
                failure_counts["retrieval_miss"] += 1
                update_retrieval_field_errors(retrieval_field_errors, case, score)
            elif score.get("id_hit") and (
                not score.get("intake_hit") or not score.get("document_hit")
            ):
                failure_counts["detail_miss"] += 1
            else:
                failure_counts["retrieved_but_not_recommended_or_id"] += 1
                recommended_ids = score.get("recommended_resource_ids", [])
                ground_truth_ids = case["ground_truth"].get("ground_truth_resource_ids", [])
                missing_ground_truth_ids = set(ground_truth_ids) - set(recommended_ids)
                wrong_recommended_ids = set(recommended_ids) - set(ground_truth_ids)
                if any(
                    ground_truth_id.startswith(recommended_id) or recommended_id.startswith(ground_truth_id)
                    for recommended_id in wrong_recommended_ids
                    for ground_truth_id in missing_ground_truth_ids
                    if recommended_id != ground_truth_id
                ):
                    failure_counts["possible_id_format"] += 1
        response = case["response"]
        input_items = response.get("input", [])
        outputs = [item for item in input_items if item.get("type") == "function_call_output"]
        executed_tool_calls = response.get("tool_calls", [])
        if not executed_tool_calls:
            no_tool_case_count += 1
        total_function_calls += len(executed_tool_calls)
        for item in executed_tool_calls:
            args = item.get("arguments") or {}
            for key, value in args.items():
                if value:
                    filter_counts[key] += 1
            for service_category in args.get("service_categories", []) or []:
                service_category_counts[service_category] += 1
        for output in outputs:
            result = parse_tool_output(output.get("output", ""))
            if not result.get("resources"):
                empty_outputs += 1
        if outputs and all(not parse_tool_output(output.get("output", "")).get("resources") for output in outputs):
            final_text = response.get("output_text", "").lower()
            if any(term in final_text for term in ["consider", "local", "general", "you might", "not able to find"]):
                general_advice_after_empty_count += 1
    return {
        "total_function_calls": total_function_calls,
        "empty_tool_outputs": empty_outputs,
        "empty_tool_output_rate": empty_outputs / total_function_calls if total_function_calls else 0,
        "no_tool_case_count": no_tool_case_count,
        "common_filters": filter_counts.most_common(),
        "common_requested_service_categories": service_category_counts.most_common(20),
        "general_advice_after_empty_count": general_advice_after_empty_count,
        "failure_counts": {
            "retrieval_miss": failure_counts["retrieval_miss"],
            "retrieved_but_not_recommended_or_id": failure_counts["retrieved_but_not_recommended_or_id"],
            "detail_miss": failure_counts["detail_miss"],
            "invalid_final_json": failure_counts["invalid_final_json"],
            "possible_id_format": failure_counts["possible_id_format"],
        },
        "retrieval_miss_field_errors": finalize_retrieval_field_error_stats(retrieval_field_errors),
    }


def aggregate_simulated_user_diagnostics(cases: list[dict]) -> dict:
    trait_counts = Counter()
    for case in cases:
        diagnostics = case.get("simulated_user_diagnostics") or {}
        trait_counts.update(diagnostics.get("traits") or [])
    return {
        "trait_counts": dict(trait_counts),
    }


def new_retrieval_field_error_stats() -> dict:
    return {
        "single": new_retrieval_field_error_group(),
        "composite": new_retrieval_field_error_group(),
    }


def new_retrieval_field_error_group() -> dict:
    return {
        "target_needs": 0,
        "service_errors": 0,
        "schedule_errors": 0,
        "location_errors": 0,
        "all_fields_errors": 0,
        "no_tool_call_errors": 0,
    }


def update_retrieval_field_errors(stats: dict, case: dict, score: dict) -> None:
    case_type = str(case.get("card", {}).get("case_type") or "unknown")
    if case_type not in stats:
        stats[case_type] = new_retrieval_field_error_group()
    group = stats[case_type]
    calls = case.get("response", {}).get("tool_calls", []) or []
    retrieved_ids = set(score.get("retrieved_resource_ids") or [])
    location_requirement = case.get("ground_truth", {}).get("location_requirement") or case.get("card", {}).get("location_requirement") or {}
    needs = case.get("ground_truth", {}).get("needs") or case.get("card", {}).get("case_spec", {}).get("needs") or []
    for need in needs:
        gt_id = need.get("ground_truth_resource_id")
        if gt_id in retrieved_ids:
            continue
        group["target_needs"] += 1
        if not calls:
            group["no_tool_call_errors"] += 1
            group["service_errors"] += 1
            group["schedule_errors"] += 1
            group["location_errors"] += 1
            group["all_fields_errors"] += 1
            continue
        service_ok = any(tool_service_matches(call.get("arguments") or {}, need) for call in calls)
        schedule_ok = any(tool_schedule_matches(call.get("arguments") or {}, need) for call in calls)
        location_ok = any(tool_location_matches(call.get("arguments") or {}, location_requirement) for call in calls)
        all_fields_ok = any(
            tool_service_matches(call.get("arguments") or {}, need)
            and tool_schedule_matches(call.get("arguments") or {}, need)
            and tool_location_matches(call.get("arguments") or {}, location_requirement)
            for call in calls
        )
        if not service_ok:
            group["service_errors"] += 1
        if not schedule_ok:
            group["schedule_errors"] += 1
        if not location_ok:
            group["location_errors"] += 1
        if not all_fields_ok:
            group["all_fields_errors"] += 1


def finalize_retrieval_field_error_stats(stats: dict) -> dict:
    finalized = {}
    for case_type, group in stats.items():
        total = group["target_needs"]
        finalized[case_type] = {
            **group,
            "service_error_rate": group["service_errors"] / total if total else 0,
            "schedule_error_rate": group["schedule_errors"] / total if total else 0,
            "location_error_rate": group["location_errors"] / total if total else 0,
            "all_fields_error_rate": group["all_fields_errors"] / total if total else 0,
            "no_tool_call_error_rate": group["no_tool_call_errors"] / total if total else 0,
        }
    return finalized


def tool_service_matches(args: dict, need: dict) -> bool:
    return bool(norm_set(args.get("service_categories")) & norm_set(need.get("service_categories")))


def tool_schedule_matches(args: dict, need: dict) -> bool:
    return normalize_schedule(args.get("schedule") or {}) == normalize_schedule(need.get("schedule") or {})


def tool_location_matches(args: dict, location_requirement: dict) -> bool:
    for key in ("counties", "cities", "zipcodes"):
        expected = norm_set(location_requirement.get(key))
        if expected:
            return bool(expected & norm_set(args.get(key)))
    return False


def normalize_schedule(schedule: dict) -> dict:
    if schedule.get("requires_24_hours") is True:
        return {"requires_24_hours": True}
    day = str(schedule.get("day") or "").strip().lower()[:3]
    if not day:
        return {}
    time = str(schedule.get("time") or "any").strip().lower()
    if time not in {"any", "morning", "afternoon"}:
        time = "any"
    return {"day": day, "time": time}


def norm_set(values) -> set[str]:
    if not isinstance(values, list):
        return set()
    return {normalize_value(value) for value in values if isinstance(value, str) and value.strip()}


def normalize_value(value: str) -> str:
    return value.lower().strip().replace(".", "").replace("-", "_").replace(" ", "_")


def aggregate_token_usage(cases: list[dict]) -> dict:
    total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    agent = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    simulated_user = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for case in cases:
        usage = case.get("token_usage", {})
        add_token_usage(total, usage.get("total", {}))
        add_token_usage(agent, usage.get("agent", {}))
        add_token_usage(simulated_user, usage.get("simulated_user", {}))
    count = len(cases) or 1
    return {
        "total": total,
        "agent": agent,
        "simulated_user": simulated_user,
        "average_per_case": {
            key: value / count
            for key, value in total.items()
        },
        "average_agent_per_case": {
            key: value / count
            for key, value in agent.items()
        },
        "average_simulated_user_per_case": {
            key: value / count
            for key, value in simulated_user.items()
        },
    }


def add_token_usage(total: dict, item: dict) -> None:
    for key in ["input_tokens", "output_tokens", "total_tokens"]:
        total[key] = total.get(key, 0) + int(item.get(key, 0) or 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze an eval run directory.")
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
