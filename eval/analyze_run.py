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
    case_paths = [
        path
        for path in sorted(case_dir(args.run_dir).glob("*.json"))
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
    summary_path = args.run_dir / "summary.json"
    old_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    passthrough_keys = [
        "provider",
        "agent_type",
        "agent_model",
        "user_model",
        "difficulty",
        "users",
        "index_path",
        "max_turns",
        "completed_cases",
        "stop_reasons",
        "jobs",
        "token_usage",
        "simulated_user_diagnostics",
    ]
    summary = {
        **{key: old_summary[key] for key in passthrough_keys if key in old_summary},
        "simulated_user_diagnostics": old_summary.get("simulated_user_diagnostics")
        or aggregate_simulated_user_diagnostics(cases),
        "summary": aggregate(scores),
        "breakdown": aggregate_breakdown(cases),
        "analysis": details,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.run_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    print(f"Analyzed {len(cases)} cases")
    print(f"Wrote {summary_path}")
    print(f"Wrote {args.run_dir / 'report.md'}")


def case_dir(run_dir: Path) -> Path:
    conversations = run_dir / "conversations"
    return conversations if conversations.exists() else run_dir


def analyze_cases(cases: list[dict], scores: list[dict]) -> dict:
    total_function_calls = 0
    empty_outputs = 0
    filter_counts = Counter()
    service_category_counts = Counter()
    no_tool_cases = []
    general_advice_after_empty = []
    retrieval_misses = []
    retrieved_not_recommended = []
    possible_id_format = []
    score_by_user = {score["user_id"]: score for score in scores}
    for case in cases:
        user_id = case["card"]["user_id"]
        score = score_by_user[user_id]
        if not score["ground_truth_hit"]:
            if not score["retrieval_ground_truth_hit"]:
                retrieval_misses.append(user_id)
            else:
                retrieved_not_recommended.append(user_id)
                recommended_ids = score.get("recommended_resource_ids", [])
                ground_truth_ids = case["ground_truth"].get("ground_truth_resource_ids", [])
                if any(
                    ground_truth_id.startswith(recommended_id) or recommended_id.startswith(ground_truth_id)
                    for recommended_id in recommended_ids
                    for ground_truth_id in ground_truth_ids
                ):
                    possible_id_format.append(user_id)
        response = case["response"]
        input_items = response.get("input", [])
        function_calls = [item for item in input_items if item.get("type") == "function_call"]
        outputs = [item for item in input_items if item.get("type") == "function_call_output"]
        if not function_calls:
            no_tool_cases.append(case["card"]["user_id"])
        total_function_calls += len(function_calls)
        for item in function_calls:
            args = json.loads(item.get("arguments") or "{}")
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
                general_advice_after_empty.append(case["card"]["user_id"])
    return {
        "total_function_calls": total_function_calls,
        "empty_tool_outputs": empty_outputs,
        "empty_tool_output_rate": empty_outputs / total_function_calls if total_function_calls else 0,
        "no_tool_case_count": len(no_tool_cases),
        "no_tool_cases": no_tool_cases,
        "common_filters": filter_counts.most_common(),
        "common_requested_service_categories": service_category_counts.most_common(20),
        "general_advice_after_empty_cases": general_advice_after_empty,
        "failure_counts": {
            "retrieval_miss": len(retrieval_misses),
            "retrieved_but_not_recommended_or_id": len(retrieved_not_recommended),
            "possible_id_format": len(possible_id_format),
        },
        "retrieval_miss_cases": retrieval_misses,
        "retrieved_but_not_recommended_or_id_cases": retrieved_not_recommended,
        "possible_id_format_cases": possible_id_format,
    }


def aggregate_simulated_user_diagnostics(cases: list[dict]) -> dict:
    trait_counts = Counter()
    for case in cases:
        diagnostics = case.get("simulated_user_diagnostics") or {}
        trait_counts.update(diagnostics.get("traits") or [])
    return {
        "trait_counts": dict(trait_counts),
    }


def render_report(summary: dict) -> str:
    agg = summary["summary"]
    analysis = summary["analysis"]
    lines = [
        "# Simulated User Evaluation Report",
        "",
        "## Setup",
        "",
        f"- Provider: {summary.get('provider')}",
        f"- Agent type: {summary.get('agent_type', 'unknown')}",
        f"- Agent model: {summary.get('agent_model')}",
        f"- User model: {summary.get('user_model')}",
        f"- Difficulty: {summary.get('difficulty', 'all')}",
        f"- Users: {summary.get('users')}",
        f"- Index path: {summary.get('index_path')}",
        f"- Cases: {agg.get('cases', 0)}",
        f"- Max simulated turns: {summary.get('max_turns')}",
        f"- Completed cases: {summary.get('completed_cases', 'n/a')}",
        f"- Stop reasons: {summary.get('stop_reasons', 'n/a')}",
        "",
        "## Headline Metrics",
        "",
        f"- Ground truth hit rate: {agg.get('ground_truth_hit_rate', 0):.2%}",
        f"- Retrieval ground truth hit rate: {agg.get('retrieval_ground_truth_hit_rate', 0):.2%}",
        f"- Average tool calls per case: {agg.get('average_tool_calls', 0):.2f}",
        f"- Average turns per case: {agg.get('average_turns', 0):.2f}",
        f"- Multiple recommendation turns: {agg.get('multiple_recommendation_turn_rate', 0):.2%}",
        f"- Recommended IDs not retrieved: {agg.get('recommended_ids_not_retrieved_rate', 0):.2%}",
        f"- Average total tokens per case: {summary.get('token_usage', {}).get('average_per_case', {}).get('total_tokens', 0):.0f}",
        f"- Cases with no tool call: {analysis['no_tool_case_count']}",
        f"- Empty tool output rate: {analysis['empty_tool_output_rate']:.2%}",
    ]
    diagnostics = summary.get("simulated_user_diagnostics") or {}
    if diagnostics:
        lines.extend(
            [
                f"- Trait counts: {diagnostics.get('trait_counts', {})}",
            ]
        )
    lines.extend(render_breakdown(summary.get("breakdown") or {}))
    lines.extend(
        [
            "## Failure Analysis",
            "",
            render_failure_paragraph(agg, analysis),
            "",
            "Failure reason counts:",
            "",
            f"- Retrieval miss: {analysis['failure_counts']['retrieval_miss']}",
            f"- Retrieved ground truth but did not recommend it or did not cite a full id: {analysis['failure_counts']['retrieved_but_not_recommended_or_id']}",
            f"- Likely id-format-only failures: {analysis['failure_counts']['possible_id_format']}",
            "",
            "Common non-empty filters in tool calls:",
            "",
        ]
    )
    for key, count in analysis["common_filters"][:12]:
        lines.append(f"- `{key}`: {count}")
    lines.extend(
        [
            "",
            "Common requested service categories:",
            "",
        ]
    )
    for key, count in analysis["common_requested_service_categories"][:12]:
        lines.append(f"- `{key}`: {count}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The agent often maps the user's need to a nearby but wrong service category or resource family.",
            "- Final recommendation scoring uses the last agent recommendation with resource IDs before completion.",
            "- Low empty-output rate means failure is mostly semantic ranking and selection, not only no-result recovery.",
            "",
            "## Recommended Fixes",
            "",
            "1. Improve the model's category selection or add a semantic retrieval layer before structured filtering.",
            "2. Keep final recommendation ids exact and copied from tool results.",
            "3. Add trajectory-level failure labels so benchmark reports separate retrieval, ranking, final-answer, and user-simulation issues.",
            "4. Inspect user-card generation quality when failures cluster around a category or access constraint.",
            "",
        ]
    )
    return "\n".join(lines)


def render_breakdown(breakdown: dict) -> list[str]:
    lines = []
    for title, key in [("## By Difficulty", "by_difficulty"), ("## By Trait", "by_trait")]:
        groups = breakdown.get(key) or {}
        if not groups:
            continue
        lines.extend([title, ""])
        for group, metrics in groups.items():
            lines.append(
                "- "
                f"{group}: "
                f"{metrics.get('ground_truth_hits', 0)}/{metrics.get('cases', 0)} "
                f"({metrics.get('ground_truth_hit_rate', 0):.2%}), "
                f"retrieval {metrics.get('retrieval_ground_truth_hits', 0)}/{metrics.get('cases', 0)} "
                f"({metrics.get('retrieval_ground_truth_hit_rate', 0):.2%}), "
                f"no_match={metrics.get('no_match_count', 0)}, "
                f"avg_turns={metrics.get('average_turns', 0):.2f}"
            )
        lines.append("")
    return lines


def render_failure_paragraph(agg: dict, analysis: dict) -> str:
    retrieval_rate = agg.get("retrieval_ground_truth_hit_rate", 0)
    empty_rate = analysis["empty_tool_output_rate"]
    if empty_rate > 0.1:
        return (
            "Retrieval remained weak "
            f"({retrieval_rate:.2%} retrieval ground truth hit rate). A substantial number of tool calls "
            f"returned no resources ({empty_rate:.2%} empty output rate), so no-result recovery should be inspected."
        )
    return (
        "Empty tool outputs were rare "
        f"({empty_rate:.2%}). Retrieval remained weak ({retrieval_rate:.2%} retrieval ground truth hit rate) "
        "because the agent often chose a nearby but wrong service category/resource family, or retrieved the "
        "right resource but failed to include the exact id in the completed final recommendation."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze an eval run directory.")
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
