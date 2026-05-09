from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from eval.metrics import aggregate, parse_tool_output, score_case


def main() -> None:
    args = parse_args()
    case_paths = [
        path
        for path in sorted(case_dir(args.run_dir).glob("*.json"))
        if path.name != "summary.json" and path.name.startswith(("su-", "llu-"))
    ]
    cases = [json.loads(path.read_text(encoding="utf-8")) for path in case_paths]
    scores = [
        score_case(case["card"], case["ground_truth"], case["transcript"], case["response"])
        for case in cases
    ]
    details = analyze_cases(cases, scores)
    summary_path = args.run_dir / "summary.json"
    old_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    passthrough_keys = [
        "provider",
        "model",
        "agent_type",
        "agent_model",
        "user_type",
        "user_model",
        "limit",
        "turns",
        "max_turns",
        "completed_cases",
        "stop_reasons",
        "sim_user",
        "sim_user_model",
        "jobs",
        "token_usage",
    ]
    summary = {
        **{key: old_summary[key] for key in passthrough_keys if key in old_summary},
        "summary": aggregate(scores),
        "scores": scores,
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
        if not score["acceptable_hit"]:
            if not score["retrieval_acceptable_hit"]:
                retrieval_misses.append(user_id)
            else:
                retrieved_not_recommended.append(user_id)
                recommended_ids = score.get("recommended_resource_ids", [])
                acceptable_ids = case["ground_truth"].get("acceptable_gt_resource_ids", [])
                if any(
                    acceptable_id.startswith(recommended_id) or recommended_id.startswith(acceptable_id)
                    for recommended_id in recommended_ids
                    for acceptable_id in acceptable_ids
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
        f"- Agent model: {summary.get('agent_model', summary.get('model'))}",
        f"- User type: {summary.get('user_type', summary.get('sim_user', 'unknown'))}",
        f"- User model: {summary.get('user_model', summary.get('sim_user_model'))}",
        f"- Cases: {agg.get('cases', 0)}",
        f"- Tool result limit: {summary.get('limit')}",
        f"- Max simulated turns: {summary.get('max_turns', summary.get('turns'))}",
        f"- Completed cases: {summary.get('completed_cases', 'n/a')}",
        f"- Stop reasons: {summary.get('stop_reasons', 'n/a')}",
        "",
        "## Headline Metrics",
        "",
        f"- Primary hit rate: {agg.get('primary_hit_rate', 0):.2%}",
        f"- Acceptable hit rate: {agg.get('acceptable_hit_rate', 0):.2%}",
        f"- Retrieval primary hit rate: {agg.get('retrieval_primary_hit_rate', 0):.2%}",
        f"- Retrieval acceptable hit rate: {agg.get('retrieval_acceptable_hit_rate', 0):.2%}",
        f"- Average tool calls per case: {agg.get('average_tool_calls', 0):.2f}",
        f"- Average clarification score: {agg.get('average_clarification_score', 0):.2f}/4",
        f"- Average total tokens per case: {summary.get('token_usage', {}).get('average_per_case', {}).get('total_tokens', 0):.0f}",
        f"- Cases with no tool call: {analysis['no_tool_case_count']}",
        f"- Empty tool output rate: {analysis['empty_tool_output_rate']:.2%}",
        "",
        "## By Difficulty",
        "",
    ]
    for difficulty, item in agg.get("by_difficulty", {}).items():
        lines.extend(
            [
                f"### {difficulty}",
                "",
                f"- Cases: {item['cases']}",
                f"- Primary hit rate: {item['primary_hit_rate']:.2%}",
                f"- Acceptable hit rate: {item['acceptable_hit_rate']:.2%}",
                f"- Retrieval primary hit rate: {item['retrieval_primary_hit_rate']:.2%}",
                f"- Retrieval acceptable hit rate: {item['retrieval_acceptable_hit_rate']:.2%}",
                f"- Average clarification score: {item['average_clarification_score']:.2f}/4",
                "",
            ]
        )
    lines.extend(
        [
            "## Failure Analysis",
            "",
            render_failure_paragraph(agg, analysis),
            "",
            "Failure reason counts:",
            "",
            f"- Retrieval miss: {analysis['failure_counts']['retrieval_miss']}",
            f"- Retrieved GT but did not recommend it or did not cite a full id: {analysis['failure_counts']['retrieved_but_not_recommended_or_id']}",
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
            "- The agent is relatively good at asking follow-up questions in conversation.",
            "- The agent often maps the user's need to a nearby but wrong service category or resource family.",
            "- Exact resource-id citation remains fragile; shortened ids are not counted as correct.",
            "- Low empty-output rate means failure is mostly semantic ranking and selection, not only no-result recovery.",
            "",
            "## Recommended Fixes",
            "",
            "1. Improve the model's category selection or add a semantic retrieval layer before structured filtering.",
            "2. Require final answers to cite exact resource ids copied from tool results.",
            "3. Track id-prefix matches as a diagnostic while keeping exact-id success strict.",
            "4. Add trajectory-level failure labels so benchmark reports separate retrieval, ranking, final-answer, and user-simulation issues.",
            "5. Keep LLM review in dataset construction for acceptable ground truth; pure category/name heuristics are too noisy.",
            "",
        ]
    )
    return "\n".join(lines)


def render_failure_paragraph(agg: dict, analysis: dict) -> str:
    retrieval_rate = agg.get("retrieval_acceptable_hit_rate", 0)
    empty_rate = analysis["empty_tool_output_rate"]
    if empty_rate > 0.1:
        return (
            "The run asked useful clarification questions, but retrieval remained weak "
            f"({retrieval_rate:.2%} retrieval acceptable hit rate). A substantial number of tool calls "
            f"returned no resources ({empty_rate:.2%} empty output rate), so no-result recovery should be inspected."
        )
    return (
        "The run asked useful clarification questions, and empty tool outputs were rare "
        f"({empty_rate:.2%}). Retrieval remained weak ({retrieval_rate:.2%} retrieval acceptable hit rate) "
        "because the agent often chose a nearby but wrong service category/resource family, or retrieved the "
        "right resource but failed to cite the exact id in the final answer."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze an eval run directory.")
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
