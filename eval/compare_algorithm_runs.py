from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRIC_KEYS = (
    "valid_tool_call_rate",
    "service_match_rate",
    "location_match_rate",
    "schedule_match_rate",
    "intake_match_rate",
    "documents_match_rate",
    "eligibility_match_rate",
    "all_match_rate",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run_stats(run_dir: Path) -> dict[str, Any]:
    summary = read_json(run_dir / "summary.json")
    records = read_jsonl(run_dir / "records.jsonl")
    return {
        "summary": summary,
        "records": records,
        "parse_none": summary.get("parse_modes", {}).get("none", sum(record.get("parse_mode") == "none" for record in records)),
        "avg_agent_turns": summary.get("turn_stats", {}).get("agent_turns_avg", count_avg_agent_turns(records)),
        "field_failures": summary.get("failure_counts") or field_failures(records),
    }


def count_avg_agent_turns(records: list[dict[str, Any]]) -> float:
    return sum(len(record.get("raw_agent_outputs") or []) for record in records) / len(records) if records else 0.0


def field_failures(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        key.replace("_rate", ""): sum(not bool(record.get("score", {}).get(key.replace("_rate", ""), False)) for record in records)
        for key in METRIC_KEYS
    }


def markdown_report(runs: list[tuple[str, Path]]) -> str:
    stats = [(name, path, run_stats(path)) for name, path in runs]
    lines = ["# Algorithm Run Comparison", ""]
    lines.extend(main_table(stats))
    lines.append("")
    lines.extend(behavior_tables(stats))
    lines.append("")
    lines.extend(diagnostic_table(stats))
    lines.append("")
    lines.extend(field_failure_table(stats))
    lines.append("")
    return "\n".join(lines)


def main_table(stats: list[tuple[str, Path, dict[str, Any]]]) -> list[str]:
    lines = [
        "## Overall",
        "",
        "| Method | Cases | All | Valid | Service | Location | Schedule | Intake | Documents | Eligibility |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, _path, stat in stats:
        overall = stat["summary"]["overall"]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(overall.get("n", 0)),
                    fmt(overall.get("all_match_rate")),
                    fmt(overall.get("valid_tool_call_rate")),
                    fmt(overall.get("service_match_rate")),
                    fmt(overall.get("location_match_rate")),
                    fmt(overall.get("schedule_match_rate")),
                    fmt(overall.get("intake_match_rate")),
                    fmt(overall.get("documents_match_rate")),
                    fmt(overall.get("eligibility_match_rate")),
                ]
            )
            + " |"
        )
    return lines


def behavior_tables(stats: list[tuple[str, Path, dict[str, Any]]]) -> list[str]:
    behaviors = sorted(
        {
            behavior
            for _name, _path, stat in stats
            for behavior in stat["summary"].get("by_user_behavior", {})
        }
    )
    lines = [
        "## By Behavior",
        "",
        "| Method | Behavior | N | All | Valid | Location |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, _path, stat in stats:
        by_behavior = stat["summary"].get("by_user_behavior", {})
        for behavior in behaviors:
            row = by_behavior.get(behavior, {"n": 0})
            lines.append(
                f"| {name} | {behavior} | {row.get('n', 0)} | {fmt(row.get('all_match_rate'))} | "
                f"{fmt(row.get('valid_tool_call_rate'))} | {fmt(row.get('location_match_rate'))} |"
            )
    return lines


def diagnostic_table(stats: list[tuple[str, Path, dict[str, Any]]]) -> list[str]:
    lines = [
        "## Diagnostics",
        "",
        "| Method | Parse None | Parse None Rate | Avg Agent Turns |",
        "|---|---:|---:|---:|",
    ]
    for name, _path, stat in stats:
        cases = stat["summary"]["overall"].get("n", 0)
        parse_none = stat["parse_none"]
        parse_none_rate = parse_none / cases if cases else 0.0
        lines.append(f"| {name} | {parse_none} | {fmt(parse_none_rate)} | {stat['avg_agent_turns']:.2f} |")
    return lines


def field_failure_table(stats: list[tuple[str, Path, dict[str, Any]]]) -> list[str]:
    fields = ("service", "location", "schedule", "intake", "documents", "eligibility")
    lines = [
        "## Field Failures",
        "",
        "| Method | Service | Location | Schedule | Intake | Documents | Eligibility |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, _path, stat in stats:
        failures = stat["field_failures"]
        values = [failure_value(failures, field) for field in fields]
        lines.append("| " + " | ".join([name, *(str(value) for value in values)]) + " |")
    return lines


def failure_value(failures: dict[str, int], field: str) -> int:
    return int(failures.get(field, failures.get(f"{field}_match", 0)))


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.3f}"


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.name, path
    name, path = value.split("=", 1)
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare tool-call eval runs for algorithm experiments.")
    parser.add_argument("runs", nargs="+", help="Run dirs, optionally as name=path.")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = markdown_report([parse_run(value) for value in args.runs])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report)


if __name__ == "__main__":
    main()
