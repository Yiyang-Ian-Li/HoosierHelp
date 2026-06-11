from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


SCORE_KEYS = (
    "service_match",
    "location_match",
    "schedule_match",
    "intake_match",
    "documents_match",
    "eligibility_match",
)


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.records)
    print_behavior_table(records)
    print()
    print_failures(records)
    if args.examples:
        print()
        print_examples(records, args.examples)


def print_behavior_table(records: list[dict[str, Any]]) -> None:
    grouped = group_by_behavior(records)
    normal_turns = {row["user_spec_id"]: agent_turn_count(row) for row in grouped.get("normal", [])}
    print("behavior,n,all,valid,agent_turns_avg,agent_turns_median,user_msgs_avg,user_msgs_median,shorter_than_normal")
    for behavior in sorted(grouped):
        rows = grouped[behavior]
        turns = [agent_turn_count(row) for row in rows]
        user_msgs = [user_message_count(row) for row in rows]
        scores = [row["score"] for row in rows]
        shorter = sum(
            agent_turn_count(row) < normal_turns.get(row["user_spec_id"], 0)
            for row in rows
            if behavior != "normal"
        )
        print(
            ",".join(
                (
                    behavior,
                    str(len(rows)),
                    f"{rate(scores, 'all_match'):.3f}",
                    f"{rate(scores, 'valid_tool_call'):.3f}",
                    f"{mean(turns):.2f}",
                    f"{median(turns):.1f}",
                    f"{mean(user_msgs):.2f}",
                    f"{median(user_msgs):.1f}",
                    str(shorter),
                )
            )
        )


def print_failures(records: list[dict[str, Any]]) -> None:
    grouped = group_by_behavior(records)
    print("field failures")
    for behavior in sorted(grouped):
        failures: Counter[str] = Counter()
        for row in grouped[behavior]:
            score = row["score"]
            for key in SCORE_KEYS:
                if not score.get(key):
                    failures[key.replace("_match", "")] += 1
        parts = ", ".join(f"{key}={value}" for key, value in failures.most_common())
        print(f"{behavior}: {parts or 'none'}")


def print_examples(records: list[dict[str, Any]], limit: int) -> None:
    grouped = group_by_behavior(records)
    print("examples")
    for behavior in sorted(grouped):
        rows = sorted(grouped[behavior], key=lambda row: (row["score"].get("all_match", True), agent_turn_count(row)))
        print(f"\n[{behavior}]")
        for row in rows[:limit]:
            print(f"{row['user_id']} all={row['score'].get('all_match')} valid={row['score'].get('valid_tool_call')} turns={agent_turn_count(row)}")
            for message in row.get("messages", [])[:6]:
                content = str(message.get("content", "")).replace("\n", " ").strip()
                if not content and message.get("tool_calls"):
                    content = "<tool_call>"
                print(f"  {message.get('role')}: {content[:240]}")


def group_by_behavior(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["user_behavior"]].append(record)
    return grouped


def rate(scores: list[dict[str, Any]], key: str) -> float:
    return sum(bool(score.get(key)) for score in scores) / len(scores) if scores else 0.0


def agent_turn_count(record: dict[str, Any]) -> int:
    return len(record.get("raw_agent_outputs") or [])


def user_message_count(record: dict[str, Any]) -> int:
    return sum(1 for message in record.get("messages", []) if message.get("role") == "user")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a tool-call eval run.")
    parser.add_argument("records", type=Path)
    parser.add_argument("--examples", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
