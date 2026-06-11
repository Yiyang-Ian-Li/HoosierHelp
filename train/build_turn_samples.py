from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from train.common import assistant_completion, read_jsonl, write_jsonl


def build_turn_samples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for record in records:
        messages = record.get("messages") or []
        raw_outputs = list(record.get("raw_agent_outputs") or [])
        assistant_index = 0
        for message_index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            raw_output = raw_outputs[assistant_index] if assistant_index < len(raw_outputs) else None
            completion = assistant_completion(message, raw_output)
            if not completion:
                assistant_index += 1
                continue
            samples.append(
                {
                    "sample_id": f"{record['user_id']}::turn-{assistant_index}",
                    "user_spec_id": record["user_spec_id"],
                    "user_id": record["user_id"],
                    "user_behavior": record["user_behavior"],
                    "turn_index": assistant_index,
                    "messages_before_assistant": messages[:message_index],
                    "student_completion": completion,
                    "is_tool_call_turn": bool(message.get("tool_calls")),
                    "expected_tool_call": record.get("expected_tool_call"),
                    "conversation_score": record.get("score"),
                    "parse_mode": record.get("parse_mode"),
                    "user_simulator_state": record.get("user_simulator_state"),
                }
            )
            assistant_index += 1
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build assistant-turn training samples from tool-call eval records.")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = build_turn_samples(read_jsonl(args.records))
    write_jsonl(args.output, samples)
    print(f"Wrote {args.output} ({len(samples)} assistant-turn samples)")


if __name__ == "__main__":
    main()
