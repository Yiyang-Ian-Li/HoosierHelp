from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from train.common import DEFAULT_MODEL, read_jsonl, render_qwen_prompt, tool_schema_from_resources
from eval.tool_call_parsers import parse_qwen_xml_tool_call
from eval.tool_call_schema import normalize_tool_args, tool_arg_scores


SOFT_REWARD_WEIGHTS = {
    "format": 0.15,
    "service": 0.15,
    "location": 0.15,
    "schedule": 0.15,
    "intake": 0.10,
    "documents": 0.10,
    "eligibility": 0.10,
    "all_exact": 0.10,
}
LONG_COMPLETION_PENALTY = 0.05
LONG_COMPLETION_CHARS = 1200


def build_grpo_dataset(
    samples: list[dict[str, Any]],
    tokenizer,
    tool_schema: dict[str, Any],
    *,
    include_non_tool_turns: bool = False,
    include_terminal_failures: bool = True,
) -> Dataset:
    rows = []
    seen = set()
    last_sample_by_user = {}
    for sample in samples:
        last_sample_by_user[sample["user_id"]] = sample["sample_id"]
    for sample in samples:
        is_tool_call_turn = bool(sample.get("is_tool_call_turn"))
        is_terminal_failure = (
            include_terminal_failures
            and sample["sample_id"] == last_sample_by_user.get(sample["user_id"])
            and not is_tool_call_turn
            and not bool((sample.get("conversation_score") or {}).get("all_match"))
        )
        if not include_non_tool_turns and not is_tool_call_turn and not is_terminal_failure:
            continue
        key = sample["sample_id"]
        if key in seen:
            continue
        seen.add(key)
        prompt = render_qwen_prompt(
            tokenizer,
            sample["messages_before_assistant"],
            tool_schema,
            privileged=False,
            behavior=sample.get("user_behavior"),
        )
        rows.append(
            {
                "prompt": prompt,
                "expected_tool_call": json.dumps(sample["expected_tool_call"], ensure_ascii=False),
                "sample_id": sample["sample_id"],
                "user_behavior": sample["user_behavior"],
                "sample_kind": "tool_call_turn" if is_tool_call_turn else "terminal_failure",
            }
        )
    return Dataset.from_list(rows)


def tool_call_reward(completions: list[str], expected_tool_call: list[str], **_: Any) -> list[float]:
    rewards = []
    for completion, expected_raw in zip(completions, expected_tool_call):
        expected = json.loads(expected_raw)
        parsed = parse_qwen_xml_tool_call(completion)
        score = soft_tool_arg_scores(parsed, expected, completion)
        reward = sum(SOFT_REWARD_WEIGHTS[key] * score[key] for key in SOFT_REWARD_WEIGHTS)
        if len(completion) > LONG_COMPLETION_CHARS:
            reward -= LONG_COMPLETION_PENALTY
        rewards.append(max(0.0, min(1.0, reward)))
    return rewards


def soft_tool_arg_scores(parsed, expected: dict[str, Any], completion: str) -> dict[str, float]:
    if parsed is None:
        return {key: 0.0 for key in SOFT_REWARD_WEIGHTS}
    predicted_args = normalize_tool_args(parsed.arguments)
    expected_args = normalize_tool_args(expected)
    exact = tool_arg_scores(predicted_args, expected_args)
    return {
        "format": format_score(parsed, completion),
        "service": set_f1(predicted_args["service_categories"], expected_args["service_categories"]),
        "location": location_score(predicted_args, expected_args),
        "schedule": schedule_score(predicted_args["schedule"], expected_args["schedule"]),
        "intake": set_f1(predicted_args["intake_methods"], expected_args["intake_methods"]),
        "documents": set_f1(predicted_args["available_documents"], expected_args["available_documents"]),
        "eligibility": set_f1(predicted_args["eligibility"], expected_args["eligibility"]),
        "all_exact": float(exact["all_match"]),
    }


def format_score(parsed, completion: str) -> float:
    stripped = completion.strip()
    if parsed.parse_mode == "qwen_xml" and stripped.startswith("<tool_call>") and stripped.endswith("</tool_call>"):
        return 1.0
    if parsed.parse_mode == "qwen_xml":
        return 0.8
    return 0.5


def location_score(predicted: dict[str, Any], expected: dict[str, Any]) -> float:
    scores = []
    for key in ("counties", "cities", "zipcodes"):
        if predicted[key] or expected[key]:
            scores.append(set_f1(predicted[key], expected[key]))
    if not scores:
        return 1.0
    return sum(scores) / len(scores)


def schedule_score(predicted: dict[str, Any], expected: dict[str, Any]) -> float:
    if not expected:
        return 1.0 if not predicted else 0.0
    if expected.get("requires_24_hours"):
        return 1.0 if predicted.get("requires_24_hours") is True else 0.0
    keys = ("day", "start_time", "end_time")
    return sum(float(predicted.get(key) == expected.get(key)) for key in keys) / len(keys)


def set_f1(predicted_values: list[str], expected_values: list[str]) -> float:
    predicted = set(predicted_values)
    expected = set(expected_values)
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    overlap = len(predicted & expected)
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO training for HoosierHelp tool-call behavior.")
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--resources", type=Path, default=Path("data/benchmark/filtered_resources_tagged.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument(
        "--include-non-tool-turns",
        action="store_true",
        help="Also train GRPO on intermediate assistant turns. Default is final tool-call turns only because the reward scores final search_resources arguments.",
    )
    parser.add_argument(
        "--include-terminal-failures",
        action="store_true",
        default=True,
        help="Also train on each failed conversation's final non-tool assistant state, so GRPO can learn to recover with a tool call.",
    )
    parser.add_argument("--no-terminal-failures", dest="include_terminal_failures", action="store_false")
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    samples = read_jsonl(args.samples)
    if args.limit_samples:
        samples = samples[: args.limit_samples]
    dataset = build_grpo_dataset(
        samples,
        tokenizer,
        tool_schema_from_resources(args.resources),
        include_non_tool_turns=args.include_non_tool_turns,
        include_terminal_failures=args.include_terminal_failures,
    )
    if len(dataset) == 0:
        raise ValueError("GRPO dataset is empty. Check samples or pass --include-non-tool-turns for an ablation.")
    kind_counts = {}
    for kind in dataset["sample_kind"]:
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
    print(f"GRPO dataset size: {len(dataset)} ({kind_counts})")
    config = GRPOConfig(
        output_dir=str(args.output_dir),
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        temperature=args.temperature,
        beta=args.beta,
        bf16=args.bf16,
        logging_steps=10,
        save_steps=100,
        report_to=[],
        remove_unused_columns=False,
        model_init_kwargs={
            "load_in_4bit": args.load_in_4bit,
            "torch_dtype": "bfloat16",
            "trust_remote_code": True,
        },
    )
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.lora_target_modules.split(","),
    )
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=tool_call_reward,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "adapter"))
    tokenizer.save_pretrained(args.output_dir / "adapter")


if __name__ == "__main__":
    main()
