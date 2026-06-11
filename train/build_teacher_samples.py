from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from eval.tool_call_parsers import parse_qwen_xml_tool_call
from train.common import DEFAULT_MODEL, read_jsonl, render_qwen_prompt, tool_schema_from_resources


def load_model(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        if args.load_in_4bit
        else None
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def completed_sample_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row.get("sample_id")
            if sample_id:
                done.add(str(sample_id))
    return done


def generate_teacher_completion(
    model,
    tokenizer,
    sample: dict[str, Any],
    tool_schema: dict[str, Any],
    max_new_tokens: int,
) -> str:
    prompt = render_qwen_prompt(
        tokenizer,
        sample["messages_before_assistant"],
        tool_schema,
        privileged=True,
        behavior=sample.get("user_behavior"),
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    eos_token_ids = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in eos_token_ids:
        eos_token_ids.append(im_end_id)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_ids,
        )
    return tokenizer.decode(output_ids[0, inputs.input_ids.shape[1] :], skip_special_tokens=True).strip()


def teacher_metadata(text: str) -> dict[str, Any]:
    parsed = parse_qwen_xml_tool_call(text)
    return {
        "teacher_parse_mode": parsed.parse_mode if parsed else None,
        "teacher_tool_call": {"name": parsed.name, "arguments": parsed.arguments} if parsed else None,
    }


def build_teacher_samples(args: argparse.Namespace) -> None:
    samples = read_jsonl(args.samples)
    if args.limit_samples:
        samples = samples[: args.limit_samples]
    done = completed_sample_ids(args.output) if args.resume else set()
    tool_schema = tool_schema_from_resources(args.resources)
    model, tokenizer = load_model(args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    written = 0
    with args.output.open(mode, encoding="utf-8") as handle:
        for sample in tqdm(samples, desc="building teacher samples"):
            sample_id = str(sample.get("sample_id"))
            if sample_id in done:
                continue
            completion = generate_teacher_completion(model, tokenizer, sample, tool_schema, args.max_new_tokens)
            row = {
                **sample,
                "teacher_completion": completion,
                **teacher_metadata(completion),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            written += 1
    print(f"Wrote {written} new teacher samples to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate privileged teacher completions for on-policy assistant-turn samples.")
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resources", type=Path, default=Path("data/benchmark/filtered_resources_tagged.csv"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    return parser.parse_args()


def main() -> None:
    build_teacher_samples(parse_args())


if __name__ == "__main__":
    main()
