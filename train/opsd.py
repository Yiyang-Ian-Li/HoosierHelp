from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from train.common import DEFAULT_MODEL, read_jsonl, render_qwen_prompt, tool_schema_from_resources


@dataclass
class EncodedSample:
    student_input_ids: torch.Tensor
    teacher_input_ids: torch.Tensor
    completion_len: int
    metadata: dict[str, Any]


class OPSDDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer,
        tool_schema: dict[str, Any],
        max_length: int,
        completion_field: str,
    ):
        self.rows = rows
        self.tokenizer = tokenizer
        self.tool_schema = tool_schema
        self.max_length = max_length
        self.completion_field = completion_field

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> EncodedSample:
        row = self.rows[index]
        completion = row.get(self.completion_field)
        if not completion:
            raise ValueError(f"sample {row.get('sample_id')} is missing required completion field `{self.completion_field}`")
        student_prompt = render_qwen_prompt(
            self.tokenizer,
            row["messages_before_assistant"],
            self.tool_schema,
            privileged=False,
            behavior=row.get("user_behavior"),
        )
        teacher_prompt = render_qwen_prompt(
            self.tokenizer,
            row["messages_before_assistant"],
            self.tool_schema,
            privileged=True,
            behavior=row.get("user_behavior"),
        )
        completion_ids = self.tokenizer(completion, add_special_tokens=False).input_ids
        if len(completion_ids) >= self.max_length:
            completion_ids = completion_ids[: self.max_length - 1]
        max_prompt_len = self.max_length - len(completion_ids)
        student_prompt_ids = self.tokenizer(student_prompt, add_special_tokens=False).input_ids[-max_prompt_len:]
        teacher_prompt_ids = self.tokenizer(teacher_prompt, add_special_tokens=False).input_ids[-max_prompt_len:]
        student_ids = student_prompt_ids + completion_ids
        teacher_ids = teacher_prompt_ids + completion_ids
        return EncodedSample(
            student_input_ids=torch.tensor(student_ids, dtype=torch.long),
            teacher_input_ids=torch.tensor(teacher_ids, dtype=torch.long),
            completion_len=min(len(completion_ids), len(student_ids) - 1, len(teacher_ids) - 1),
            metadata={"sample_id": row.get("sample_id"), "user_behavior": row.get("user_behavior"), "completion_field": self.completion_field},
        )


def collate_samples(samples: list[EncodedSample], pad_token_id: int) -> dict[str, Any]:
    max_student_len = max(sample.student_input_ids.numel() for sample in samples)
    max_teacher_len = max(sample.teacher_input_ids.numel() for sample in samples)
    student_ids = torch.full((len(samples), max_student_len), pad_token_id, dtype=torch.long)
    teacher_ids = torch.full((len(samples), max_teacher_len), pad_token_id, dtype=torch.long)
    student_attention_mask = torch.zeros_like(student_ids)
    teacher_attention_mask = torch.zeros_like(teacher_ids)
    student_lengths = []
    teacher_lengths = []
    completion_lens = []
    for index, sample in enumerate(samples):
        student_len = sample.student_input_ids.numel()
        teacher_len = sample.teacher_input_ids.numel()
        student_ids[index, :student_len] = sample.student_input_ids
        teacher_ids[index, :teacher_len] = sample.teacher_input_ids
        student_attention_mask[index, :student_len] = 1
        teacher_attention_mask[index, :teacher_len] = 1
        student_lengths.append(student_len)
        teacher_lengths.append(teacher_len)
        completion_lens.append(sample.completion_len)
    return {
        "student_input_ids": student_ids,
        "teacher_input_ids": teacher_ids,
        "student_attention_mask": student_attention_mask,
        "teacher_attention_mask": teacher_attention_mask,
        "student_lengths": torch.tensor(student_lengths, dtype=torch.long),
        "teacher_lengths": torch.tensor(teacher_lengths, dtype=torch.long),
        "completion_lens": torch.tensor(completion_lens, dtype=torch.long),
        "metadata": [sample.metadata for sample in samples],
    }


def load_model_and_tokenizer(args: argparse.Namespace):
    tokenizer_path = str(args.adapter or args.model)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
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
    if args.load_in_4bit:
        checkpoint_kwargs = {"use_reentrant": False} if args.gradient_checkpointing else None
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs=checkpoint_kwargs,
        )
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.lora_target_modules.split(","),
    )
    model = get_peft_model(model, lora)
    model.train()
    return model, tokenizer


def forward_kl_for_sample(model, sample: EncodedSample, temperature: float) -> torch.Tensor:
    device = model.device
    student_ids = sample.student_input_ids.to(device).unsqueeze(0)
    teacher_ids = sample.teacher_input_ids.to(device).unsqueeze(0)
    completion_len = sample.completion_len
    if completion_len <= 0:
        return torch.zeros((), device=device)

    with torch.no_grad():
        teacher_logits = model(teacher_ids).logits[:, :-1, :]
    student_logits = model(student_ids).logits[:, :-1, :]

    student_slice = student_logits[:, -completion_len:, :] / temperature
    teacher_slice = teacher_logits[:, -completion_len:, :] / temperature
    teacher_probs = F.softmax(teacher_slice, dim=-1)
    student_log_probs = F.log_softmax(student_slice, dim=-1)
    return -(teacher_probs * student_log_probs).sum(dim=-1).mean() * (temperature**2)


def completion_mask(lengths: torch.Tensor, completion_lens: torch.Tensor, max_logits_len: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(max_logits_len, device=device).unsqueeze(0)
    starts = (lengths.to(device) - 1 - completion_lens.to(device)).unsqueeze(1)
    ends = (lengths.to(device) - 1).unsqueeze(1)
    return (positions >= starts) & (positions < ends)


def forward_kl_for_batch(model, batch: dict[str, Any], temperature: float) -> torch.Tensor:
    device = model.device
    student_ids = batch["student_input_ids"].to(device)
    teacher_ids = batch["teacher_input_ids"].to(device)
    student_attention_mask = batch["student_attention_mask"].to(device)
    teacher_attention_mask = batch["teacher_attention_mask"].to(device)
    student_lengths = batch["student_lengths"].to(device)
    teacher_lengths = batch["teacher_lengths"].to(device)
    completion_lens = batch["completion_lens"].to(device)
    if int(completion_lens.max().item()) <= 0:
        return torch.zeros((), device=device)
    if student_ids.shape[0] == 1:
        completion_len = int(completion_lens[0].item())
        logits_to_keep = completion_len + 1
        student_len = int(student_lengths[0].item())
        with torch.no_grad():
            teacher_logits = model(teacher_ids, attention_mask=teacher_attention_mask, logits_to_keep=logits_to_keep).logits
        student_logits = model(student_ids, attention_mask=student_attention_mask, logits_to_keep=logits_to_keep).logits

        teacher_selected = teacher_logits[0, :completion_len, :] / temperature
        student_selected = student_logits[0, :completion_len, :] / temperature
        teacher_probs = F.softmax(teacher_selected, dim=-1)
        student_log_probs = F.log_softmax(student_selected, dim=-1)
        return -(teacher_probs * student_log_probs).sum(dim=-1).mean() * (temperature**2)

    with torch.no_grad():
        teacher_logits = model(teacher_ids, attention_mask=teacher_attention_mask).logits[:, :-1, :] / temperature
    student_logits = model(student_ids, attention_mask=student_attention_mask).logits[:, :-1, :] / temperature

    student_mask = completion_mask(student_lengths, completion_lens, student_logits.shape[1], device)
    teacher_mask = completion_mask(teacher_lengths, completion_lens, teacher_logits.shape[1], device)
    if not bool(student_mask.any()) or not bool(teacher_mask.any()):
        return torch.zeros((), device=device)

    teacher_selected = teacher_logits[teacher_mask]
    student_selected = student_logits[student_mask]
    token_count = min(teacher_selected.shape[0], student_selected.shape[0])
    teacher_selected = teacher_selected[:token_count]
    student_selected = student_selected[:token_count]
    teacher_probs = F.softmax(teacher_selected, dim=-1)
    student_log_probs = F.log_softmax(student_selected, dim=-1)
    return -(teacher_probs * student_log_probs).sum(dim=-1).mean() * (temperature**2)


def train(args: argparse.Namespace) -> Path:
    rows = read_jsonl(args.samples)
    if args.limit_samples:
        rows = rows[: args.limit_samples]
    tool_schema = tool_schema_from_resources(args.resources)
    model, tokenizer = load_model_and_tokenizer(args)
    dataset = OPSDDataset(rows, tokenizer, tool_schema, args.max_length, args.completion_field)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda samples: collate_samples(samples, tokenizer.pad_token_id),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "args.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n", encoding="utf-8")
    metrics_path = args.output_dir / "train_metrics.jsonl"
    step = 0
    with metrics_path.open("w", encoding="utf-8") as metrics:
        for epoch in range(args.epochs):
            progress = tqdm(loader, desc=f"opsd epoch {epoch + 1}")
            for batch in progress:
                loss = forward_kl_for_batch(model, batch, args.temperature)
                loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                step += 1
                progress.set_postfix(loss=float(loss.detach().cpu()))
                if step % args.logging_steps == 0:
                    metrics.write(json.dumps({"step": step, "epoch": epoch + 1, "loss": float(loss.detach().cpu())}) + "\n")
                    metrics.flush()
                if args.max_steps and step >= args.max_steps:
                    if step % args.gradient_accumulation_steps != 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    model.save_pretrained(args.output_dir / "adapter")
                    tokenizer.save_pretrained(args.output_dir / "adapter")
                    return args.output_dir
            if step % args.gradient_accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
    model.save_pretrained(args.output_dir / "adapter")
    tokenizer.save_pretrained(args.output_dir / "adapter")
    return args.output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="On-policy self-distillation with privileged behavior prompts.")
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--resources", type=Path, default=Path("data/benchmark/filtered_resources_tagged.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--completion-field", default="teacher_completion")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
