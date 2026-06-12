from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.llm import create_chat_completion_with_retries, create_response_with_retries, make_openai_client
from eval.tool_call_parsers import ParsedToolCall, parse_qwen_xml_tool_calls, parse_responses_tool_calls
from eval.tool_call_prompts import AGENT_SYSTEM_PROMPT


AGENT_GENERATION_TOKEN_LIMIT = 8192


@dataclass
class BackendOutput:
    text: str
    tool_calls: list[ParsedToolCall]
    raw: Any = None
    token_usage: dict[str, int] | None = None

    @property
    def tool_call(self) -> ParsedToolCall | None:
        return self.tool_calls[0] if self.tool_calls else None


class AgentBackend:
    def generate(self, messages: list[dict[str, Any]], tool_schema: dict[str, Any]) -> BackendOutput:
        raise NotImplementedError


class LocalHFBackend(AgentBackend):
    def __init__(
        self,
        model_name: str,
        adapter: Path | None = None,
        temperature: float = 0.0,
        load_in_4bit: bool = True,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.max_new_tokens = AGENT_GENERATION_TOKEN_LIMIT
        self.temperature = temperature
        tokenizer_path = str(adapter or model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        quantization_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            if load_in_4bit
            else None
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        if adapter is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def generate(self, messages: list[dict[str, Any]], tool_schema: dict[str, Any]) -> BackendOutput:
        prompt = self._chat_prompt([{"role": "system", "content": AGENT_SYSTEM_PROMPT}, *messages], tool_schema)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        eos_token_ids = [self.tokenizer.eos_token_id]
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in eos_token_ids:
            eos_token_ids.append(im_end_id)
        generation_args = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": eos_token_ids,
        }
        if self.temperature > 0:
            generation_args.update({"do_sample": True, "temperature": self.temperature, "top_p": 0.9})
        else:
            generation_args.update({"do_sample": False, "temperature": None, "top_p": None, "top_k": None})
        with self.torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_args)
        text = self.tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        return BackendOutput(text=text, tool_calls=parse_qwen_xml_tool_calls(text))

    def _chat_prompt(self, messages: list[dict[str, Any]], tool_schema: dict[str, Any]) -> str:
        tools = [qwen_tool_schema(tool_schema)]
        if getattr(self.tokenizer, "chat_template", None):
            kwargs = {"tokenize": False, "add_generation_prompt": True, "tools": tools}
            try:
                return self.tokenizer.apply_chat_template(messages, **kwargs, enable_thinking=False)
            except TypeError:
                return self.tokenizer.apply_chat_template(messages, **kwargs)
        rendered = "\n".join(f"{msg['role']}: {msg.get('content', '')}" for msg in messages)
        return f"{rendered}\nassistant:"


class ResponsesAPIBackend(AgentBackend):
    def __init__(self, provider: str, model: str):
        self.client = make_openai_client(provider)
        self.model = model

    def generate(self, messages: list[dict[str, Any]], tool_schema: dict[str, Any]) -> BackendOutput:
        response = create_response_with_retries(
            self.client,
            model=self.model,
            instructions=AGENT_SYSTEM_PROMPT,
            tools=[tool_schema],
            input=messages,
            max_output_tokens=AGENT_GENERATION_TOKEN_LIMIT,
        )
        token_usage = empty_token_usage()
        add_response_usage(token_usage, response)
        text = getattr(response, "output_text", "") or ""
        raw = response.model_dump(mode="json") if hasattr(response, "model_dump") else None
        return BackendOutput(
            text=text,
            tool_calls=parse_responses_tool_calls(response),
            raw=raw,
            token_usage=token_usage,
        )


class LlamaCppServerBackend(AgentBackend):
    def __init__(self, model: str, temperature: float = 0.0):
        self.client = make_openai_client("llama_cpp")
        self.model = model
        self.max_tokens = AGENT_GENERATION_TOKEN_LIMIT
        self.temperature = temperature

    def generate(self, messages: list[dict[str, Any]], tool_schema: dict[str, Any]) -> BackendOutput:
        response = create_chat_completion_with_retries(
            self.client,
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_prompt(tool_schema)},
                *messages,
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        choice = response.choices[0] if response.choices else None
        message = choice.message if choice is not None else None
        text = (getattr(message, "content", None) or "").strip()
        raw = response.model_dump(mode="json") if hasattr(response, "model_dump") else None
        token_usage = empty_token_usage()
        add_chat_completion_usage(token_usage, response)
        return BackendOutput(
            text=text,
            tool_calls=parse_qwen_xml_tool_calls(text),
            raw=raw,
            token_usage=token_usage,
        )

    def _system_prompt(self, tool_schema: dict[str, Any]) -> str:
        schema_text = json_dumps(qwen_tool_schema(tool_schema))
        return (
            f"{AGENT_SYSTEM_PROMPT}\n\n"
            "Tool schema for search_resources:\n"
            f"{schema_text}"
        )


def qwen_tool_schema(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": schema["name"],
            "description": schema.get("description", ""),
            "parameters": schema["parameters"],
        },
    }


def backend_metadata(args) -> dict[str, Any]:
    keys = (
        "backend",
        "provider",
        "model",
        "adapter",
        "agent_temperature",
        "load_in_4bit",
    )
    result = {}
    for key in keys:
        value = getattr(args, key, None)
        if isinstance(value, Path):
            value = str(value)
        result[key] = value
    result["agent_generation_token_limit"] = AGENT_GENERATION_TOKEN_LIMIT
    return result


def make_backend(args) -> AgentBackend:
    if args.backend == "llama_cpp":
        return LlamaCppServerBackend(
            model=args.model,
            temperature=args.agent_temperature,
        )
    if args.backend == "responses":
        return ResponsesAPIBackend(args.provider, args.model)
    if args.backend == "local":
        return LocalHFBackend(
            model_name=args.model,
            adapter=args.adapter,
            temperature=args.agent_temperature,
            load_in_4bit=args.load_in_4bit,
        )
    raise ValueError(f"Unsupported backend: {args.backend}")


def empty_token_usage() -> dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def add_response_usage(total: dict[str, int], response: Any) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        total[key] += int(_usage_attr(usage, key) or 0)


def add_chat_completion_usage(total: dict[str, int], response: Any) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    total["input_tokens"] += int(_usage_attr(usage, "prompt_tokens") or 0)
    total["output_tokens"] += int(_usage_attr(usage, "completion_tokens") or 0)
    total["total_tokens"] += int(_usage_attr(usage, "total_tokens") or 0)


def json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2)


def _usage_attr(usage: Any, name: str) -> Any:
    if isinstance(usage, dict):
        return usage.get(name)
    return getattr(usage, name, None)
