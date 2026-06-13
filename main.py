from __future__ import annotations

from pathlib import Path
from typing import Any

from eval.tool_call_eval import EvalConfig, run as run_tool_call_eval


# Defaults come from EvalConfig. Keep these dicts to the few values a local run
# usually changes, so main.py stays small while remaining the only entrypoint.
LLM_CONFIG: dict[str, Any] = {
    # Agent defaults to local llama.cpp Qwen3.6. For API runs, set
    # backend="responses", provider="openrouter", model="openai/gpt-4.1-mini".
    "backend": "llama_cpp",
    "provider": "openai",
    "model": "qwen3.6-35b-a3b",
    "adapter": None,
    "agent_generation_token_limit": 2048,
    "agent_enable_thinking": False,
    "agent_thinking_budget_tokens": None,
    "agent_temperature": 0.0,
    # User defaults to the same local model. Leave user_model=None to resolve
    # from provider defaults.
    "user_provider": "llama_cpp",
    "user_model": None,
    "user_generation_token_limit": 512,
    "user_enable_thinking": False,
    "user_thinking_budget_tokens": None,
    "user_temperature": 0.0,
}

RUN_CONFIG: dict[str, Any] = {
    "sample_count": 64,
    "sample_seed": 1,
    "sample_progress_every": 0,
    "resources": Path("data/benchmark/filtered_resources_tagged.csv"),
    "output_dir": None,
    "max_agent_turns": 8,
    "user_behaviors": ["normal", "rambling", "impatience", "self_contradictory", "unsupported_request"],
    "user_seed": 7,
    "jobs": 1,
}


PATH_FIELDS = {"adapter", "specs", "resources", "output_dir"}


def main() -> None:
    config = EvalConfig()
    apply_config(config, LLM_CONFIG)
    apply_config(config, RUN_CONFIG)
    run_tool_call_eval(config)


def apply_config(eval_config: EvalConfig, values: dict[str, Any]) -> EvalConfig:
    for key, value in values.items():
        if not hasattr(eval_config, key):
            raise KeyError(f"Unknown eval config key: {key}")
        if key in PATH_FIELDS and value is not None:
            value = Path(value)
        setattr(eval_config, key, value)
    return eval_config


if __name__ == "__main__":
    main()
