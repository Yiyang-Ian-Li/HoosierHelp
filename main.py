from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from eval.tool_call_eval import run as run_tool_call_eval


CONFIG = {
    "backend": "responses",  # "local" or "responses"
    "provider": "openai",  # used only by backend="responses"
    # "model": "Qwen/Qwen3-4B-Instruct-2507",
    'model': 'gpt-4.1-mini',
    "adapter": None,
    "specs": "data/benchmark/case_specs.json",
    "resources": "data/benchmark/filtered_resources_tagged.csv",
    "output_dir": None,
    "limit_conversations": 250,
    "max_turns": 8,
    "agent_max_new_tokens": 256,
    "agent_temperature": 0.0,
    "user_provider": "openai",
    "user_model": "gpt-4.1-mini",
    "user_temperature": 0.0,
    "user_max_output_tokens": 256,
    "user_behaviors": ["rambling", "impatience", "self_contradictory", "unsupported_request"],
    "user_seed": 7,
    "jobs": 1,
    "load_in_4bit": True,
}


def main() -> None:
    run_tool_call_eval(eval_args(CONFIG))


def eval_args(config: dict) -> Namespace:
    return Namespace(
        backend=config["backend"],
        provider=config["provider"],
        model=config["model"],
        adapter=Path(config["adapter"]) if config["adapter"] else None,
        specs=Path(config["specs"]),
        resources=Path(config["resources"]),
        output_dir=Path(config["output_dir"]) if config["output_dir"] else None,
        limit_conversations=config["limit_conversations"],
        max_agent_turns=config["max_turns"],
        agent_max_new_tokens=config["agent_max_new_tokens"],
        agent_temperature=config["agent_temperature"],
        user_provider=config["user_provider"],
        user_model=config["user_model"],
        user_temperature=config["user_temperature"],
        user_max_output_tokens=config["user_max_output_tokens"],
        user_behaviors=config["user_behaviors"],
        user_seed=config["user_seed"],
        jobs=config["jobs"],
        load_in_4bit=config["load_in_4bit"],
    )


if __name__ == "__main__":
    main()
