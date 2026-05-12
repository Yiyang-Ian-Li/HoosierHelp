from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from agent.llm import load_dotenv
from eval.run_eval import run as run_eval


CONFIG = {
    "provider": "openai",
    "index_path": "data/benchmark/filtered_resources_tagged.csv",
    "agent_type": "default",  # "default" or "react"
    "agent_model": "gpt-4.1-mini",
    "user_model": "gpt-4.1-mini",
    "users": "data/benchmark/user_cards.json",
    "difficulty": "all",
    "output_dir": "experiments",
    "limit_users": 0,
    "max_turns": 8,
    "jobs": 8,
}


def main() -> None:
    load_dotenv()
    run_eval(eval_args(CONFIG))


def eval_args(config: dict) -> Namespace:
    return Namespace(
        provider=config["provider"],
        agent_type=config["agent_type"],
        agent_model=config["agent_model"],
        user_model=config["user_model"],
        index_path=Path(config["index_path"]),
        users=Path(config["users"]),
        difficulty=config["difficulty"],
        output_dir=Path(config["output_dir"]),
        limit_users=config["limit_users"],
        max_turns=config["max_turns"],
        jobs=config["jobs"],
    )


if __name__ == "__main__":
    main()
