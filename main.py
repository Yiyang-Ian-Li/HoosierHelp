from __future__ import annotations

import json
import os
import time

from agent import Agent
from agent.llm import load_dotenv, make_openai_client
from tools.indiana211 import (
    execute_search_resources,
    load_indiana_csv,
    search_resources_tool_schema,
)


CONFIG = {
    "query": "I need a food pantry in Marion County.",
    "index_path": "data/indiana211/indiana211_resources_deduped.csv",
    "provider": "openai",  # "openai" or "openrouter"
    "model": None,
    "limit": 10,
}


def main() -> None:
    load_dotenv()
    config = CONFIG
    index = load_indiana_csv(config["index_path"])
    model = config["model"] or os.getenv("AGENT_MODEL") or default_model(config["provider"])
    client = make_openai_client(config["provider"])

    agent = Agent(
        client=client,
        model=model,
        tools=[search_resources_tool_schema(index)],
        tool_functions={
            "search_resources": lambda args, limit: execute_search_resources(index, args, limit)
        },
    )

    started = time.time()
    response = agent.ask(config["query"], limit=config["limit"])
    elapsed = time.time() - started

    print(f"Loaded {len(index.resources)} resources")
    print(f"Query: {config['query']}")
    print(f"Finished in {elapsed:.1f}s")
    print()
    print(response["output_text"])
    print()
    print("Tool calls:")
    for call in response["tool_calls"]:
        print(json.dumps(call, ensure_ascii=False))


def default_model(provider: str) -> str:
    if provider == "openrouter":
        return "openai/gpt-5"
    return "gpt-5"


if __name__ == "__main__":
    main()
