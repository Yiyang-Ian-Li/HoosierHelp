from __future__ import annotations

import json
import time

from agent import Agent
from agent.llm import load_dotenv, make_openai_client
from eval.agent_instructions import agent_instructions
from tools.indiana211 import (
    execute_search_resources,
    load_resource_index,
    search_resources_tool_schema,
)


CONFIG = {
    "provider": "openai",
    "index_path": "data/benchmark/filtered_resources_tagged.csv",
    "agent_type": "default",  # "default" or "react"
    "agent_model": "gpt-4.1-mini",
    "query": "I need a food pantry in Marion County.",
    "limit": 10,
}


def main() -> None:
    load_dotenv()
    index = load_resource_index(CONFIG["index_path"])
    client = make_openai_client(CONFIG["provider"])
    agent = Agent(
        client=client,
        model=CONFIG["agent_model"],
        tools=[search_resources_tool_schema(index)],
        tool_functions={
            "search_resources": lambda args, limit: execute_search_resources(index, args, limit)
        },
        instructions=agent_instructions(CONFIG["agent_type"]),
    )

    started = time.time()
    response = agent.ask(CONFIG["query"], limit=CONFIG["limit"])
    elapsed = time.time() - started

    print(f"Loaded {len(index.resources)} resources")
    print(f"Agent type: {CONFIG['agent_type']}")
    print(f"Agent model: {CONFIG['agent_model']}")
    print(f"Query: {CONFIG['query']}")
    print(f"Finished in {elapsed:.1f}s")
    print()
    print(response["output_text"])
    print()
    print("Tool calls:")
    for call in response["tool_calls"]:
        print(json.dumps(call, ensure_ascii=False))


if __name__ == "__main__":
    main()
