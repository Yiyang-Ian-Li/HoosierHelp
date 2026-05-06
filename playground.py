from __future__ import annotations

import argparse
import time
from pathlib import Path

from agent211 import Agent211, load_indiana_csv, load_resource_index
from agent211.llm import load_dotenv, make_openai_client, rerank_with_llm


# Edit these values when you want to try different settings.
DATA_MODE = "full"  # "full" or "curated"
PLANNER = "llm"  # "heuristic" or "llm"
USE_RERANK = False
PROVIDER = "openrouter"  # "openrouter" or "openai"
MODEL = None  # None uses AGENT211_MODEL or the default below.
LIMIT = 5
RETRIEVAL_LIMIT = 30

DEFAULT_QUERY = "I need a food pantry in South Bend."


def main() -> None:
    args = parse_args()
    query = args.query or DEFAULT_QUERY

    root = Path(__file__).resolve().parent
    load_dotenv(root / ".env")

    index = load_index(root, args.data)
    model = args.model or MODEL or default_model(args.provider)

    client = None
    if args.planner == "llm" or args.rerank:
        client = make_openai_client(args.provider)

    reranker = None
    if args.rerank:
        reranker = lambda q, results, limit: rerank_with_llm(
            q, results, client, model, limit=limit
        )

    agent = Agent211(
        index,
        client=client,
        model=model,
        use_openai_tools=args.planner == "llm",
        reranker=reranker,
        retrieval_limit=args.retrieval_limit,
    )

    print(f"Loaded {len(index.resources)} resources from {args.data} data")
    print(f"Planner: {args.planner}; rerank: {args.rerank}; model: {model}")
    print(f"Query: {query}")
    print()

    started = time.time()
    response = agent.ask(query, limit=args.limit)
    print(f"Finished in {time.time() - started:.1f}s")
    print()

    print("TOOL CALL")
    print(response.tool_calls[0] if response.tool_calls else None)
    print()

    print("ANSWER")
    print(response.answer)
    print()

    print("RAW RESULTS")
    for result in response.results:
        r = result.resource
        print(
            f"{result.score:>5} | {r.resource_id} | {r.service_name} | "
            f"{r.agency_name} | {r.city} | {', '.join(r.service_area)}"
        )
        print(f"      matched_filters: {result.matched_filters}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple 211 agent playground")
    parser.add_argument("query", nargs="?", help="User query to test")
    parser.add_argument("--data", choices=("full", "curated"), default=DATA_MODE)
    parser.add_argument("--planner", choices=("heuristic", "llm"), default=PLANNER)
    parser.add_argument("--rerank", action="store_true", default=USE_RERANK)
    parser.add_argument("--provider", choices=("openrouter", "openai"), default=PROVIDER)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--limit", type=int, default=LIMIT)
    parser.add_argument("--retrieval-limit", type=int, default=RETRIEVAL_LIMIT)
    return parser.parse_args()


def load_index(root: Path, data_mode: str):
    if data_mode == "full":
        return load_indiana_csv(root / "data/indiana211/indiana211_resources_deduped.csv")
    return load_resource_index(
        root / "data/indiana211/benchmark_curated/resource_index_curated.jsonl"
    )


def default_model(provider: str) -> str:
    import os

    if os.getenv("AGENT211_MODEL"):
        return os.environ["AGENT211_MODEL"]
    if provider == "openrouter":
        return "openai/gpt-4.1-mini"
    return "gpt-4.1-mini"


if __name__ == "__main__":
    main()
