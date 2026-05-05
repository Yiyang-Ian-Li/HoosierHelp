from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import Agent211
from .evaluation import evaluate_cases, load_benchmark_cases
from .index import DEFAULT_RESOURCE_INDEX, load_resource_index
from .llm import load_dotenv, make_openai_client, rerank_with_llm


def main() -> None:
    parser = argparse.ArgumentParser(description="211agent benchmark/search CLI")
    parser.add_argument(
        "--index",
        default=str(DEFAULT_RESOURCE_INDEX),
        help="Path to resource_index JSONL",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask", help="Run one query")
    ask_parser.add_argument("query")
    ask_parser.add_argument("--limit", type=int, default=10)
    ask_parser.add_argument("--json", action="store_true", help="Print raw JSON result")
    _add_agent_options(ask_parser)

    eval_parser = subparsers.add_parser("eval", help="Evaluate benchmark JSONL")
    eval_parser.add_argument("benchmark")
    eval_parser.add_argument("--limit", type=int, default=10)
    eval_parser.add_argument(
        "--use-constraints",
        action="store_true",
        help="Use benchmark metadata constraints as explicit search filters.",
    )
    eval_parser.add_argument("--output", type=Path, help="Optional JSON output path")
    _add_agent_options(eval_parser)

    args = parser.parse_args()
    load_dotenv()
    index = load_resource_index(args.index)
    model = args.model or _default_model(args.provider)
    client = make_openai_client(args.provider) if args.planner == "llm" or args.rerank else None

    reranker = None
    if args.rerank:
        reranker = lambda query, results, limit: rerank_with_llm(
            query, results, client, model, limit=limit
        )

    agent = Agent211(
        index,
        client=client,
        model=model,
        use_openai_tools=args.planner == "llm",
        reranker=reranker,
        retrieval_limit=args.retrieval_limit,
    )

    if args.command == "ask":
        response = agent.ask(args.query, limit=args.limit)
        if args.json:
            print(
                json.dumps(
                    {
                        "query": response.query,
                        "tool_calls": response.tool_calls,
                        "results": [
                            {
                                "resource_id": result.resource.resource_id,
                                "service_name": result.resource.service_name,
                                "agency_name": result.resource.agency_name,
                                "score": result.score,
                                "matched_filters": result.matched_filters,
                            }
                            for result in response.results
                        ],
                    },
                    indent=2,
                )
            )
        else:
            print(response.answer)
        return

    if args.command == "eval":
        cases = load_benchmark_cases(args.benchmark)
        summary = evaluate_cases(
            agent, cases, limit=args.limit, use_constraints=args.use_constraints
        )
        payload = {
            "case_count": summary.case_count,
            "recall_at_1": summary.recall_at_1,
            "recall_at_3": summary.recall_at_3,
            "recall_at_5": summary.recall_at_5,
            "mrr": summary.mrr,
            "records": [record.__dict__ for record in summary.records],
        }
        text = json.dumps(payload, indent=2)
        if args.output:
            args.output.write_text(text, encoding="utf-8")
        print(text)

def _default_model(provider: str) -> str:
    import os

    if os.getenv("AGENT211_MODEL"):
        return os.environ["AGENT211_MODEL"]
    if provider == "openrouter":
        return "openai/gpt-4.1-mini"
    return "gpt-4.1-mini"


def _add_agent_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--planner",
        choices=("llm", "heuristic"),
        default="llm",
        help="Use OpenAI tool calling or a no-network heuristic baseline.",
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "openrouter"),
        default="openrouter",
        help="LLM provider for --planner llm and --rerank.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Planner/reranker model. Defaults to AGENT211_MODEL or gpt-4.1-mini.",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Use the LLM to rerank retrieved candidates.",
    )
    parser.add_argument(
        "--retrieval-limit",
        type=int,
        default=30,
        help="Candidates to retrieve before optional reranking.",
    )


if __name__ == "__main__":
    main()
