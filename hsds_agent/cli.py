from __future__ import annotations

import argparse
import json

from .agent import ResourceAgent
from .database import DEFAULT_DB_PATH, connect, initialize, seed_from_json
from .llm import OpenAICompatibleLLM


def main() -> None:
    parser = argparse.ArgumentParser(description="HSDS resource agent prototype")
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB_PATH),
        help="Path to SQLite database",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("seed", help="Initialize and seed the database")

    ask_parser = subparsers.add_parser("ask", help="Ask a resource question")
    ask_parser.add_argument("question")
    ask_parser.add_argument(
        "--debug",
        action="store_true",
        help="Print tool calls and candidate scores",
    )
    ask_parser.add_argument(
        "--provider",
        choices=("openai", "openrouter"),
        default="openrouter",
        help="OpenAI-compatible provider to use",
    )

    args = parser.parse_args()
    conn = connect(DEFAULT_DB_PATH if args.db == str(DEFAULT_DB_PATH) else args.db)
    initialize(conn)

    if args.command == "seed":
        seed_from_json(conn)
        print(f"Seeded database at {args.db}")
        return

    if args.command == "ask":
        llm = OpenAICompatibleLLM.from_env(args.provider)
        answer = ResourceAgent(conn, llm).answer(args.question)
        print(answer.answer)
        if args.debug:
            print("\nTool calls:")
            print(json.dumps(answer.tool_calls, indent=2))
            print("\nCandidate scores:")
            for candidate in answer.candidates:
                print(f"- {candidate.service_id}: {candidate.score}")


if __name__ == "__main__":
    main()
