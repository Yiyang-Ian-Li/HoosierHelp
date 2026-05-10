from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import AGENT_INSTRUCTIONS, REACT_INSTRUCTIONS, Agent
from agent.llm import load_dotenv, make_openai_client
from eval.metrics import aggregate, score_case
from eval.simulated_user import LLMSimulatedUser
from tools.indiana211 import execute_search_resources, load_indiana_csv, search_resources_tool_schema


DEFAULT_USERS = Path("data/benchmark/user_cards.json")
DEFAULT_GROUND_TRUTH = Path("data/benchmark/ground_truth.json")
DEFAULT_OUTPUT_DIR = Path("experiments")


def run(args: argparse.Namespace) -> Path:
    load_dotenv()
    users = json.loads(args.users.read_text(encoding="utf-8"))
    ground_truth = {
        item["user_id"]: item
        for item in json.loads(args.ground_truth.read_text(encoding="utf-8"))
    }
    if args.limit_users:
        users = users[: args.limit_users]
    index = load_indiana_csv(args.index_path)
    tools = [search_resources_tool_schema(index)]

    run_id = experiment_name(args, len(users))
    output_dir = args.output_dir / run_id
    conversations_dir = output_dir / "conversations"
    output_dir.mkdir(parents=True, exist_ok=True)
    conversations_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    scores = []
    if args.jobs == 1:
        for idx, card in enumerate(users, start=1):
            print(f"[{idx}/{len(users)}] {card['user_id']} {card['target_service_categories'][0]}")
            case = run_case_for_card(args, index, tools, card, ground_truth[card["user_id"]])
            cases.append(case)
            scores.append(case["score"])
            write_json(conversations_dir / f"{card['user_id']}.json", case)
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(run_case_for_card, args, index, tools, card, ground_truth[card["user_id"]]): (
                    idx,
                    card,
                )
                for idx, card in enumerate(users, start=1)
            }
            for future in as_completed(futures):
                idx, card = futures[future]
                case = future.result()
                print(f"[{idx}/{len(users)}] {card['user_id']} {card['target_service_categories'][0]}")
                cases.append(case)
                scores.append(case["score"])
                write_json(conversations_dir / f"{card['user_id']}.json", case)
    cases.sort(key=lambda case: case["card"]["user_id"])
    scores = [case["score"] for case in cases]
    summary = {
        "provider": args.provider,
        "model": args.agent_model,
        "agent_type": args.agent_type,
        "agent_model": args.agent_model,
        "user_type": "llm",
        "user_model": args.user_model,
        "sim_user": "llm",
        "sim_user_model": args.user_model,
        "jobs": args.jobs,
        "limit": "model_selected",
        "max_turns": args.max_turns,
        "completed_cases": sum(1 for case in cases if case["completed"]),
        "stop_reasons": {
            reason: sum(1 for case in cases if case["stop_reason"] == reason)
            for reason in sorted({case["stop_reason"] for case in cases})
        },
        "token_usage": aggregate_token_usage(cases),
        "summary": aggregate(scores),
        "scores": scores,
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    print(f"Wrote {output_dir / 'summary.json'}")
    print(f"Wrote {output_dir / 'report.md'}")
    return output_dir


def run_case_for_card(args, index, tools: list[dict], card: dict, ground_truth: dict) -> dict:
    client = make_openai_client(args.provider)
    agent = Agent(
        client=client,
        model=args.agent_model,
        tools=tools,
        tool_functions={"search_resources": lambda tool_args, limit: execute_search_resources(index, tool_args)},
        instructions=agent_instructions(args.agent_type),
    )
    simulated_user = LLMSimulatedUser(card, make_openai_client(args.provider), args.user_model)
    return run_case(agent, simulated_user, card, ground_truth, args.max_turns)


def run_case(agent: Agent, simulated_user, card: dict, ground_truth: dict, max_turns: int) -> dict:
    history = []
    transcript = []
    user_message = simulated_user.opening()
    final_response = {}
    completed = False
    stop_reason = "max_turns"
    agent_token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for _ in range(max_turns):
        transcript.append({"role": "user", "content": user_message})
        final_response = agent.ask(user_message, history=history, limit=None)
        add_token_usage(agent_token_usage, final_response.get("token_usage", {}))
        agent_text = final_response["output_text"]
        completed = completion_flag(agent_text)
        transcript.append({"role": "agent", "content": agent_text})
        history = final_response["input"]
        if completed:
            stop_reason = "completed"
            break
        next_user_message = simulated_user.respond(agent_text)
        if not next_user_message:
            stop_reason = "simulated_user_stopped"
            break
        user_message = next_user_message
    token_usage = {
        "agent": agent_token_usage,
        "simulated_user": getattr(simulated_user, "token_usage", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}),
    }
    token_usage["total"] = combine_token_usage(token_usage["agent"], token_usage["simulated_user"])
    return {
        "card": card,
        "ground_truth": ground_truth,
        "transcript": transcript,
        "response": final_response,
        "completed": completed,
        "stop_reason": stop_reason,
        "token_usage": token_usage,
        "score": score_case(card, ground_truth, transcript, final_response),
    }


def completion_flag(agent_text: str) -> bool:
    lines = [line.strip().lower() for line in agent_text.splitlines() if line.strip()]
    if not lines:
        return False
    return lines[-1] == "completed: true"


def add_token_usage(total: dict, item: dict) -> None:
    for key in ["input_tokens", "output_tokens", "total_tokens"]:
        total[key] = total.get(key, 0) + int(item.get(key, 0) or 0)


def combine_token_usage(*items: dict) -> dict:
    total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for item in items:
        add_token_usage(total, item)
    return total


def aggregate_token_usage(cases: list[dict]) -> dict:
    total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    agent = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    simulated_user = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for case in cases:
        usage = case.get("token_usage", {})
        add_token_usage(total, usage.get("total", {}))
        add_token_usage(agent, usage.get("agent", {}))
        add_token_usage(simulated_user, usage.get("simulated_user", {}))
    count = len(cases) or 1
    return {
        "total": total,
        "agent": agent,
        "simulated_user": simulated_user,
        "average_per_case": {
            key: value / count
            for key, value in total.items()
        },
        "average_agent_per_case": {
            key: value / count
            for key, value in agent.items()
        },
        "average_simulated_user_per_case": {
            key: value / count
            for key, value in simulated_user.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simulated-user evaluation.")
    parser.add_argument("--provider", default="openai", choices=["openai", "openrouter"])
    parser.add_argument("--agent-model", default=os.getenv("AGENT_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--model", dest="agent_model")
    parser.add_argument("--user-model", default=os.getenv("SIM_USER_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--sim-user-model", dest="user_model")
    parser.add_argument("--index-path", type=Path, default=Path("data/indiana211/indiana211_resources_deduped.csv"))
    parser.add_argument("--users", type=Path, default=DEFAULT_USERS)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit-users", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--agent-type", choices=["default", "react"], default="default")
    parser.add_argument("--jobs", type=int, default=8)
    return parser.parse_args()


def experiment_name(args: argparse.Namespace, case_count: int) -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return "__".join(
        [
            timestamp,
            f"agent-{slug(args.agent_type)}",
            f"agentmodel-{slug(args.agent_model)}",
            f"usermodel-{slug(args.user_model)}",
            f"n{case_count}",
        ]
    )


def slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")


def agent_instructions(agent_type: str) -> str:
    if agent_type == "react":
        return REACT_INSTRUCTIONS
    return AGENT_INSTRUCTIONS


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def jsonable(value):
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump())
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return jsonable(value.__dict__)
    return value


def render_report(summary: dict) -> str:
    agg = summary["summary"]
    lines = [
        "# Simulated User Evaluation Report",
        "",
        f"- Provider: {summary['provider']}",
        f"- Agent type: {summary.get('agent_type')}",
        f"- Agent model: {summary.get('agent_model', summary['model'])}",
        f"- User model: {summary.get('user_model')}",
        f"- Cases: {agg.get('cases', 0)}",
        f"- Primary hit rate: {agg.get('primary_hit_rate', 0):.2%}",
        f"- Acceptable hit rate: {agg.get('acceptable_hit_rate', 0):.2%}",
        f"- Retrieval primary hit rate: {agg.get('retrieval_primary_hit_rate', 0):.2%}",
        f"- Retrieval acceptable hit rate: {agg.get('retrieval_acceptable_hit_rate', 0):.2%}",
        f"- Average tool calls: {agg.get('average_tool_calls', 0):.2f}",
        f"- Average clarification score: {agg.get('average_clarification_score', 0):.2f}/4",
        f"- Average total tokens per case: {summary.get('token_usage', {}).get('average_per_case', {}).get('total_tokens', 0):.0f}",
        f"- Completed cases: {summary.get('completed_cases', 0)}",
        "",
        "## By Difficulty",
        "",
    ]
    for difficulty, item in agg.get("by_difficulty", {}).items():
        lines.extend(
            [
                f"### {difficulty}",
                "",
                f"- Cases: {item['cases']}",
                f"- Primary hit rate: {item['primary_hit_rate']:.2%}",
                f"- Acceptable hit rate: {item['acceptable_hit_rate']:.2%}",
                f"- Retrieval primary hit rate: {item['retrieval_primary_hit_rate']:.2%}",
                f"- Retrieval acceptable hit rate: {item['retrieval_acceptable_hit_rate']:.2%}",
                f"- Average clarification score: {item['average_clarification_score']:.2f}/4",
                "",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    run(parse_args())
