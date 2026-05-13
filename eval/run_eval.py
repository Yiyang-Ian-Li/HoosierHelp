from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import Agent
from agent.llm import load_dotenv, make_openai_client
from eval.analyze_run import analyze_run_dir
from eval.agent_instructions import agent_instructions
from eval.metrics import (
    count_function_calls,
    parse_final_json as parse_recommendation_json,
    parse_final_json_mode,
    parse_strict_final_json,
    recommended_ids_from_final_json as metric_recommended_ids_from_final_json,
)
from eval.simulated_user import LLMSimulatedUser
from tools.indiana211 import execute_search_resources, load_resource_index, search_resources_tool_schema


DEFAULT_USERS = Path("data/benchmark/user_cards.json")
DEFAULT_OUTPUT_DIR = Path("experiments")
MAX_TOOL_CALLS = 3


def run(args: argparse.Namespace) -> Path:
    load_dotenv()
    users = json.loads(args.users.read_text(encoding="utf-8"))
    if args.case_type != "all":
        users = [card for card in users if card.get("case_type") == args.case_type]
    ground_truth_by_case = load_ground_truth_from_cards(users)
    if args.limit_users:
        users = users[: args.limit_users]
    index = load_resource_index(args.index_path)
    run_id = experiment_name(args, len(users))
    output_dir = args.output_dir / run_id
    conversations_dir = output_dir / "conversations"
    output_dir.mkdir(parents=True, exist_ok=True)
    conversations_dir.mkdir(parents=True, exist_ok=True)
    print(
        "Starting eval: "
        f"provider={args.provider} "
        f"agent_model={args.agent_model} "
        f"user_model={args.user_model} "
        f"users={args.users} "
        f"cases={len(users)} "
        f"jobs={args.jobs} "
        f"output={output_dir}",
        flush=True,
    )
    cases = []
    if args.jobs == 1:
        for idx, card in enumerate(users, start=1):
            print(f"[{idx}/{len(users)}] start {card['user_id']} {card_label(card, ground_truth_by_case)}", flush=True)
            case = run_case_for_card(args, index, card, ground_truth_by_case[case_id(card)])
            print(
                f"[{idx}/{len(users)}] done {card['user_id']} "
                f"stop={case['stop_reason']} "
                f"turns={case_turn_count(case)} tools={case_tool_call_count(case)}",
                flush=True,
            )
            cases.append(case)
            write_json(conversations_dir / f"{card['user_id']}.json", case)
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            for idx, card in enumerate(users, start=1):
                print(f"[{idx}/{len(users)}] queued {card['user_id']} {card_label(card, ground_truth_by_case)}", flush=True)
            futures = {
                executor.submit(run_case_for_card, args, index, card, ground_truth_by_case[case_id(card)]): (
                    idx,
                    card,
                )
                for idx, card in enumerate(users, start=1)
            }
            for future in as_completed(futures):
                idx, card = futures[future]
                case = future.result()
                print(
                    f"[{idx}/{len(users)}] done {card['user_id']} "
                    f"stop={case['stop_reason']} "
                    f"turns={case_turn_count(case)} tools={case_tool_call_count(case)}",
                    flush=True,
                )
                cases.append(case)
                write_json(conversations_dir / f"{card['user_id']}.json", case)
    cases.sort(key=lambda case: case["card"]["user_id"])
    run_metadata = {
        "provider": args.provider,
        "agent_type": args.agent_type,
        "agent_model": args.agent_model,
        "user_model": args.user_model,
        "case_type": args.case_type,
        "users": str(args.users),
        "index_path": str(args.index_path),
        "jobs": args.jobs,
        "max_turns": args.max_turns,
    }
    write_json(output_dir / "run.json", run_metadata)
    print(f"Wrote {output_dir / 'run.json'}")
    analyze_run_dir(output_dir)
    return output_dir


def run_case_for_card(args, index, card: dict, ground_truth: dict) -> dict:
    client = make_openai_client(args.provider)
    tools = [search_resources_tool_schema(index)]
    agent = Agent(
        client=client,
        model=args.agent_model,
        tools=tools,
        tool_functions={"search_resources": lambda tool_args, limit: execute_search_resources(index, tool_args)},
        instructions=agent_instructions(args.agent_type),
        max_tool_calls=MAX_TOOL_CALLS,
    )
    simulated_user = LLMSimulatedUser(card, make_openai_client(args.provider), args.user_model)
    return run_case(agent, simulated_user, card, ground_truth, args.max_turns)




def load_ground_truth_from_cards(users: list[dict]) -> dict[str, dict]:
    return {
        case_id(card): {
            "case_id": case_id(card),
            "user_id": card.get("user_id", case_id(card)),
            "ground_truth_resource_ids": card.get("ground_truth_resource_ids", []),
            "ground_truth_resources": card.get("ground_truth_resources", []),
            "target_service_categories": card.get("target_service_categories", []),
            "location_requirement": card.get("location_requirement") or card.get("case_spec", {}).get("location_requirement", {}),
            "needs": card.get("case_spec", {}).get("needs", []),
        }
        for card in users
    }


def case_id(card: dict) -> str:
    return card.get("case_id") or card["user_id"].split("__", 1)[0]


def card_label(card: dict, ground_truth_by_case: dict) -> str:
    categories = card.get("target_service_categories")
    if not categories:
        ground_truth = ground_truth_by_case.get(case_id(card), {})
        categories = ground_truth.get("target_service_categories") or []
    return categories[0] if categories else "-"


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
        structured_result = parse_agent_result(agent_text)
        final_response["structured_result"] = structured_result
        completed = structured_result["completed"]
        transcript.append({"role": "agent", "content": agent_text})
        history = final_response["input"]
        if completed:
            stop_reason = structured_result["status"]
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
        "simulated_user_diagnostics": getattr(simulated_user, "diagnostics", lambda: {})(),
        "token_usage": token_usage,
    }


def case_turn_count(case: dict) -> int:
    return len([turn for turn in case.get("transcript", []) if turn.get("role") == "user"])


def case_tool_call_count(case: dict) -> int:
    return count_function_calls(case.get("response", {}))


def completion_flag(agent_text: str) -> bool:
    return parse_agent_result(agent_text).get("completed") is True


def parse_agent_result(agent_text: str) -> dict:
    parsed = parse_recommendation_json(agent_text)
    if parsed is None:
        return _agent_result("continue", final_json_valid=False)
    recommended = metric_recommended_ids_from_final_json(parsed)
    strict_valid = parse_strict_final_json(agent_text) is not None
    parse_mode = parse_final_json_mode(agent_text)
    if parsed.get("recommendations"):
        return _agent_result(
            "recommended",
            recommended,
            final_json=parsed,
            final_json_valid=True,
            final_json_strict_valid=strict_valid,
            final_json_parse_mode=parse_mode,
        )
    return _agent_result(
        "no_match",
        final_json=parsed,
        final_json_valid=True,
        final_json_strict_valid=strict_valid,
        final_json_parse_mode=parse_mode,
    )


def _agent_result(
    status: str,
    recommended: list[str] | None = None,
    final_json: dict | None = None,
    final_json_valid: bool = False,
    final_json_strict_valid: bool = False,
    final_json_parse_mode: str = "none",
) -> dict:
    recommended = recommended or []
    return {
        "status": status,
        "completed": status in {"recommended", "no_match"},
        "recommended_resource_ids": recommended,
        "final_json": final_json,
        "final_json_valid": final_json_valid,
        "final_json_strict_valid": final_json_strict_valid,
        "final_json_parse_mode": final_json_parse_mode,
    }


def add_token_usage(total: dict, item: dict) -> None:
    for key in ["input_tokens", "output_tokens", "total_tokens"]:
        total[key] = total.get(key, 0) + int(item.get(key, 0) or 0)


def combine_token_usage(*items: dict) -> dict:
    total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for item in items:
        add_token_usage(total, item)
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simulated-user evaluation.")
    parser.add_argument("--provider", default="openai", choices=["openai", "openrouter"])
    parser.add_argument("--agent-model", default=os.getenv("AGENT_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--model", dest="agent_model")
    parser.add_argument("--user-model", default=os.getenv("SIM_USER_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--sim-user-model", dest="user_model")
    parser.add_argument("--index-path", type=Path, default=Path("data/benchmark/filtered_resources_tagged.csv"))
    parser.add_argument("--users", type=Path, default=DEFAULT_USERS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit-users", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--agent-type", choices=["default", "react"], default="default")
    parser.add_argument("--case-type", choices=["all", "single", "composite"], default="all")
    parser.add_argument("--jobs", type=int, default=8)
    return parser.parse_args()


def experiment_name(args: argparse.Namespace, case_count: int) -> str:
    timestamp = time.strftime("%Y%m%d-%H%M")
    agent_part = f"{slug(args.agent_type)}-{slug(args.agent_model)}"
    user_part = f"{user_set_name(args.users)}-{slug(args.user_model)}"
    case_type_part = "" if args.case_type == "all" else f"_{args.case_type}"
    return f"{timestamp}_{agent_part}_{user_part}{case_type_part}_n{case_count}"


def user_set_name(users_path: Path) -> str:
    name = users_path.name.lower()
    if "single_noncollab" in name or "noncollab" in name:
        return "noncollab"
    if "normal" in name:
        return "normal"
    if "all_variants" in name:
        return "mixed"
    return "base"


def slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")


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


if __name__ == "__main__":
    run(parse_args())
