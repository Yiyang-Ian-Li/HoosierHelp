from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.llm import make_openai_client


CASE_SPECS_PATH = Path("data/benchmark/case_specs.json")
USER_CARDS_PATH = Path("data/benchmark/user_cards.json")

TRAIT_POOL = [
    "",
    "",
    "",
    "incomplete_answer",
    "rambling",
    "impatience",
    "inconsistency",
    "unreasonable_demand",
]

CARD_INSTRUCTIONS = """
You create hidden simulated-user cards for an Indiana 211 resource-search
benchmark.

Convert the given deterministic case spec into a realistic hidden user profile.
Return strict JSON only. Do not include markdown.

Rules:
- The user-facing need should come only from `user_requirements`.
- Fee, document, and eligibility information from `user_qualification` are facts
  about the user, not stated needs. The simulated user should reveal them only if
  asked or naturally relevant.
- Write `profile` as a coherent hidden background for a real person, not a list
  of tool filters. Add concrete everyday context that makes the situation feel
  realistic and coherent.
- Any added background detail must be related to and consistent with the case
  spec. It should echo the existing location, need, access method, and user
  qualification facts, without adding a new requirement or changing which
  resource should match.
- Keep the whole card internally consistent. `need_summary`, `known_facts`, and
  `answering_guidance` must agree with the richer `profile`.

Return this JSON shape:
{
  "profile": "120-180 word hidden profile in natural language",
  "need_summary": "one sentence summary of the user's need",
  "known_facts": ["fact", "fact"],
  "answering_guidance": ["how the user should answer if asked about missing info"]
}
""".strip()


def main() -> None:
    args = parse_args()
    specs = json.loads(args.case_specs.read_text(encoding="utf-8"))
    if args.limit:
        specs = specs[: args.limit]
    rng = random.Random(args.seed)
    traits_by_case = {
        spec["case_id"]: ([trait] if (trait := rng.choice(TRAIT_POOL)) else [])
        for spec in specs
    }
    cards = build_user_cards(specs, traits_by_case, args)
    validate_cards(cards)
    write_json(args.users_out, cards)
    print(f"Wrote {args.users_out}")


def build_user_cards(specs: list[dict], traits_by_case: dict[str, list[str]], args: argparse.Namespace) -> list[dict]:
    if args.jobs == 1:
        return [
            build_user_card_for_spec(args.provider, args.model, spec, traits_by_case[spec["case_id"]])
            for spec in print_specs(specs)
        ]

    cards_by_case = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                build_user_card_for_spec,
                args.provider,
                args.model,
                spec,
                traits_by_case[spec["case_id"]],
            ): (idx, spec)
            for idx, spec in enumerate(specs, start=1)
        }
        for future in as_completed(futures):
            idx, spec = futures[future]
            card = future.result()
            cards_by_case[spec["case_id"]] = card
            print(f"[{idx}/{len(specs)}] done {spec['case_id']}", flush=True)
    return [cards_by_case[spec["case_id"]] for spec in specs]


def print_specs(specs: list[dict]):
    total = len(specs)
    for idx, spec in enumerate(specs, start=1):
        print(f"[{idx}/{total}] {spec['case_id']}", flush=True)
        yield spec


def build_user_card(client, model: str, spec: dict, traits: list[str]) -> dict:
    generated = generate_profile(client, model, spec)
    card = {
        "case_id": spec["case_id"],
        "user_id": spec["case_id"],
        "difficulty": spec["difficulty"],
        "target_service_categories": spec["target_service_categories"],
        "ground_truth_resource_ids": spec["ground_truth_resource_ids"],
        "traits": traits,
        "profile": generated["profile"],
        "need_summary": generated["need_summary"],
        "known_facts": generated.get("known_facts", []),
        "answering_guidance": generated.get("answering_guidance", []),
        "case_spec": {
            "location": spec["location"],
            "user_requirements": spec["user_requirements"],
            "user_qualification": spec["user_qualification"],
        },
    }
    return card


def build_user_card_for_spec(provider: str, model: str, spec: dict, traits: list[str]) -> dict:
    return build_user_card(make_openai_client(provider), model, spec, traits)


def generate_profile(client, model: str, spec: dict) -> dict:
    response = client.responses.create(
        model=model,
        instructions=CARD_INSTRUCTIONS,
        input=[
            {
                "role": "user",
                "content": json.dumps(llm_visible_spec(spec), ensure_ascii=False),
            }
        ],
    )
    return json_object(getattr(response, "output_text", "") or "")


def llm_visible_spec(spec: dict) -> dict:
    return {
        "case_id": spec["case_id"],
        "difficulty": spec["difficulty"],
        "need_category": spec["need_category"],
        "location": spec["location"],
        "user_requirements": spec["user_requirements"],
        "user_qualification": spec["user_qualification"],
    }


def validate_cards(cards: list[dict]) -> None:
    for card in cards:
        text = json.dumps(
            {
                "profile": card.get("profile"),
                "need_summary": card.get("need_summary"),
                "known_facts": card.get("known_facts"),
                "answering_guidance": card.get("answering_guidance"),
            },
            ensure_ascii=False,
        ).lower()
        if re.search(r"\bin211-[a-z0-9]+(?:-[a-z0-9]+)*\b", text):
            raise RuntimeError(f"resource id leaked into user card: {card['case_id']}")
        if "opening" in card:
            raise RuntimeError(f"user card must not include opening: {card['case_id']}")
        if not card.get("profile"):
            raise RuntimeError(f"missing profile: {card['case_id']}")
        if not card.get("ground_truth_resource_ids"):
            raise RuntimeError(f"missing ground_truth_resource_ids: {card['case_id']}")


def json_object(value: str) -> dict:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LLM did not return valid JSON: {value[:500]}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("LLM JSON output must be an object.")
    return parsed


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LLM-generated simulated-user cards from case specs.")
    parser.add_argument("--case-specs", type=Path, default=CASE_SPECS_PATH)
    parser.add_argument("--users-out", type=Path, default=USER_CARDS_PATH)
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    args.jobs = max(1, args.jobs)
    return args


if __name__ == "__main__":
    main()
