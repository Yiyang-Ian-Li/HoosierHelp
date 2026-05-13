from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.llm import create_response_with_retries, make_openai_client


USER_SPECS_PATH = Path("data/benchmark/case_specs.json")
USER_CARDS_PATH = Path("data/benchmark/user_cards.json")
PROFILE_MAX_OUTPUT_TOKENS = 900

TRAIT_GROUPS = [
    "normal",
    "incomplete_answer",
    "rambling",
    "impatience",
    "inconsistency",
    "unreasonable_demand",
]

CARD_INSTRUCTIONS = """
You create hidden simulated-user cards for an Indiana 211 resource-search benchmark.

Convert the deterministic user spec into a realistic hidden user card. Return
strict JSON only. 

Rules:
- Create one plain-language `need_summary` for each listed need. Do not use
  benchmark service category labels verbatim. For example, say "help getting
  groceries" instead of "Food", or "someone to explain local programs I might
  qualify for" instead of "Information and Referral".
- `opening` must be one natural first message from the user. It must mention all
  service needs in plain language. It must not reveal all location and schedule
  constraints.
- `trait_openings.unreasonable_demand` must be an alternate first message for
  the unreasonable_demand trait. It must mention all underlying service needs
  but exaggerate them into unrealistic demands. Do not add unrelated needs.
- `profile` should describe a coherent, vivid, realistic person whose situation
  naturally fits the spec. Add concrete background details that align with the
  constraints and the plain-language need summaries. Do not add new needs or
  constraints that could change the matching resource. Do not use benchmark
  service category labels verbatim in the profile.
- `location` always contains county, city, state, and ZIP for background. Only
  the location field present in `constraints` is the location constraint the
  user needs satisfied.
- Treat constraints as fixed truth. Location and schedule facts are firm
  requirements. Every service need has its own schedule requirement.
- Do not soften firm constraints with words like "prefers", "would like", or
  "ideally" in the hidden profile.
- Do not mention resource IDs, agency names, site names, or specific resource names.

Return exactly this JSON shape:
{
  "opening": "one natural first message that mentions all service needs",
  "trait_openings": {
    "unreasonable_demand": "alternate first message for the unreasonable_demand trait"
  },
  "needs": [
    {
      "need_id": "need_1",
      "need_summary": "plain-language description of this need"
    }
  ],
  "profile": "120-180 word hidden profile in natural language"
}
""".strip()


def main() -> None:
    args = parse_args()
    specs = json.loads(args.case_specs.read_text(encoding="utf-8"))
    if args.limit:
        specs = specs[: args.limit]
    rng = random.Random(args.seed)
    traits_by_case = balanced_traits_by_case(specs, rng)
    cards = build_user_cards(specs, traits_by_case, args)
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


def balanced_traits_by_case(specs: list[dict], rng: random.Random) -> dict[str, list[str]]:
    traits_by_case = {}
    for case_type in sorted({spec.get("case_type", "") for spec in specs}):
        group = [spec for spec in specs if spec.get("case_type", "") == case_type]
        assignments = [
            TRAIT_GROUPS[index % len(TRAIT_GROUPS)]
            for index in range(len(group))
        ]
        rng.shuffle(assignments)
        for spec, trait in zip(group, assignments):
            traits_by_case[spec["case_id"]] = [trait]
    return traits_by_case


def build_user_card(client, model: str, spec: dict, traits: list[str]) -> dict:
    generated = generate_profile(client, model, spec)
    card = {
        "case_id": spec["case_id"],
        "user_id": spec["case_id"],
        "case_type": spec["case_type"],
        "traits": traits,
        "opening": opening_for_traits(generated, traits),
        "default_opening": generated["opening"],
        "trait_openings": generated["trait_openings"],
        "profile": generated["profile"],
        "location": spec["location"],
        "location_requirement": spec["location_requirement"],
        "needs": user_visible_needs(spec, generated),
        "target_service_categories": spec["target_service_categories"],
        "ground_truth_resource_ids": spec["ground_truth_resource_ids"],
        "ground_truth_resources": spec["ground_truth_resources"],
        "case_spec": {
            "location": spec["location"],
            "location_requirement": spec["location_requirement"],
            "needs": spec["needs"],
        },
    }
    return card


def build_user_card_for_spec(provider: str, model: str, spec: dict, traits: list[str]) -> dict:
    return build_user_card(make_openai_client(provider), model, spec, traits)


def generate_profile(client, model: str, spec: dict) -> dict:
    response = create_response_with_retries(
        client,
        model=model,
        instructions=CARD_INSTRUCTIONS,
        input=[
            {
                "role": "user",
                "content": json.dumps(llm_visible_spec(spec), ensure_ascii=False),
            }
        ],
        max_output_tokens=PROFILE_MAX_OUTPUT_TOKENS,
    )
    generated = validated_generated_profile(getattr(response, "output_text", "") or "")
    validate_generated_needs(spec, generated)
    return generated


def llm_visible_spec(spec: dict) -> dict:
    visible = {
        "case_id": spec["case_id"],
        "case_type": spec["case_type"],
        "location": spec["location"],
        "constraints": {
            "location_requirement": spec["location_requirement"],
            "needs": [
                {
                    "need_id": need["need_id"],
                    "benchmark_service_categories_do_not_copy": need["service_categories"],
                    "schedule": need["schedule"],
                }
                for need in spec["needs"]
            ],
        },
    }
    return visible


def user_visible_needs(spec: dict, generated: dict) -> list[dict]:
    summaries = {need["need_id"]: need["need_summary"] for need in generated["needs"]}
    return [
        {
            "need_id": need["need_id"],
            "need_summary": summaries[need["need_id"]],
            "schedule": copy_requirement(need["schedule"]),
        }
        for need in spec["needs"]
    ]


def copy_requirement(value):
    if isinstance(value, dict):
        return {key: copy_requirement(item) for key, item in value.items()}
    if isinstance(value, list):
        return [copy_requirement(item) for item in value]
    return value


def json_object(value: str) -> dict:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LLM did not return valid JSON: {value[:500]}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("LLM JSON output must be an object.")
    return parsed


def validated_generated_profile(value: str) -> dict:
    parsed = json_object(value)
    for key in ("opening", "profile"):
        if not isinstance(parsed.get(key), str) or not parsed[key].strip():
            raise RuntimeError(f"LLM JSON output missing non-empty `{key}`.")
        parsed[key] = parsed[key].strip()
    needs = parsed.get("needs")
    if not isinstance(needs, list) or not needs:
        raise RuntimeError("LLM JSON output missing non-empty `needs` list.")
    normalized_needs = []
    for item in needs:
        if not isinstance(item, dict):
            raise RuntimeError("LLM `needs` entries must be objects.")
        need_id = item.get("need_id")
        summary = item.get("need_summary")
        if not isinstance(need_id, str) or not need_id.strip():
            raise RuntimeError("LLM `needs` entry missing non-empty `need_id`.")
        if not isinstance(summary, str) or not summary.strip():
            raise RuntimeError("LLM `needs` entry missing non-empty `need_summary`.")
        normalized_needs.append(
            {
                "need_id": need_id.strip(),
                "need_summary": summary.strip(),
            }
        )
    trait_openings = parsed.get("trait_openings")
    if not isinstance(trait_openings, dict):
        raise RuntimeError("LLM JSON output missing `trait_openings` object.")
    unreasonable = trait_openings.get("unreasonable_demand")
    if not isinstance(unreasonable, str) or not unreasonable.strip():
        raise RuntimeError("LLM JSON output missing non-empty unreasonable_demand trait opening.")
    return {
        "opening": parsed["opening"],
        "trait_openings": {"unreasonable_demand": unreasonable.strip()},
        "needs": normalized_needs,
        "profile": parsed["profile"],
    }


def validate_generated_needs(spec: dict, generated: dict) -> None:
    expected = [need["need_id"] for need in spec["needs"]]
    actual = [need["need_id"] for need in generated["needs"]]
    if actual != expected:
        raise RuntimeError(f"LLM need ids must match spec need ids exactly: expected={expected} actual={actual}")


def opening_for_traits(generated: dict, traits: list[str]) -> str:
    if "unreasonable_demand" in traits:
        return generated["trait_openings"]["unreasonable_demand"]
    return generated["opening"]


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LLM-generated simulated-user cards from case specs.")
    parser.add_argument("--case-specs", type=Path, default=USER_SPECS_PATH)
    parser.add_argument("--users-out", type=Path, default=USER_CARDS_PATH)
    parser.add_argument("--provider", default="openrouter")
    parser.add_argument("--model", default="openai/gpt-4.1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    args.jobs = max(1, args.jobs)
    return args


if __name__ == "__main__":
    main()
