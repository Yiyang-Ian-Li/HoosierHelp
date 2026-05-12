from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.llm import make_openai_client


USER_SPECS_PATH = Path("data/benchmark/case_specs.json")
USER_CARDS_PATH = Path("data/benchmark/user_cards.json")
MAX_PROFILE_GENERATION_ATTEMPTS = 4
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
strict JSON only. Do not include markdown.

Rules:
- `need_summary` should describe the user's service need in plain help-seeker
  language. Do not use benchmark service category labels verbatim. For example,
  say "help getting groceries" instead of "Food", or "someone to explain local
  programs I might qualify for" instead of "Information and Referral".
- `profile` should describe a coherent, vivid, realistic person whose situation
  naturally fits the spec. Add concrete background details that align with the
  known facts and the plain-language need summary. Do not add new needs or
  constraints that could change the matching resource. Do not use benchmark
  service category labels verbatim in the profile.
- `location` always contains county, city, state, and ZIP for background. Only
  the location field present in `constraints` is the location constraint the
  user needs satisfied.
- Treat constraints as fixed truth. Location and intake facts are firm
  requirements. If a schedule fact is present, it is a firm requirement; if no
  schedule fact is present, the user has no schedule constraint. If a
  documents_available fact is present, it is a firm limitation: listed documents
  are the only documents the user can provide, and `none` means the user cannot
  currently provide documents. If no documents_available fact is present, do not
  add a document limitation to the profile.
- Do not soften firm facts with words like "prefers", "would like", or
  "ideally".
- Do not mention resource IDs, agency names, site names, or specific resource names.

Return exactly this JSON shape:
{
  "need_summary": "one sentence plain-language service need, without benchmark category labels",
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
    for difficulty in sorted({spec.get("difficulty", "") for spec in specs}):
        group = [spec for spec in specs if spec.get("difficulty", "") == difficulty]
        assignments = [
            TRAIT_GROUPS[index % len(TRAIT_GROUPS)]
            for index in range(len(group))
        ]
        rng.shuffle(assignments)
        for spec, trait in zip(group, assignments, strict=True):
            traits_by_case[spec["case_id"]] = [trait]
    return traits_by_case


def build_user_card(client, model: str, spec: dict, traits: list[str]) -> dict:
    generated = generate_profile(client, model, spec)
    card = {
        "case_id": spec["case_id"],
        "user_id": spec["case_id"],
        "difficulty": spec["difficulty"],
        "target_service_categories": spec["target_service_categories"],
        "ground_truth_resource_ids": spec["ground_truth_resource_ids"],
        "traits": traits,
        "need_summary": generated["need_summary"],
        "profile": generated["profile"],
        "known_facts": constraint_facts_from_spec(spec),
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
    last_error: Exception | None = None
    for attempt in range(1, MAX_PROFILE_GENERATION_ATTEMPTS + 1):
        response = client.responses.create(
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
        try:
            return validated_generated_profile(getattr(response, "output_text", "") or "")
        except RuntimeError as exc:
            last_error = exc
            print(
                f"{spec['case_id']} profile generation attempt "
                f"{attempt}/{MAX_PROFILE_GENERATION_ATTEMPTS} failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
    raise RuntimeError(f"{spec['case_id']} profile generation failed after retries") from last_error


def llm_visible_spec(spec: dict) -> dict:
    visible = {
        "case_id": spec["case_id"],
        "difficulty": spec["difficulty"],
        "benchmark_service_categories_do_not_copy": spec["target_service_categories"],
        "primary_resource_service_name": spec["primary_resource"]["service_name"],
        "location": spec["location"],
        "constraints": constraint_facts_from_spec(spec),
    }
    if spec["difficulty"] == "hard":
        visible["user_qualification"] = spec["user_qualification"]
    return visible


def constraint_facts_from_spec(spec: dict) -> list[str]:
    requirements = spec["user_requirements"]
    facts = [
        location_fact(requirements),
        f"intake_requirement: {', '.join(requirements.get('intake_methods', []))}",
    ]
    schedule = schedule_fact(requirements)
    if schedule:
        facts.append(schedule)
    if spec["difficulty"] == "hard":
        documents = (spec.get("user_qualification") or {}).get("documents_available") or []
        if documents:
            facts.append(f"documents_available: {', '.join(documents)}")
        else:
            facts.append("documents_available: none")
    return facts


def location_fact(requirements: dict) -> str:
    if requirements.get("zipcodes"):
        return f"location_requirement: zipcode={', '.join(requirements['zipcodes'])}"
    if requirements.get("cities"):
        return f"location_requirement: city={', '.join(requirements['cities'])}"
    if requirements.get("counties"):
        return f"location_requirement: county={', '.join(requirements['counties'])}"
    raise RuntimeError("missing location requirement")


def schedule_fact(requirements: dict) -> str | None:
    if requirements.get("available_days"):
        return f"schedule_requirement: available_days={', '.join(requirements['available_days'])}"
    if requirements.get("available_time_windows"):
        windows = json.dumps(requirements["available_time_windows"], ensure_ascii=False)
        return f"schedule_requirement: available_time_windows={windows}"
    if requirements.get("requires_24_hours"):
        return "schedule_requirement: requires_24_hours=true"
    return None


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
    for key in ("need_summary", "profile"):
        if not isinstance(parsed.get(key), str) or not parsed[key].strip():
            raise RuntimeError(f"LLM JSON output missing non-empty `{key}`.")
        parsed[key] = parsed[key].strip()
    return {
        "need_summary": parsed["need_summary"],
        "profile": parsed["profile"],
    }


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
