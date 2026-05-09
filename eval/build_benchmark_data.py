from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.llm import load_dotenv, make_openai_client
from tools.curated_categories import SERVICE_CATEGORY_DESCRIPTIONS
from tools.indiana211 import Resource, load_indiana_csv


OUT_DIR = Path("data/benchmark")
USERS_PATH = OUT_DIR / "user_cards.json"
GROUND_TRUTH_PATH = OUT_DIR / "ground_truth.json"
REPORT_PATH = OUT_DIR / "dataset_report.md"

INDIANA_LOCATIONS = [
    ("MARION", "Indianapolis", "46204"),
    ("ALLEN", "Fort Wayne", "46802"),
    ("LAKE", "Gary", "46402"),
    ("ST. JOSEPH", "South Bend", "46601"),
    ("VANDERBURGH", "Evansville", "47708"),
    ("ELKHART", "Elkhart", "46516"),
    ("MONROE", "Bloomington", "47404"),
    ("DELAWARE", "Muncie", "47305"),
    ("TIPPECANOE", "Lafayette", "47901"),
    ("VIGO", "Terre Haute", "47807"),
]

GENERATOR_INSTRUCTIONS = """
You create hidden user profiles for a benchmark of Indiana 211 resource-search
agents. Follow a tau-bench-like setup: the profile is hidden from the agent and
used by an LLM simulated user during a multi-turn interaction.

Create realistic users, not cooperative test fixtures. The user may be vague,
emotional, tangential, uncertain, or ask for something unrealistic. The user
must not know resource IDs or ground-truth resources. The profile should make
the provided ground-truth resources reasonable answers, but should not mention
their names.

Return strict JSON only. No markdown.
""".strip()


def main() -> None:
    args = parse_args()
    load_dotenv()
    index = load_indiana_csv(args.index_path)
    resource_sets = select_resource_sets(index.resources, args.cases, args.seed)
    client = make_openai_client(args.provider)
    cards = []
    ground_truth = []
    for idx, (category, resources) in enumerate(resource_sets, start=1):
        print(f"[{idx}/{len(resource_sets)}] {resources[0].resource_id}")
        location = user_location_for_resource(resources[0], idx)
        card = generate_card(client, args.model, idx, category, resources, location)
        candidates = review_candidates(index.resources, category, resources[0])
        acceptable = review_acceptable_resources(client, args.model, card, resources[0], candidates)
        cards.append(card)
        ground_truth.append(ground_truth_for_card(card, resources[0], acceptable))
    validate_dataset(cards, ground_truth)
    write_json(args.users_out, cards)
    write_json(args.ground_truth_out, ground_truth)
    args.report_out.write_text(render_report(cards, ground_truth), encoding="utf-8")
    print(f"Wrote {args.users_out}")
    print(f"Wrote {args.ground_truth_out}")
    print(f"Wrote {args.report_out}")


def select_resource_sets(resources: list[Resource], cases: int, seed: int) -> list[tuple[str, list[Resource]]]:
    rng = random.Random(seed)
    eligible = [
        resource
        for resource in resources
        if resource.service_categories
        and resource.service_area
        and resource.zipcode
        and (resource.phone or resource.website)
        and len(resource.service_categories) <= 3
        and not _is_low_value_resource(resource)
    ]
    by_category: dict[str, list[Resource]] = {}
    for resource in eligible:
        for category in resource.service_categories:
            if not resource_category_is_semantic_match(resource, category):
                continue
            by_category.setdefault(category, []).append(resource)
    categories = sorted(by_category)
    selected = []
    used = set()
    cursor = 0
    while len(selected) < cases:
        category = categories[cursor % len(categories)]
        cursor += 1
        pool = [r for r in by_category[category] if r.resource_id not in used]
        if not pool:
            continue
        primary = rng.choice(pool)
        acceptable = [
            r
            for r in by_category[category]
            if r.resource_id != primary.resource_id
            and _service_area_overlaps(primary, r)
            and not _is_low_value_resource(r)
            and _resources_are_close_alternatives(primary, r)
        ]
        acceptable = sorted(
            acceptable,
            key=lambda r: (
                r.city != primary.city,
                r.zipcode != primary.zipcode,
                r.service_name.lower(),
                r.resource_id,
            ),
        )[:4]
        selected.append((category, [primary, *acceptable]))
        used.add(primary.resource_id)
    return selected


def generate_card(
    client,
    model: str,
    idx: int,
    category: str,
    resources: list[Resource],
    location: tuple[str, str, str],
) -> dict:
    primary = resources[0]
    county, city, zipcode = location
    payload = {
        "user_id": f"llu-{idx:03d}",
        "target_service_category": category,
        "primary_resource": resource_brief(primary),
        "acceptable_resources": [resource_brief(resource) for resource in resources[1:]],
        "required_schema": {
            "user_id": "string",
            "opening": "one short first user message",
            "background": "brief hidden background",
            "need": "real underlying need",
            "location": {"city": city, "zipcode": zipcode, "county": county},
            "household": "realistic household or personal situation",
            "urgency": "same day | 48 hours | within one week | moderate | low",
            "hard_constraints": ["requirements that are truly required"],
            "soft_preferences": ["preferences that are not hard requirements"],
            "known_facts": ["facts the user can reveal if asked"],
            "unknowns": ["details the user genuinely does not know"],
            "non_collab": {
                "tangential": "optional irrelevant detail",
                "unreasonable_demand": "optional unrealistic ask",
                "emotional": "emotional stance",
                "implicit_vague": "how the user may be vague",
                "contradictory": "optional mild contradiction that can be clarified",
            },
            "disclosure_policy": {
                "reveal_location": "when asked or after first follow-up",
                "do_not_reveal": ["resource IDs", "ground truth", "tool fields"],
            },
            "target_service_categories": [category],
            "difficulty": "easy | medium | hard",
        },
        "quality_rules": [
            "Do not copy agency eligibility text into the user's hard constraints.",
            "The user should describe their real-world problem, not the resource's internal rules.",
            "Use hard_constraints only for constraints the user would naturally know or insist on.",
            "Use unknowns for program rules, documents, fees, or eligibility details the user would not know.",
            "Keep the user need focused on target_service_category.",
            "Use the provided location as the user's location even if the resource office is elsewhere.",
        ],
    }
    response = client.responses.create(
        model=model,
        instructions=GENERATOR_INSTRUCTIONS,
        input=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
    )
    card = parse_json_object(getattr(response, "output_text", "") or "")
    card["user_id"] = f"llu-{idx:03d}"
    card["target_service_categories"] = [category]
    card["location"] = {"city": city, "zipcode": zipcode, "county": county}
    card.setdefault("opening", "I need help finding a local resource.")
    card.setdefault("difficulty", "medium")
    return card


def review_candidates(resources: list[Resource], category: str, primary: Resource) -> list[Resource]:
    candidates = [
        resource
        for resource in resources
        if resource.resource_id != primary.resource_id
        and category in resource.service_categories
        and resource.service_area
        and resource.zipcode
        and (resource.phone or resource.website)
        and _service_area_overlaps(primary, resource)
        and not _is_low_value_resource(resource)
        and resource_category_is_semantic_match(resource, category)
    ]
    candidates = sorted(
        candidates,
        key=lambda resource: (
            not _resources_are_close_alternatives(primary, resource),
            resource.city != primary.city,
            resource.zipcode != primary.zipcode,
            resource.service_name.lower(),
            resource.resource_id,
        ),
    )
    return [primary, *candidates[:15]]


GT_REVIEWER_INSTRUCTIONS = """
You review benchmark ground truth for an Indiana 211 resource-search task.
Given a hidden user profile, one primary resource, and candidate resources,
select which candidates are acceptable answers.

Rules:
- Always include the primary resource unless it clearly contradicts the profile.
- A resource is acceptable only if it solves the user's main need and hard constraints.
- Do not include a resource merely because it shares a broad category, county, or one word in the name.
- Include close substitutes even when the service name differs, if a real user would reasonably be helped.
- Exclude adjacent services, directories, government offices, or informational pages that do not solve the stated need.

Return strict JSON only: {"acceptable_resource_ids": ["..."]}.
""".strip()


def review_acceptable_resources(
    client,
    model: str,
    card: dict,
    primary: Resource,
    candidates: list[Resource],
) -> list[Resource]:
    payload = {
        "hidden_user_profile": card,
        "primary_resource_id": primary.resource_id,
        "candidate_resources": [
            {"resource_id": resource.resource_id, **resource_brief(resource)}
            for resource in candidates
        ],
    }
    response = client.responses.create(
        model=model,
        instructions=GT_REVIEWER_INSTRUCTIONS,
        input=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
    )
    data = parse_json_object(getattr(response, "output_text", "") or "")
    allowed = {resource.resource_id: resource for resource in candidates}
    selected_ids = [
        resource_id
        for resource_id in data.get("acceptable_resource_ids", [])
        if resource_id in allowed
    ]
    if primary.resource_id not in selected_ids:
        selected_ids.insert(0, primary.resource_id)
    selected_ids = selected_ids[:5]
    return [allowed[resource_id] for resource_id in selected_ids]


def ground_truth_for_card(card: dict, primary: Resource, acceptable_resources: list[Resource]) -> dict:
    resources = [primary, *[resource for resource in acceptable_resources if resource.resource_id != primary.resource_id]]
    return {
        "user_id": card["user_id"],
        "primary_gt_resource_ids": [primary.resource_id],
        "acceptable_gt_resource_ids": [resource.resource_id for resource in resources],
        "target_service_categories": card["target_service_categories"],
        "matching_notes": [
            {
                "resource_id": resource.resource_id,
                "service_name": resource.service_name,
                "agency_name": resource.agency_name,
                "city": resource.city,
                "zipcode": resource.zipcode,
                "service_area": list(resource.service_area),
                "service_categories": list(resource.service_categories),
            }
            for resource in resources
        ],
    }


def validate_dataset(cards: list[dict], ground_truth: list[dict]) -> None:
    if len(cards) != len(ground_truth):
        raise RuntimeError("cards and ground truth length mismatch")
    for card, gt in zip(cards, ground_truth):
        if card["user_id"] != gt["user_id"]:
            raise RuntimeError(f"user_id mismatch: {card['user_id']} vs {gt['user_id']}")
        if not card.get("opening"):
            raise RuntimeError(f"missing opening: {card['user_id']}")
        if not gt.get("primary_gt_resource_ids"):
            raise RuntimeError(f"missing primary gt: {card['user_id']}")
        serialized = json.dumps(card, ensure_ascii=False).lower()
        for resource_id in gt["acceptable_gt_resource_ids"]:
            if resource_id.lower() in serialized:
                raise RuntimeError(f"card leaks resource id {resource_id}: {card['user_id']}")


def resource_brief(resource: Resource) -> dict:
    return {
        "service_name": resource.service_name,
        "agency_name": resource.agency_name,
        "service_categories": list(resource.service_categories),
        "service_area": list(resource.service_area),
        "city": resource.city,
        "zipcode": resource.zipcode,
        "eligibility": resource.eligibility,
        "schedule": resource.site_schedule,
        "intake": resource.site_details,
        "fees": resource.fee_structure,
        "documents": resource.documents_required,
        "eligibility_tags": list(resource.eligibility_tags),
        "schedule_tags": list(resource.schedule_tags),
        "intake_methods": list(resource.intake_methods),
        "document_requirements": list(resource.document_requirements),
        "fee_options": list(resource.fee_options),
    }


def user_location_for_resource(resource: Resource, idx: int) -> tuple[str, str, str]:
    if "STATEWIDE" in resource.service_area or "ALL" in resource.service_area:
        return INDIANA_LOCATIONS[idx % len(INDIANA_LOCATIONS)]
    county = next((item for item in resource.service_area if item not in {"STATEWIDE", "ALL"}), "")
    return county, resource.city, resource.zipcode


def parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise RuntimeError("generator did not return an object")
    return data


def _service_area_overlaps(a: Resource, b: Resource) -> bool:
    if set(a.service_area) & set(b.service_area):
        return True
    return "STATEWIDE" in a.service_area or "STATEWIDE" in b.service_area


def _is_low_value_resource(resource: Resource) -> bool:
    text = " ".join([resource.service_name, resource.agency_name]).lower()
    low_value_terms = {
        "locator",
        "directory",
        "post office",
        "representative",
        "senator",
        "political party",
        "township assistance",
        "county commissioner",
        "commissioners",
        "assessor",
        "recorder",
        "treasurer",
        "auditor",
        "municipal government",
        "city and town offices",
    }
    return any(term in text for term in low_value_terms)


def _resources_are_close_alternatives(primary: Resource, candidate: Resource) -> bool:
    primary_name = primary.service_name.lower().strip()
    candidate_name = candidate.service_name.lower().strip()
    if primary_name == candidate_name:
        return True
    if primary_name in {"information and referral"}:
        return False

    primary_terms = set(tokenize(primary.service_name))
    candidate_terms = set(tokenize(candidate.service_name))
    if not primary_terms or not candidate_terms:
        return False

    overlap = primary_terms & candidate_terms
    required_overlap = min(2, len(primary_terms), len(candidate_terms))
    return len(overlap) >= required_overlap


def resource_category_is_semantic_match(resource: Resource, category: str) -> bool:
    category_terms = set(tokenize(category + " " + SERVICE_CATEGORY_DESCRIPTIONS.get(category, "")))
    resource_terms = set(
        tokenize(
            " ".join(
                [
                    resource.service_name,
                    resource.agency_name,
                    resource.site_details,
                    resource.eligibility,
                ]
            )
        )
    )
    if category in {"Government Offices and Public Services", "Information and Referral"}:
        return bool(category_terms & resource_terms)
    return len(category_terms & resource_terms) >= 2


def tokenize(text: str) -> list[str]:
    stopwords = {
        "and",
        "or",
        "the",
        "for",
        "with",
        "help",
        "services",
        "service",
        "program",
        "programs",
        "support",
        "assistance",
        "resources",
        "resource",
    }
    return [
        token
        for token in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()
        if len(token) > 2 and token not in stopwords
    ]


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render_report(cards: list[dict], ground_truth: list[dict]) -> str:
    difficulties = {level: sum(card.get("difficulty") == level for card in cards) for level in ["easy", "medium", "hard"]}
    categories = sorted({category for card in cards for category in card.get("target_service_categories", [])})
    return "\n".join(
        [
            "# LLM Simulated User Benchmark Data",
            "",
            f"- User cards: {len(cards)}",
            f"- Ground truth rows: {len(ground_truth)}",
            f"- Difficulty distribution: {difficulties}",
            f"- Covered service categories: {len(categories)}",
            "",
            "## Categories",
            "",
            *[f"- {category}" for category in categories],
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LLM-simulated-user benchmark data.")
    parser.add_argument("--provider", default="openai", choices=["openai", "openrouter"])
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--index-path", default="data/indiana211/indiana211_resources_deduped.csv")
    parser.add_argument("--cases", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--users-out", type=Path, default=USERS_PATH)
    parser.add_argument("--ground-truth-out", type=Path, default=GROUND_TRUTH_PATH)
    parser.add_argument("--report-out", type=Path, default=REPORT_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    main()
