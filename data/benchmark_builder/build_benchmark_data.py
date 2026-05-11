from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.indiana211 import Resource, load_indiana_csv


OUT_DIR = Path("data/benchmark")
CASE_SPECS_PATH = OUT_DIR / "case_specs.json"
REPORT_PATH = OUT_DIR / "dataset_report.md"

BASE_REQUIREMENT_FIELDS = {"counties", "service_categories"}
MAX_EXTRA_REQUIREMENTS = 3
MAX_ATTEMPT_MULTIPLIER = 120
BASE_ELIGIBILITY_TAGS = {"resident", "income"}
PERSONA_ELIGIBILITY_TAGS = {"children", "pregnant", "senior", "veteran"}
CONTEXT_ELIGIBILITY_TAGS = {"disability", "homeless"}
SCHEDULE_RELEVANT_INTAKE = {"call", "walk_in", "appointment", "text"}
SCHEDULE_REQUIREMENT_FIELDS = {
    "available_days",
    "available_at_or_after",
    "requires_weekend",
    "requires_24_hours",
}
DIFFICULTY_EXTRA_COUNTS = {
    "easy": 1,
    "medium": 2,
    "hard": 3,
}


@dataclass(frozen=True)
class Candidate:
    resource: Resource
    category: str
    county: str


def main() -> None:
    args = parse_args()
    resources = load_indiana_csv(args.index_path).resources
    difficulty_targets = parse_difficulty_targets(args)
    specs, stats = build_case_specs(
        resources,
        cases=sum(difficulty_targets.values()),
        seed=args.seed,
        difficulty_targets=difficulty_targets,
        progress_every=args.progress_every,
    )
    write_json(args.case_specs_out, specs)
    args.report_out.write_text(render_report(specs, stats), encoding="utf-8")
    print(f"Wrote {args.case_specs_out}")
    print(f"Wrote {args.report_out}")


def build_case_specs(
    resources: list[Resource],
    cases: int,
    seed: int,
    difficulty_targets: dict[str, int],
    progress_every: int = 100,
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    eligible = [resource for resource in resources if is_benchmark_resource(resource)]
    candidates = build_candidates(eligible)
    rng.shuffle(candidates)

    selected: list[dict] = []
    used_primary_ids: set[str] = set()
    category_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    max_per_category = max(1, (cases + len(all_categories(eligible)) - 1) // max(len(all_categories(eligible)), 1))
    max_attempts = max(len(candidates), cases * MAX_ATTEMPT_MULTIPLIER)
    attempted = 0
    skipped_category_cap = 0
    start_time = time.monotonic()

    while len(selected) < cases and attempted < max_attempts and candidates:
        candidate = candidates[attempted % len(candidates)]
        attempted += 1
        resource = candidate.resource
        if resource.resource_id in used_primary_ids:
            continue
        if category_counts[candidate.category] >= max_per_category and len(selected) < cases // 2:
            skipped_category_cap += 1
            continue
        difficulty = choose_needed_difficulty(difficulty_targets, difficulty_counts, rng)
        if not difficulty:
            break
        spec = sample_case_spec(eligible, candidate, rng, difficulty)
        if not spec:
            continue
        selected.append(spec)
        used_primary_ids.add(resource.resource_id)
        category_counts[candidate.category] += 1
        difficulty_counts[spec["difficulty"]] += 1
        if progress_every and (attempted == 1 or attempted % progress_every == 0 or len(selected) == cases):
            print_progress(attempted, max_attempts, len(selected), cases, start_time)

    if progress_every:
        sys.stderr.write("\n")
    for idx, spec in enumerate(selected, start=1):
        spec["case_id"] = f"spec-{idx:03d}"
    validate_case_specs(selected, eligible)
    stats = {
        "resource_rows": len(resources),
        "eligible_resources": len(eligible),
        "candidate_probes": len(candidates),
        "attempted_probes": attempted,
        "selected_specs": len(selected),
        "max_attempts": max_attempts,
        "skipped_category_cap": skipped_category_cap,
        "difficulty_targets": difficulty_targets,
    }
    return selected, stats


def build_candidates(resources: list[Resource]) -> list[Candidate]:
    candidates = []
    for resource in resources:
        for category in resource.service_categories:
            for county in sampled_counties(resource):
                candidates.append(Candidate(resource=resource, category=category, county=county))
    return candidates


def sample_case_spec(
    resources: list[Resource],
    candidate: Candidate,
    rng: random.Random,
    difficulty: str,
) -> dict | None:
    resource = candidate.resource
    base_requirements = {
        "counties": [candidate.county],
        "service_categories": [candidate.category],
    }
    qualification_options = user_qualification_options(resource, candidate.category, rng)
    requirement_options = user_requirement_options(resource, base_requirements, rng, difficulty)
    for qualification in qualification_options:
        for requirements, extra_count in requirement_options:
            matches = [
                item
                for item in resources
                if matches_user_requirements(item, requirements)
                and user_qualifies_for_resource(qualification, item)
            ]
            if len(matches) == 1 and matches[0].resource_id == resource.resource_id:
                return make_case_spec(resource, candidate, requirements, qualification, extra_count)
    return None


def user_requirement_options(
    resource: Resource,
    base: dict,
    rng: random.Random,
    difficulty: str,
) -> list[tuple[dict, int]]:
    intake_extras = intake_requirement_extras(resource)
    other_extras = non_intake_requirement_extras(resource)
    rng.shuffle(intake_extras)
    rng.shuffle(other_extras)
    options: list[tuple[dict, int]] = []
    extra_count = DIFFICULTY_EXTRA_COUNTS[difficulty]
    other_count = extra_count - 1
    if not intake_extras or other_count < 0:
        return []
    if other_count > len(other_extras):
        return []
    for intake in intake_extras:
        combos = [(intake, *combo) for combo in combinations(other_extras, other_count)]
        rng.shuffle(combos)
        for combo in combos:
            if repeated_fields(combo):
                continue
            if not requirement_combo_is_valid(combo):
                continue
            requirements = copy_requirements(base)
            for item in combo:
                requirements = merge_requirements(requirements, item)
            options.append((requirements, extra_count))
    rng.shuffle(options)
    return options


def intake_requirement_extras(resource: Resource) -> list[dict]:
    extras = []
    for value in user_facing_intake_methods(resource.intake_methods):
        extras.append({"intake_methods": [value]})
    return dedupe_requirements(extras)


def non_intake_requirement_extras(resource: Resource) -> list[dict]:
    extras = []
    if resource.city and in_person_intake(resource):
        extras.append({"cities": [resource.city]})
    if resource.zipcode and in_person_intake(resource):
        extras.append({"zipcodes": [resource.zipcode]})
    extras.extend(schedule_requirement_extras(resource))
    return dedupe_requirements(extras)


def user_facing_intake_methods(values: tuple[str, ...]) -> list[str]:
    return [value for value in values if value != "empty"]


def in_person_intake(resource: Resource) -> bool:
    return "walk_in" in resource.intake_methods or "appointment" in resource.intake_methods


def schedule_requirement_extras(resource: Resource) -> list[dict]:
    if resource.schedule_status != "structured":
        return []
    windows = resource.schedule_windows
    extras = []
    weekend_days = sorted({window.day for window in windows if window.day in {"sat", "sun"}})
    if weekend_days:
        extras.append({"requires_weekend": True})
        extras.append({"available_days": [weekend_days[0]]})
    if any(window.end_minute > 18 * 60 or is_24_hour_window(window) for window in windows):
        extras.append({"available_at_or_after": "18:00"})
    if any(is_24_hour_window(window) for window in windows):
        extras.append({"requires_24_hours": True})
    return extras


def user_qualification_options(resource: Resource, category: str, rng: random.Random) -> list[dict]:
    base = {
        "eligibility": list(user_eligibility_for_resource(resource, category, rng)),
        "fee_capacity": fee_capacity_for_resource(resource),
        "documents_available": list(required_documents(resource)),
    }
    generous = {
        "eligibility": sorted(set(base["eligibility"]) | {"resident"}),
        "fee_capacity": "can_pay",
        "documents_available": sorted(set(base["documents_available"]) | common_documents()),
    }
    minimal = {
        "eligibility": list(base["eligibility"]),
        "fee_capacity": base["fee_capacity"],
        "documents_available": list(base["documents_available"]),
    }
    options = dedupe_json_objects([base, minimal, generous])
    rng.shuffle(options)
    return options


def user_eligibility_for_resource(resource: Resource, category: str, rng: random.Random) -> tuple[str, ...]:
    tags = set(resource_eligibility_tags(resource))
    eligibility = set(tags & BASE_ELIGIBILITY_TAGS)
    eligibility |= tags & CONTEXT_ELIGIBILITY_TAGS
    persona_tags = sorted(tags & PERSONA_ELIGIBILITY_TAGS)
    if persona_tags:
        eligibility.add(choose_persona_tag(persona_tags, category, rng))
    return tuple(sorted(eligibility))


def choose_persona_tag(persona_tags: list[str], category: str, rng: random.Random) -> str:
    category_text = category.lower()
    category_preferences = [
        ("pregnancy", "pregnant"),
        ("reproductive", "pregnant"),
        ("veteran", "veteran"),
        ("military", "veteran"),
        ("youth", "children"),
        ("school", "children"),
        ("family", "children"),
        ("senior", "senior"),
    ]
    for keyword, tag in category_preferences:
        if keyword in category_text and tag in persona_tags:
            return tag
    return rng.choice(persona_tags)


def resource_eligibility_tags(resource: Resource) -> tuple[str, ...]:
    ignored = {"empty", "open"}
    return tuple(value for value in resource.eligibility_tags if value not in ignored)


def required_documents(resource: Resource) -> tuple[str, ...]:
    ignored = {"empty", "none", "varies"}
    return tuple(value for value in resource.document_requirements if value not in ignored)


def common_documents() -> set[str]:
    return {"photo_id", "proof_of_address", "proof_of_income", "insurance_card"}


def fee_capacity_for_resource(resource: Resource) -> str:
    fees = set(resource.fee_options)
    if "free" in fees:
        return "must_be_free"
    if "insurance" in fees:
        return "has_insurance"
    return "can_pay"


def matches_user_requirements(resource: Resource, requirements: dict) -> bool:
    if not any_exact(requirements.get("service_categories"), resource.service_categories):
        return False
    if not matches_county(requirements.get("counties"), resource):
        return False
    if requirements.get("cities") and not any_exact(requirements["cities"], (resource.city,)):
        return False
    if requirements.get("zipcodes") and not any_exact(requirements["zipcodes"], (resource.zipcode,)):
        return False
    if requirements.get("intake_methods") and not any_exact(requirements["intake_methods"], resource.intake_methods):
        return False
    return matches_schedule_requirements(resource, requirements)


def matches_county(counties: list[str] | None, resource: Resource) -> bool:
    if not counties:
        return True
    return any_exact(counties, resource.service_area) or any_exact(("STATEWIDE", "ALL"), resource.service_area)


def matches_schedule_requirements(resource: Resource, requirements: dict) -> bool:
    has_schedule = bool(
        requirements.get("available_days")
        or requirements.get("available_at_or_after")
        or requirements.get("requires_weekend")
        or requirements.get("requires_24_hours")
    )
    if not has_schedule:
        return True
    if resource.schedule_status != "structured":
        return False
    windows = resource.schedule_windows
    if requirements.get("available_days"):
        days = set(requirements["available_days"])
        if not any(window.day in days for window in windows):
            return False
    if requirements.get("available_at_or_after"):
        minute = parse_hhmm(requirements["available_at_or_after"])
        if not any(is_24_hour_window(window) or window.end_minute > minute for window in windows):
            return False
    if requirements.get("requires_weekend"):
        if not any(window.day in {"sat", "sun"} for window in windows):
            return False
    if requirements.get("requires_24_hours"):
        if not any(is_24_hour_window(window) for window in windows):
            return False
    return True


def user_qualifies_for_resource(qualification: dict, resource: Resource) -> bool:
    user_eligibility = set(qualification.get("eligibility") or ())
    if not eligibility_is_compatible(user_eligibility, resource):
        return False
    user_documents = set(qualification.get("documents_available") or ())
    if not set(required_documents(resource)).issubset(user_documents):
        return False
    return fee_is_compatible(str(qualification.get("fee_capacity") or "can_pay"), resource.fee_options)


def eligibility_is_compatible(user_eligibility: set[str], resource: Resource) -> bool:
    tags = set(resource_eligibility_tags(resource))
    required = tags & (BASE_ELIGIBILITY_TAGS | CONTEXT_ELIGIBILITY_TAGS)
    if not required.issubset(user_eligibility):
        return False
    persona_tags = tags & PERSONA_ELIGIBILITY_TAGS
    if persona_tags and not (persona_tags & user_eligibility):
        return False
    return True


def fee_is_compatible(fee_capacity: str, fee_options: tuple[str, ...]) -> bool:
    fees = set(fee_options or ())
    if not fees or "unknown" in fees or "varies" in fees:
        return fee_capacity != "must_be_free"
    if "free" in fees:
        return True
    if fee_capacity == "must_be_free":
        return False
    if fee_capacity == "has_insurance":
        return bool(fees & {"insurance", "sliding_scale"})
    if fee_capacity == "can_pay":
        return bool(fees & {"payment_required", "sliding_scale", "insurance"})
    return False


def make_case_spec(
    resource: Resource,
    candidate: Candidate,
    requirements: dict,
    qualification: dict,
    extra_requirement_count: int,
) -> dict:
    return {
        "case_id": "",
        "difficulty": difficulty_for_extra_count(extra_requirement_count),
        "need_category": candidate.category,
        "location": {
            "county": candidate.county,
            "city": resource.city,
            "state": "IN",
            "zipcode": resource.zipcode,
        },
        "user_requirements": requirements,
        "user_qualification": qualification,
        "ground_truth_resource_ids": [resource.resource_id],
        "target_service_categories": [candidate.category],
        "primary_resource": resource_summary(resource),
    }


def validate_case_specs(specs: list[dict], resources: list[Resource]) -> None:
    seen = set()
    by_id = {resource.resource_id: resource for resource in resources}
    for spec in specs:
        ground_truth_ids = spec.get("ground_truth_resource_ids") or []
        if len(ground_truth_ids) != 1:
            raise RuntimeError(f"case must have exactly one ground truth resource: {spec.get('case_id')}")
        primary_id = ground_truth_ids[0]
        if primary_id in seen:
            raise RuntimeError(f"duplicate primary resource: {primary_id}")
        seen.add(primary_id)
        matches = [
            resource.resource_id
            for resource in resources
            if matches_user_requirements(resource, spec["user_requirements"])
            and user_qualifies_for_resource(spec["user_qualification"], resource)
        ]
        if matches != [primary_id]:
            raise RuntimeError(f"primary is not unique for case: {primary_id}; matches={matches[:5]}")
        if spec["location"]["county"] in {"STATEWIDE", "ALL"}:
            raise RuntimeError(f"invalid location county: {primary_id}")
        if primary_id not in by_id:
            raise RuntimeError(f"primary resource missing from index: {primary_id}")
        if not spec["user_requirements"].get("intake_methods"):
            raise RuntimeError(f"missing intake requirement: {primary_id}")


def merge_requirements(left: dict, right: dict) -> dict:
    merged = copy_requirements(left)
    for key, value in right.items():
        if isinstance(value, list):
            merged[key] = sorted(set(merged.get(key, [])) | set(value))
        elif isinstance(value, bool):
            merged[key] = bool(value or merged.get(key, False))
        elif value:
            merged[key] = value
    return merged


def copy_requirements(requirements: dict) -> dict:
    return {
        key: list(value) if isinstance(value, list) else value
        for key, value in requirements.items()
    }


def dedupe_requirements(items: list[dict]) -> list[dict]:
    deduped = []
    seen = set()
    for item in items:
        key = json.dumps(item, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def dedupe_json_objects(items: list[dict]) -> list[dict]:
    deduped = []
    seen = set()
    for item in items:
        key = json.dumps(item, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def repeated_fields(items: tuple[dict, ...]) -> bool:
    fields = [field for item in items for field in item]
    return len(fields) != len(set(fields))


def requirement_combo_is_valid(items: tuple[dict, ...]) -> bool:
    fields = {field for item in items for field in item}
    if not fields & SCHEDULE_REQUIREMENT_FIELDS:
        return True
    for item in items:
        intake_methods = set(item.get("intake_methods") or ())
        if intake_methods & SCHEDULE_RELEVANT_INTAKE:
            return True
    return False


def sampled_counties(resource: Resource) -> list[str]:
    counties = [county for county in resource.service_area if county not in {"STATEWIDE", "ALL"}]
    return counties[:3]


def is_benchmark_resource(resource: Resource) -> bool:
    return (
        (resource.state or "").upper() == "IN"
        and bool(resource.service_categories)
        and bool(sampled_counties(resource))
        and bool(resource.zipcode)
        and bool(resource.phone or resource.website)
        and bool(user_facing_intake_methods(resource.intake_methods))
        and len(resource.service_categories) <= 3
        and "empty" not in resource.intake_methods
        and resource.schedule_status in {"structured", "appointment_only"}
    )


def any_exact(needles, haystack) -> bool:
    return bool(set(needles or ()) & set(haystack or ()))


def is_24_hour_window(window) -> bool:
    return window.start_minute == 0 and window.end_minute >= 24 * 60


def parse_hhmm(value: str) -> int:
    hour, minute = value.split(":", 1)
    return int(hour) * 60 + int(minute)


def difficulty_for_extra_count(extra_count: int) -> str:
    if extra_count <= 1:
        return "easy"
    if extra_count == 2:
        return "medium"
    return "hard"


def choose_needed_difficulty(
    targets: dict[str, int],
    counts: Counter[str],
    rng: random.Random,
) -> str | None:
    remaining = [
        difficulty
        for difficulty, target in targets.items()
        if counts[difficulty] < target
    ]
    if not remaining:
        return None
    weights = [targets[difficulty] - counts[difficulty] for difficulty in remaining]
    return rng.choices(remaining, weights=weights, k=1)[0]


def all_categories(resources: list[Resource]) -> set[str]:
    return {category for resource in resources for category in resource.service_categories}


def resource_summary(resource: Resource) -> dict:
    return {
        "resource_id": resource.resource_id,
        "service_name": resource.service_name,
        "agency_name": resource.agency_name,
        "service_area": list(resource.service_area),
        "city": resource.city,
        "zipcode": resource.zipcode,
        "schedule": resource.site_schedule,
        "schedule_status": resource.schedule_status,
        "intake_methods": list(resource.intake_methods),
        "document_requirements": list(resource.document_requirements),
        "fee_options": list(resource.fee_options),
        "eligibility_tags": list(resource.eligibility_tags),
    }


def render_report(specs: list[dict], stats: dict) -> str:
    category_counts = Counter(spec["need_category"] for spec in specs)
    difficulty_counts = Counter(spec["difficulty"] for spec in specs)
    county_counts = Counter(spec["location"]["county"] for spec in specs)
    city_counts = Counter(spec["location"]["city"] for spec in specs if spec["location"].get("city"))
    requirement_fields = Counter(
        field
        for spec in specs
        for field in spec["user_requirements"]
        if field not in BASE_REQUIREMENT_FIELDS
    )
    requirement_values = requirement_value_counts(specs)
    fee_counts = Counter(spec["user_qualification"]["fee_capacity"] for spec in specs)
    eligibility_counts = Counter(
        tag
        for spec in specs
        for tag in spec["user_qualification"].get("eligibility", [])
    )
    document_counts = Counter(
        doc
        for spec in specs
        for doc in spec["user_qualification"].get("documents_available", [])
    )
    resource_schedule_counts = Counter(spec["primary_resource"].get("schedule_status", "") for spec in specs)
    resource_intake_counts = Counter(
        method
        for spec in specs
        for method in spec["primary_resource"].get("intake_methods", [])
    )
    resource_fee_counts = Counter(
        fee
        for spec in specs
        for fee in spec["primary_resource"].get("fee_options", [])
    )
    resource_document_counts = Counter(
        doc
        for spec in specs
        for doc in spec["primary_resource"].get("document_requirements", [])
    )
    resource_eligibility_counts = Counter(
        tag
        for spec in specs
        for tag in spec["primary_resource"].get("eligibility_tags", [])
    )
    schedule_without_intake = count_schedule_requirements_without_relevant_intake(specs)
    lines = [
        "# Deterministic Case Spec Benchmark Data",
        "",
        f"- Resource rows: {stats['resource_rows']}",
        f"- Benchmark-eligible resources: {stats['eligible_resources']}",
        f"- Candidate probes: {stats['candidate_probes']}",
        f"- Attempted probes: {stats['attempted_probes']}",
        f"- Valid case specs selected: {stats['selected_specs']}",
        f"- Max attempts: {stats['max_attempts']}",
        f"- Category-cap skips: {stats['skipped_category_cap']}",
        f"- Categories covered: {len(category_counts)}",
        f"- Difficulty targets: {json.dumps(stats['difficulty_targets'], sort_keys=True)}",
        "",
        "## Matching Semantics",
        "",
        "- `user_requirements` are user-stated needs.",
        "- Every case has an explicit `intake_methods` requirement.",
        "- Difficulty is the number of non-location/category user requirements: easy=1, medium=2, hard=3.",
        "- `user_qualification` describes whether the user qualifies for resource-side eligibility, fee, and document requirements.",
        "- A case is kept only when exactly one resource satisfies both `user_requirements` and `user_qualification`.",
        "- Ground truth is embedded in each case spec as the singleton `ground_truth_resource_ids` field.",
        f"- Schedule requirements without schedule-relevant intake method: {schedule_without_intake}",
        "",
        "## Difficulty",
        "",
        *[f"- `{difficulty}`: {count}" for difficulty, count in difficulty_counts.most_common()],
        "",
        "## User Requirement Fields",
        "",
        *[f"- `{field}`: {count}" for field, count in requirement_fields.most_common()],
        "",
        *render_value_section("## User Requirement Values", requirement_values),
        "## Location Counties",
        "",
        *[f"- {county}: {count}" for county, count in county_counts.most_common()],
        "",
        "## Location Cities",
        "",
        *[f"- {city}: {count}" for city, count in city_counts.most_common()],
        "",
        "## Fee Capacity",
        "",
        *[f"- `{field}`: {count}" for field, count in fee_counts.most_common()],
        "",
        "## Eligibility Facts",
        "",
        *[f"- `{field}`: {count}" for field, count in eligibility_counts.most_common()],
        "",
        "## Document Facts",
        "",
        *[f"- `{field}`: {count}" for field, count in document_counts.most_common()],
        "",
        "## Primary Resource Schedule Status",
        "",
        *[f"- `{field}`: {count}" for field, count in resource_schedule_counts.most_common()],
        "",
        "## Primary Resource Intake Methods",
        "",
        *[f"- `{field}`: {count}" for field, count in resource_intake_counts.most_common()],
        "",
        "## Primary Resource Fee Options",
        "",
        *[f"- `{field}`: {count}" for field, count in resource_fee_counts.most_common()],
        "",
        "## Primary Resource Document Requirements",
        "",
        *[f"- `{field}`: {count}" for field, count in resource_document_counts.most_common()],
        "",
        "## Primary Resource Eligibility Tags",
        "",
        *[f"- `{field}`: {count}" for field, count in resource_eligibility_counts.most_common()],
        "",
        "## Categories",
        "",
        *[f"- {category}: {count}" for category, count in category_counts.most_common()],
        "",
    ]
    return "\n".join(lines)


def requirement_value_counts(specs: list[dict]) -> dict[str, Counter]:
    counts: dict[str, Counter] = {}
    for spec in specs:
        for field, value in spec["user_requirements"].items():
            if field in BASE_REQUIREMENT_FIELDS:
                continue
            counter = counts.setdefault(field, Counter())
            if isinstance(value, list):
                counter.update(value)
            elif isinstance(value, bool):
                counter.update([str(value).lower()])
            elif value:
                counter.update([str(value)])
    return counts


def render_value_section(title: str, values: dict[str, Counter]) -> list[str]:
    lines = [title, ""]
    for field in sorted(values):
        lines.append(f"### `{field}`")
        lines.append("")
        lines.extend(f"- `{value}`: {count}" for value, count in values[field].most_common())
        lines.append("")
    return lines


def count_schedule_requirements_without_relevant_intake(specs: list[dict]) -> int:
    count = 0
    for spec in specs:
        requirements = spec["user_requirements"]
        if not (set(requirements) & SCHEDULE_REQUIREMENT_FIELDS):
            continue
        if not (set(requirements.get("intake_methods") or ()) & SCHEDULE_RELEVANT_INTAKE):
            count += 1
    return count


def print_progress(attempted: int, max_attempts: int, selected: int, target: int, start_time: float) -> None:
    elapsed = max(time.monotonic() - start_time, 0.001)
    rate = attempted / elapsed
    remaining = (max_attempts - attempted) / rate if rate else 0
    width = 28
    filled = int(width * selected / target) if target else width
    bar = "#" * filled + "." * (width - filled)
    sys.stderr.write(
        f"\r[{bar}] selected={selected}/{target} "
        f"attempted={attempted}/{max_attempts} elapsed={elapsed:.1f}s eta={remaining:.1f}s"
    )
    sys.stderr.flush()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic benchmark case specs.")
    parser.add_argument("--index-path", type=Path, default=Path("data/indiana211/indiana211_resources_deduped.csv"))
    parser.add_argument("--easy", type=int, required=True)
    parser.add_argument("--medium", type=int, required=True)
    parser.add_argument("--hard", type=int, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--case-specs-out", type=Path, default=CASE_SPECS_PATH)
    parser.add_argument("--report-out", type=Path, default=REPORT_PATH)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def parse_difficulty_targets(args: argparse.Namespace) -> dict[str, int]:
    targets = {
        "easy": int(args.easy),
        "medium": int(args.medium),
        "hard": int(args.hard),
    }
    if any(value < 0 for value in targets.values()):
        raise SystemExit("--easy, --medium, and --hard must be non-negative.")
    if sum(targets.values()) <= 0:
        raise SystemExit("At least one of --easy, --medium, or --hard must be positive.")
    return targets


if __name__ == "__main__":
    main()
