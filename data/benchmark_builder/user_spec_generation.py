from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.indiana211 import Resource

MAX_ATTEMPT_MULTIPLIER = 120
LOCATION_REQUIREMENT_FIELDS = {"counties", "cities", "zipcodes"}
SCHEDULE_RELEVANT_INTAKE = {"call", "walk_in", "appointment", "text"}
SCHEDULE_REQUIREMENT_FIELDS = {
    "available_days",
    "available_time_windows",
    "requires_24_hours",
}
@dataclass(frozen=True)
class Candidate:
    resource: Resource
    category: str
    county: str


def build_user_specs(
    resources: list[Resource],
    difficulty_targets: dict[str, int],
    seed: int,
    progress_every: int = 100,
) -> list[dict]:
    validate_difficulty_targets(difficulty_targets)
    cases = sum(difficulty_targets.values())
    rng = random.Random(seed)
    candidates = build_candidates(resources)
    rng.shuffle(candidates)

    selected: list[dict] = []
    used_primary_ids: set[str] = set()
    category_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    category_total = len(all_categories(resources))
    max_per_category = max(1, (cases + category_total - 1) // max(category_total, 1))
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
        spec = sample_case_spec(resources, candidate, rng, difficulty)
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
    validate_case_specs(selected, resources)
    print(
        "Selected "
        f"{len(selected)}/{cases} specs from {len(resources)} filtered resources "
        f"after {attempted} candidate probes "
        f"(category-cap skips: {skipped_category_cap})."
    )
    return selected


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
    qualification = user_qualification_for_resource(resource) if difficulty == "hard" else {}
    requirement_options = user_requirement_options(resource, candidate, rng, difficulty)
    for requirements in requirement_options:
        requirement_matches = [
            item
            for item in resources
            if matches_user_requirements(item, requirements)
        ]
        if resource.resource_id not in {item.resource_id for item in requirement_matches}:
            continue
        if difficulty in {"easy", "medium"}:
            if len(requirement_matches) != 1:
                continue
            if requirement_matches[0].resource_id != resource.resource_id:
                continue
            qualified_matches = requirement_matches
        elif difficulty == "hard":
            qualified_matches = [
                item
                for item in requirement_matches
                if user_qualifies_for_resource(qualification, item)
            ]
            if len(requirement_matches) <= 1:
                continue
            if len(qualified_matches) != 1 or qualified_matches[0].resource_id != resource.resource_id:
                continue
        else:
            raise ValueError(f"Unknown difficulty: {difficulty}")
        return make_case_spec(
            resource,
            candidate,
            requirements,
            qualification,
            difficulty,
            requirement_matches,
            qualified_matches,
        )
    return None


def user_requirement_options(
    resource: Resource,
    candidate: Candidate,
    rng: random.Random,
    difficulty: str,
) -> list[dict]:
    location_extras = location_requirement_extras(resource, candidate, difficulty)
    intake_extras = intake_requirement_extras(resource)
    schedule_extras = schedule_requirement_extras(resource)
    rng.shuffle(location_extras)
    rng.shuffle(intake_extras)
    rng.shuffle(schedule_extras)
    options: list[dict] = []
    if not location_extras or not intake_extras:
        return []

    if difficulty == "easy":
        combos = [
            (location, intake)
            for location in location_extras
            for intake in intake_extras
        ]
    elif difficulty == "medium":
        if not schedule_extras:
            return []
        combos = [
            (location, intake, schedule)
            for location in location_extras
            for intake in intake_extras
            for schedule in schedule_extras
        ]
    elif difficulty == "hard":
        if not schedule_extras:
            return []
        county_locations = [item for item in location_extras if "counties" in item]
        combos = [
            (location, intake, schedule)
            for location in county_locations
            for intake in intake_extras
            for schedule in schedule_extras
        ]
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")

    base = {"service_categories": [candidate.category]}
    rng.shuffle(combos)
    for combo in combos:
        if repeated_fields(combo):
            continue
        if not requirement_combo_is_valid(combo):
            continue
        requirements = copy_requirements(base)
        for item in combo:
            requirements = merge_requirements(requirements, item)
        options.append(requirements)
    rng.shuffle(options)
    return options


def location_requirement_extras(resource: Resource, candidate: Candidate, difficulty: str) -> list[dict]:
    extras = [{"counties": [candidate.county]}]
    if difficulty in {"easy", "medium"}:
        if resource.city:
            extras.append({"cities": [resource.city]})
        if resource.zipcode:
            extras.append({"zipcodes": [resource.zipcode]})
    return dedupe_requirements(extras)


def user_qualification_for_resource(resource: Resource) -> dict:
    return {"documents_available": list(required_documents(resource))}


def user_qualifies_for_resource(qualification: dict, resource: Resource) -> bool:
    user_documents = set(qualification.get("documents_available") or ())
    return set(required_documents(resource)).issubset(user_documents)


def required_documents(resource: Resource) -> tuple[str, ...]:
    ignored = {"empty", "none", "varies"}
    return tuple(value for value in resource.document_requirements if value not in ignored)


def qualification_hard_reason(
    qualification: dict,
    requirement_matches: list[Resource],
    qualified_matches: list[Resource],
) -> dict:
    qualified_ids = {resource.resource_id for resource in qualified_matches}
    excluded = [resource for resource in requirement_matches if resource.resource_id not in qualified_ids]
    user_documents = set(qualification.get("documents_available") or ())
    return {
        "type": "document_unique",
        "excluded_by": {
            "documents": sum(
                1
                for resource in excluded
                if not set(required_documents(resource)).issubset(user_documents)
            )
        },
    }


def intake_requirement_extras(resource: Resource) -> list[dict]:
    extras = []
    for value in user_facing_intake_methods(resource.intake_methods):
        extras.append({"intake_methods": [value]})
    return dedupe_requirements(extras)


def user_facing_intake_methods(values: tuple[str, ...]) -> list[str]:
    return [value for value in values if value != "empty"]


def schedule_requirement_extras(resource: Resource) -> list[dict]:
    if resource.schedule_status != "structured":
        return []
    windows = resource.schedule_windows
    extras = []
    if any(is_24_hour_window(window) for window in windows):
        extras.append({"requires_24_hours": True})
    for day in sorted({window.day for window in windows}):
        extras.append({"available_days": [day]})
    for window in windows:
        if is_24_hour_window(window):
            continue
        midpoint = rounded_midpoint(window.start_minute, window.end_minute)
        if midpoint is not None:
            extras.append(
                {
                    "available_time_windows": [
                        {
                            "day": window.day,
                            "start": format_hhmm(midpoint),
                        }
                    ]
                }
            )
    return dedupe_requirements(extras)


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
        or requirements.get("available_time_windows")
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
    for requested_window in requirements.get("available_time_windows") or []:
        if not any(window_satisfies_request(window, requested_window) for window in windows):
            return False
    if requirements.get("requires_24_hours"):
        if not any(is_24_hour_window(window) for window in windows):
            return False
    return True


def window_satisfies_request(window, requested: dict) -> bool:
    if window.day != requested.get("day"):
        return False
    start = parse_hhmm(str(requested.get("start", "")))
    if start is None:
        return False
    end = parse_hhmm(str(requested.get("end", "")))
    if is_24_hour_window(window):
        return True
    if end is None:
        return window.start_minute <= start < window.end_minute
    return window.start_minute <= start and window.end_minute >= end


def make_case_spec(
    resource: Resource,
    candidate: Candidate,
    requirements: dict,
    qualification: dict,
    difficulty: str,
    requirement_matches: list[Resource],
    qualified_matches: list[Resource],
) -> dict:
    hard_reason = None
    if difficulty == "hard":
        hard_reason = qualification_hard_reason(qualification, requirement_matches, qualified_matches)
    return {
        "case_id": "",
        "difficulty": difficulty,
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
        "matching_diagnostics": {
            "requirements_only_match_count": len(requirement_matches),
            "qualified_match_count": len(qualified_matches),
            **({"hard_reason": hard_reason} if hard_reason else {}),
        },
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
        requirement_matches = [
            resource.resource_id
            for resource in resources
            if matches_user_requirements(resource, spec["user_requirements"])
        ]
        qualified_matches = [
            resource.resource_id
            for resource in resources
            if matches_user_requirements(resource, spec["user_requirements"])
            and user_qualifies_for_resource(spec["user_qualification"], resource)
        ]
        if spec["difficulty"] in {"easy", "medium"}:
            if requirement_matches != [primary_id]:
                raise RuntimeError(
                    f"{spec['difficulty']} case is not requirement-unique: "
                    f"{primary_id}; matches={requirement_matches[:5]}"
                )
        elif spec["difficulty"] == "hard":
            if len(requirement_matches) <= 1:
                raise RuntimeError(f"hard case is not document-hard: {primary_id}")
            if qualified_matches != [primary_id]:
                raise RuntimeError(
                    f"hard case is not document-unique: {primary_id}; "
                    f"matches={qualified_matches[:5]}"
                )
        else:
            raise RuntimeError(f"unknown difficulty: {spec['difficulty']}")
        if spec["location"]["county"] in {"STATEWIDE", "ALL"}:
            raise RuntimeError(f"invalid location county: {primary_id}")
        if primary_id not in by_id:
            raise RuntimeError(f"primary resource missing from index: {primary_id}")
        if not spec["user_requirements"].get("intake_methods"):
            raise RuntimeError(f"missing intake requirement: {primary_id}")
        validate_requirement_shape(spec)


def validate_requirement_shape(spec: dict) -> None:
    requirements = spec["user_requirements"]
    location_fields = set(requirements) & LOCATION_REQUIREMENT_FIELDS
    schedule_fields = set(requirements) & SCHEDULE_REQUIREMENT_FIELDS
    difficulty = spec["difficulty"]
    if len(location_fields) != 1:
        raise RuntimeError(f"{difficulty} case must have exactly one location requirement: {spec['case_id']}")
    if difficulty == "easy":
        if schedule_fields:
            raise RuntimeError(f"easy case should not have schedule requirements: {spec['case_id']}")
    elif difficulty == "medium":
        if len(schedule_fields) != 1:
            raise RuntimeError(f"medium case must have exactly one schedule requirement: {spec['case_id']}")
    elif difficulty == "hard":
        if location_fields != {"counties"}:
            raise RuntimeError(f"hard case must use county as the requirement location: {spec['case_id']}")
        if len(schedule_fields) != 1:
            raise RuntimeError(f"hard case must have exactly one schedule requirement: {spec['case_id']}")


def merge_requirements(left: dict, right: dict) -> dict:
    merged = copy_requirements(left)
    for key, value in right.items():
        if isinstance(value, list):
            merged[key] = merge_list_values(merged.get(key, []), value)
        elif isinstance(value, bool):
            merged[key] = bool(value or merged.get(key, False))
        elif value:
            merged[key] = value
    return merged


def merge_list_values(left: list, right: list) -> list:
    values = list(left) + list(right)
    if all(isinstance(value, str) for value in values):
        return sorted(set(values))
    deduped = []
    seen = set()
    for value in values:
        key = json.dumps(value, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


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


def any_exact(needles, haystack) -> bool:
    return bool(set(needles or ()) & set(haystack or ()))


def is_24_hour_window(window) -> bool:
    return window.start_minute == 0 and window.end_minute >= 24 * 60


def rounded_midpoint(start_minute: int, end_minute: int) -> int | None:
    if end_minute <= start_minute:
        return None
    midpoint = (start_minute + end_minute) // 2
    rounded = round(midpoint / 30) * 30
    if start_minute <= rounded < end_minute:
        return rounded
    if start_minute < midpoint < end_minute:
        return midpoint
    return None


def format_hhmm(minute: int) -> str:
    minute = max(0, min(minute, 23 * 60 + 59))
    return f"{minute // 60:02d}:{minute % 60:02d}"


def parse_hhmm(value: str) -> int | None:
    try:
        hour, minute = value.split(":", 1)
        parsed = int(hour) * 60 + int(minute)
    except (AttributeError, ValueError):
        return None
    if parsed < 0 or parsed >= 24 * 60:
        return None
    return parsed


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
        "schedule_status": resource.schedule_status,
        "schedule_windows": [
            {
                "day": window.day,
                "start": format_hhmm(window.start_minute),
                "end": format_hhmm(window.end_minute),
            }
            for window in resource.schedule_windows
        ],
        "intake_methods": list(resource.intake_methods),
        "document_requirements": list(resource.document_requirements),
    }


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


def validate_difficulty_targets(targets: dict[str, int]) -> None:
    missing = {"easy", "medium", "hard"} - set(targets)
    if missing:
        raise ValueError(f"Missing difficulty target(s): {', '.join(sorted(missing))}")
    if any(value < 0 for value in targets.values()):
        raise ValueError("Difficulty targets must be non-negative.")
    if sum(targets.values()) <= 0:
        raise ValueError("At least one difficulty target must be positive.")
