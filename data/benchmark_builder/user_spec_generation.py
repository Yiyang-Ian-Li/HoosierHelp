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


MAX_ATTEMPT_MULTIPLIER = 160
LOCATION_REQUIREMENT_FIELDS = {"counties", "cities", "zipcodes"}
SCHEDULE_REQUIREMENT_FIELDS = {
    "day",
    "requires_24_hours",
}
TIME_VALUES = {"any", "morning", "afternoon"}
DAY_PARTS = {
    "morning": (5 * 60, 12 * 60),
    "afternoon": (12 * 60, 17 * 60),
}
IGNORED_DOCUMENTS = {"empty", "none", "varies"}


@dataclass(frozen=True)
class Candidate:
    resource: Resource
    category: str
    county: str


def build_user_specs(
    resources: list[Resource],
    case_type_targets: dict[str, int],
    seed: int,
    progress_every: int = 100,
) -> list[dict]:
    validate_case_type_targets(case_type_targets)
    cases = sum(case_type_targets.values())
    rng = random.Random(seed)
    candidates = build_candidates(resources)
    rng.shuffle(candidates)

    selected: list[dict] = []
    used_resource_ids: set[str] = set()
    category_counts: Counter[str] = Counter()
    case_type_counts: Counter[str] = Counter()
    category_total = len(all_categories(resources))
    max_per_category = max(1, (cases + category_total - 1) // max(category_total, 1))
    max_attempts = max(len(candidates), cases * MAX_ATTEMPT_MULTIPLIER)
    attempted = 0
    skipped_category_cap = 0
    start_time = time.monotonic()

    while len(selected) < cases and attempted < max_attempts and candidates:
        candidate = candidates[attempted % len(candidates)]
        attempted += 1
        if candidate.resource.resource_id in used_resource_ids:
            continue
        if category_counts[candidate.category] >= max_per_category and len(selected) < cases // 2:
            skipped_category_cap += 1
            continue

        case_type = choose_needed_case_type(case_type_targets, case_type_counts, rng)
        if not case_type:
            break
        spec = sample_case_spec(resources, candidates, candidate, rng, case_type, used_resource_ids)
        if not spec:
            continue

        selected.append(spec)
        for resource_id in spec["ground_truth_resource_ids"]:
            used_resource_ids.add(resource_id)
        for category in spec["target_service_categories"]:
            category_counts[category] += 1
        case_type_counts[spec["case_type"]] += 1
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
    candidates: list[Candidate],
    candidate: Candidate,
    rng: random.Random,
    case_type: str,
    used_resource_ids: set[str],
) -> dict | None:
    if case_type == "single":
        return sample_single_case(resources, candidate, rng)
    if case_type == "composite":
        return sample_composite_case(resources, candidates, candidate, rng, used_resource_ids)
    raise ValueError(f"Unknown case_type: {case_type}")


def sample_single_case(resources: list[Resource], candidate: Candidate, rng: random.Random) -> dict | None:
    for location in shuffled(location_requirement_options(candidate.resource, candidate), rng):
        for schedule in shuffled(schedule_requirement_options(candidate.resource), rng):
            need = make_need(candidate.resource, candidate.category, schedule)
            matches = matching_resources_for_need(resources, need, location)
            if [resource.resource_id for resource in matches] != [candidate.resource.resource_id]:
                continue
            return make_case_spec(
                case_type="single",
                location=location,
                needs=[need],
                resources=[candidate.resource],
                requirement_matches=[matches],
            )
    return None


def sample_composite_case(
    resources: list[Resource],
    candidates: list[Candidate],
    primary: Candidate,
    rng: random.Random,
    used_resource_ids: set[str],
) -> dict | None:
    primary_locations = shuffled(location_requirement_options(primary.resource, primary), rng)
    primary_schedules = shuffled(schedule_requirement_options(primary.resource), rng)
    secondary_candidates = shuffled(
        [
            candidate
            for candidate in candidates
            if candidate.resource.resource_id != primary.resource.resource_id
            and candidate.resource.resource_id not in used_resource_ids
            and candidate.category != primary.category
            and shares_location(candidate.resource, primary_locations)
        ],
        rng,
    )

    for location in primary_locations:
        for primary_schedule in primary_schedules:
            primary_need = make_need(primary.resource, primary.category, primary_schedule)
            primary_matches = matching_resources_for_need(resources, primary_need, location)
            if [resource.resource_id for resource in primary_matches] != [primary.resource.resource_id]:
                continue
            for secondary in secondary_candidates:
                if not location_matches_resource(location, secondary.resource):
                    continue
                for secondary_schedule in shuffled(schedule_requirement_options(secondary.resource), rng):
                    if schedules_overlap(primary_schedule, secondary_schedule):
                        continue
                    secondary_need = make_need(secondary.resource, secondary.category, secondary_schedule)
                    secondary_matches = matching_resources_for_need(resources, secondary_need, location)
                    if [resource.resource_id for resource in secondary_matches] != [secondary.resource.resource_id]:
                        continue
                    return make_case_spec(
                        case_type="composite",
                        location=location,
                        needs=[primary_need, secondary_need],
                        resources=[primary.resource, secondary.resource],
                        requirement_matches=[primary_matches, secondary_matches],
                    )
    return None


def make_need(resource: Resource, category: str, schedule: dict) -> dict:
    return {
        "service_categories": [category],
        "schedule": copy_requirements(schedule),
        "ground_truth_resource_id": resource.resource_id,
    }


def location_requirement_options(resource: Resource, candidate: Candidate) -> list[dict]:
    options = [{"counties": [candidate.county]}]
    if resource.city:
        options.append({"cities": [resource.city]})
    if resource.zipcode:
        options.append({"zipcodes": [resource.zipcode]})
    return dedupe_requirements(options)


def schedule_requirement_options(resource: Resource) -> list[dict]:
    if resource.schedule_status != "structured":
        return []
    options = []
    if any(is_24_hour_window(window) for window in resource.schedule_windows):
        options.append({"requires_24_hours": True})
    for day in sorted({window.day for window in resource.schedule_windows}):
        options.append({"day": day, "time": "any"})
    for day_part in ("morning", "afternoon"):
        start, end = DAY_PARTS[day_part]
        for day in sorted(
            {
                window.day
                for window in resource.schedule_windows
                if window_overlaps_range(window, start, end)
            }
        ):
            options.append({"day": day, "time": day_part})
    return dedupe_requirements(options)


def matching_resources_for_need(resources: list[Resource], need: dict, location: dict) -> list[Resource]:
    return [
        resource
        for resource in resources
        if matches_need(resource, need, location)
    ]


def matches_need(resource: Resource, need: dict, location: dict) -> bool:
    return (
        any_exact(need.get("service_categories"), resource.service_categories)
        and location_matches_resource(location, resource)
        and matches_schedule_requirements(resource, need["schedule"])
    )


def location_matches_resource(location: dict, resource: Resource) -> bool:
    if location.get("counties") and not matches_county(location["counties"], resource):
        return False
    if location.get("cities") and not any_exact(location["cities"], (resource.city,)):
        return False
    if location.get("zipcodes") and not any_exact(location["zipcodes"], (resource.zipcode,)):
        return False
    return True


def matches_county(counties: list[str] | None, resource: Resource) -> bool:
    if not counties:
        return True
    return any_exact(counties, resource.service_area) or any_exact(("STATEWIDE", "ALL"), resource.service_area)


def matches_schedule_requirements(resource: Resource, requirements: dict) -> bool:
    if resource.schedule_status != "structured":
        return False
    windows = resource.schedule_windows
    if requirements.get("day"):
        return any(window_satisfies_schedule(window, requirements) for window in windows)
    if requirements.get("requires_24_hours"):
        return any(is_24_hour_window(window) for window in windows)
    return False


def window_satisfies_schedule(window, schedule: dict) -> bool:
    if window.day != schedule.get("day"):
        return False
    day_part = schedule.get("time", "any")
    if day_part == "any":
        return True
    start, end = DAY_PARTS[day_part]
    return is_24_hour_window(window) or window_overlaps_range(window, start, end)


def window_overlaps_range(window, start: int, end: int) -> bool:
    return window.start_minute < end and window.end_minute > start


def schedules_overlap(left: dict, right: dict) -> bool:
    if left.get("requires_24_hours") or right.get("requires_24_hours"):
        return False
    return bool(schedule_slots(left) & schedule_slots(right))


def schedule_slots(schedule: dict) -> set[tuple[str, str]]:
    if schedule.get("requires_24_hours"):
        return set()
    day = schedule.get("day")
    if not day:
        return set()
    if schedule.get("time") == "any":
        return {(day, part) for part in ("morning", "afternoon", "all")}
    return {(day, schedule.get("time"))}


def shares_location(resource: Resource, locations: list[dict]) -> bool:
    return any(location_matches_resource(location, resource) for location in locations)


def make_case_spec(
    case_type: str,
    location: dict,
    needs: list[dict],
    resources: list[Resource],
    requirement_matches: list[list[Resource]],
) -> dict:
    return {
        "case_id": "",
        "case_type": case_type,
        "location": location_background(resources[0], location),
        "location_requirement": copy_requirements(location),
        "needs": numbered_needs(needs),
        "ground_truth_resource_ids": [resource.resource_id for resource in resources],
        "ground_truth_resources": [ground_truth_resource(resource) for resource in resources],
        "target_service_categories": [
            category
            for need in needs
            for category in need["service_categories"]
        ],
        "matching_diagnostics": {
            "need_match_counts": [len(matches) for matches in requirement_matches],
        },
    }


def numbered_needs(needs: list[dict]) -> list[dict]:
    return [
        {
            "need_id": f"need_{idx}",
            **copy_requirements(need),
        }
        for idx, need in enumerate(needs, start=1)
    ]


def location_background(resource: Resource, location: dict) -> dict:
    county = (location.get("counties") or sampled_counties(resource) or [""])[0]
    return {
        "county": county,
        "city": resource.city,
        "state": "IN",
        "zipcode": resource.zipcode,
    }


def ground_truth_resource(resource: Resource) -> dict:
    return {
        "resource_id": resource.resource_id,
        "service_name": resource.service_name,
        "agency_name": resource.agency_name,
        "intake_methods": user_facing_intake_methods(resource.intake_methods),
        "document_requirements": list(required_documents(resource)),
    }


def user_facing_intake_methods(values: tuple[str, ...]) -> list[str]:
    return [value for value in values if value != "empty"]


def required_documents(resource: Resource) -> tuple[str, ...]:
    return tuple(value for value in resource.document_requirements if value not in IGNORED_DOCUMENTS)


def validate_case_specs(specs: list[dict], resources: list[Resource]) -> None:
    seen = set()
    by_id = {resource.resource_id: resource for resource in resources}
    for spec in specs:
        case_type = spec.get("case_type")
        ground_truth_ids = spec.get("ground_truth_resource_ids") or []
        expected_count = 1 if case_type == "single" else 2 if case_type == "composite" else 0
        if len(ground_truth_ids) != expected_count:
            raise RuntimeError(f"{case_type} case has invalid ground truth count: {spec.get('case_id')}")
        if len(set(ground_truth_ids)) != len(ground_truth_ids):
            raise RuntimeError(f"duplicate ground truth inside case: {spec.get('case_id')}")
        for resource_id in ground_truth_ids:
            if resource_id in seen:
                raise RuntimeError(f"duplicate ground truth resource across cases: {resource_id}")
            if resource_id not in by_id:
                raise RuntimeError(f"ground truth resource missing from index: {resource_id}")
            seen.add(resource_id)
        validate_requirement_shape(spec)
        if len(spec.get("ground_truth_resources") or []) != expected_count:
            raise RuntimeError(f"missing structured ground truth resources: {spec.get('case_id')}")
        for need, expected_id in zip(spec["needs"], ground_truth_ids):
            matches = [
                resource.resource_id
                for resource in resources
                if matches_need(resource, need, spec["location_requirement"])
            ]
            if matches != [expected_id]:
                raise RuntimeError(
                    f"{case_type} need is not unique: {expected_id}; matches={matches[:5]}"
                )
        if case_type == "composite" and schedules_overlap(spec["needs"][0]["schedule"], spec["needs"][1]["schedule"]):
            raise RuntimeError(f"composite schedules overlap: {spec.get('case_id')}")


def validate_requirement_shape(spec: dict) -> None:
    location_fields = set(spec.get("location_requirement") or {}) & LOCATION_REQUIREMENT_FIELDS
    if len(location_fields) != 1:
        raise RuntimeError(f"case must have exactly one location requirement: {spec.get('case_id')}")
    needs = spec.get("needs") or []
    if spec.get("case_type") == "single" and len(needs) != 1:
        raise RuntimeError(f"single case must have exactly one need: {spec.get('case_id')}")
    if spec.get("case_type") == "composite" and len(needs) != 2:
        raise RuntimeError(f"composite case must have exactly two needs: {spec.get('case_id')}")
    for need in needs:
        if not isinstance(need.get("need_id"), str) or not need["need_id"].strip():
            raise RuntimeError(f"need must have a need_id: {spec.get('case_id')}")
        if len(need.get("service_categories") or []) != 1:
            raise RuntimeError(f"need must have exactly one service category: {spec.get('case_id')}")
        schedule_fields = set(need.get("schedule") or {}) & SCHEDULE_REQUIREMENT_FIELDS
        if len(schedule_fields) != 1:
            raise RuntimeError(f"need must have exactly one schedule requirement: {spec.get('case_id')}")
        schedule = need["schedule"]
        if schedule.get("day") and schedule.get("day") not in DAY_VALUES():
            raise RuntimeError(f"invalid schedule day: {spec.get('case_id')}")
        if schedule.get("time") and schedule.get("time") not in TIME_VALUES:
            raise RuntimeError(f"invalid schedule time: {spec.get('case_id')}")


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


def sampled_counties(resource: Resource) -> list[str]:
    counties = [county for county in resource.service_area if county not in {"STATEWIDE", "ALL"}]
    return counties[:3]


def any_exact(needles, haystack) -> bool:
    return bool(set(needles or ()) & set(haystack or ()))


def is_24_hour_window(window) -> bool:
    return window.start_minute == 0 and window.end_minute >= 24 * 60


def choose_needed_case_type(
    targets: dict[str, int],
    counts: Counter[str],
    rng: random.Random,
) -> str | None:
    remaining = [
        case_type
        for case_type, target in targets.items()
        if counts[case_type] < target
    ]
    if not remaining:
        return None
    weights = [targets[case_type] - counts[case_type] for case_type in remaining]
    return rng.choices(remaining, weights=weights, k=1)[0]


def all_categories(resources: list[Resource]) -> set[str]:
    return {category for resource in resources for category in resource.service_categories}


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


def validate_case_type_targets(targets: dict[str, int]) -> None:
    missing = {"single", "composite"} - set(targets)
    if missing:
        raise ValueError(f"Missing case type target(s): {', '.join(sorted(missing))}")
    if any(value < 0 for value in targets.values()):
        raise ValueError("Case type targets must be non-negative.")
    if sum(targets.values()) <= 0:
        raise ValueError("At least one case type target must be positive.")


def shuffled(items: list, rng: random.Random) -> list:
    copied = list(items)
    rng.shuffle(copied)
    return copied


def DAY_VALUES() -> tuple[str, ...]:
    return ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
