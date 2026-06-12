from __future__ import annotations

import random
from collections import defaultdict

from tools.indiana211 import Resource, search_resources, request_from_tool_args
from tools.indiana211_models import ResourceIndex
from tools.indiana211_schedule import ScheduleWindow, format_minutes, is_24_hour_window


OPTIONAL_FIELD_PROBABILITY = {
    "schedule": 0.70,
    "location": 1.00,
    "intake_methods": 0.55,
    "available_documents": 0.45,
    "eligibility": 0.45,
}

EXTRA_DOCUMENTS = (
    "photo_id",
    "proof_of_address",
    "proof_of_income",
    "utility_bill",
    "social_security",
    "insurance_card",
    "lease",
    "birth_certificate",
)
EXTRA_ELIGIBILITY = (
    "low_income",
    "resident",
    "homeless",
    "veteran",
    "senior",
    "youth",
    "family",
    "pregnant",
    "disability",
    "uninsured",
    "medicaid",
)
IGNORED_REQUIREMENTS = {"empty", "none", "varies", "unknown"}
CASE_TYPES = ("single", "composite")
CONSTRAINT_PROFILES = ("all_hard", "acceptable")


def build_user_specs(
    resources: list[Resource],
    count: int,
    seed: int,
    progress_every: int = 100,
) -> list[dict]:
    rng = random.Random(seed)
    index = ResourceIndex(resources)
    by_category = resources_by_category(resources)
    categories = sorted(by_category)
    if not categories:
        raise RuntimeError("No benchmark service categories available.")

    specs = []
    category_index = 0
    attempts = 0
    max_attempts = max(count * 50, 1000)
    while len(specs) < count and attempts < max_attempts:
        attempts += 1
        case_type = CASE_TYPES[len(specs) % len(CASE_TYPES)]
        constraint_profile = CONSTRAINT_PROFILES[(len(specs) // len(CASE_TYPES)) % len(CONSTRAINT_PROFILES)]
        category = categories[category_index % len(categories)]
        category_index += 1
        resource = rng.choice(by_category[category])
        spec = make_user_spec(resource, category, rng, case_type=case_type, constraint_profile=constraint_profile, by_category=by_category)
        if spec is None:
            continue
        if not source_resources_visible(index, spec):
            continue
        specs.append(spec)
        if progress_every and (len(specs) == 1 or len(specs) % progress_every == 0 or len(specs) == count):
            print(f"[user-specs] selected={len(specs)}/{count} attempts={attempts}")

    if len(specs) < count:
        raise RuntimeError(f"Only generated {len(specs)}/{count} user specs after {attempts} attempts.")
    for idx, spec in enumerate(specs, start=1):
        spec["user_spec_id"] = f"user-spec-{idx:03d}"
        spec["case_id"] = spec["user_spec_id"]
    print(f"Selected {len(specs)}/{count} user specs from {len(resources)} resources.")
    return specs


def resources_by_category(resources: list[Resource]) -> dict[str, list[Resource]]:
    by_category: dict[str, list[Resource]] = defaultdict(list)
    for resource in resources:
        for category in resource.service_categories:
            by_category[category].append(resource)
    return dict(by_category)


def make_user_spec(
    resource: Resource,
    category: str,
    rng: random.Random,
    case_type: str = "single",
    constraint_profile: str = "all_hard",
    by_category: dict[str, list[Resource]] | None = None,
) -> dict | None:
    needs = [make_need(resource, category, rng, constraint_profile, "need-1")]
    if case_type == "composite":
        second = sample_second_resource(resource, category, by_category or {}, rng)
        if second is None:
            return None
        second_resource, second_category = second
        needs.append(make_need(second_resource, second_category, rng, constraint_profile, "need-2"))
    ground_truth_resources = [
        {
            "need_id": need["need_id"],
            "resource_id": need["ground_truth_resource_id"],
            "resource_name": need["ground_truth_resource_name"],
            "service_categories": need["service_categories"],
        }
        for need in needs
    ]
    return {
        "user_spec_id": "",
        "case_id": "",
        "case_type": case_type,
        "constraint_profile": constraint_profile,
        "source_resource_id": resource.resource_id,
        "source_resource_ids": [need["ground_truth_resource_id"] for need in needs],
        "needs": needs,
        "ground_truth_resources": ground_truth_resources,
    }


def make_need(resource: Resource, category: str, rng: random.Random, constraint_profile: str, need_id: str) -> dict:
    location = sample_location(resource, rng, constraint_profile) if include_field("location", rng) else {}
    schedule = sample_schedule(resource, rng) if include_field("schedule", rng) else {}
    intake_methods = sample_intake(resource, rng, constraint_profile) if include_field("intake_methods", rng) else []
    available_documents = sample_available_documents(resource, rng) if include_field("available_documents", rng) else []
    eligibility = sample_eligibility(resource, rng) if include_field("eligibility", rng) else []
    return {
        "need_id": need_id,
        "ground_truth_resource_id": resource.resource_id,
        "ground_truth_resource_name": resource.service_name,
        "service_categories": [category],
        "schedule": schedule,
        "location": location,
        "intake_methods": intake_methods,
        "available_documents": available_documents,
        "eligibility": eligibility,
    }


def source_resources_visible(index: ResourceIndex, spec: dict, limit: int = 10) -> bool:
    for need in spec.get("needs") or []:
        source_id = need.get("ground_truth_resource_id")
        if not source_id:
            return False
        result_ids = [
            result.resource.resource_id
            for result in search_resources(index, request_from_tool_args(expected_tool_args_for_need(need)), limit=limit)
        ]
        if source_id not in result_ids:
            return False
    return True


def expected_tool_args_for_need(need: dict) -> dict:
    location = need.get("location") or {}
    return {
        "service_categories": need.get("service_categories") or [],
        "schedule": need.get("schedule") or {},
        "counties": location.get("counties") or [],
        "cities": location.get("cities") or [],
        "zipcodes": location.get("zipcodes") or [],
        "intake_methods": need.get("intake_methods") or [],
        "available_documents": need.get("available_documents") or [],
        "eligibility": need.get("eligibility") or [],
    }


def sample_second_resource(
    first_resource: Resource,
    first_category: str,
    by_category: dict[str, list[Resource]],
    rng: random.Random,
) -> tuple[Resource, str] | None:
    categories = [category for category in sorted(by_category) if category != first_category]
    rng.shuffle(categories)
    for category in categories:
        candidates = [item for item in by_category[category] if item.resource_id != first_resource.resource_id]
        if candidates:
            return rng.choice(candidates), category
    return None


def include_field(field: str, rng: random.Random) -> bool:
    return rng.random() < OPTIONAL_FIELD_PROBABILITY[field]


def sample_location(resource: Resource, rng: random.Random, constraint_profile: str = "all_hard") -> dict:
    if constraint_profile == "acceptable":
        result = {}
        if resource.zipcode:
            result["zipcodes"] = [resource.zipcode]
        if resource.city:
            result["cities"] = [resource.city]
        if resource.counties:
            result["counties"] = [rng.choice(resource.counties)]
        return result
    options = []
    if resource.counties:
        options.append({"counties": [rng.choice(resource.counties)]})
    if resource.city:
        options.append({"cities": [resource.city]})
    if resource.zipcode:
        options.append({"zipcodes": [resource.zipcode]})
    return rng.choice(options) if options else {}


def sample_schedule(resource: Resource, rng: random.Random) -> dict:
    windows = tuple(resource.schedule_windows)
    if not windows:
        return {}
    if any(is_24_hour_window(window) for window in windows) and rng.random() < 0.15:
        return {"requires_24_hours": True}
    window = rng.choice([item for item in windows if not is_24_hour_window(item)] or list(windows))
    return schedule_window_requirement(window, rng)


def schedule_window_requirement(window: ScheduleWindow, rng: random.Random) -> dict:
    if is_24_hour_window(window):
        return {"requires_24_hours": True}
    start = window.start_minute
    end = window.end_minute
    if end - start > 120:
        latest_start = end - 60
        start = rng.randrange(start, latest_start + 1, 30)
        end = min(end, start + rng.choice((60, 90, 120)))
    style = rng.choice(("window", "time", "day"))
    if style == "day":
        return {"day": window.day}
    if style == "time":
        time = rng.randrange(start, max(start + 1, end), 30)
        return {"day": window.day, "time": format_minutes(time)}
    return {
        "day": window.day,
        "start_time": format_minutes(start),
        "end_time": format_minutes(end),
    }


def sample_intake(resource: Resource, rng: random.Random, constraint_profile: str = "all_hard") -> list[str]:
    methods = [method for method in resource.intake_methods if method != "empty"]
    if not methods:
        return []
    if constraint_profile == "acceptable" and len(methods) > 1:
        return rng.sample(methods, k=min(len(methods), rng.choice((2, 3))))
    return [rng.choice(methods)]


def sample_available_documents(resource: Resource, rng: random.Random) -> list[str]:
    required = [doc for doc in resource.document_requirements if doc not in IGNORED_REQUIREMENTS]
    extras = rng.sample(EXTRA_DOCUMENTS, k=rng.randint(0, 2))
    return dedupe([*required, *extras])


def sample_eligibility(resource: Resource, rng: random.Random) -> list[str]:
    required = [tag for tag in resource.eligibility_tags if tag not in IGNORED_REQUIREMENTS]
    extras = rng.sample(EXTRA_ELIGIBILITY, k=rng.randint(0, 2))
    return dedupe([*required, *extras])


def dedupe(values: list[str]) -> list[str]:
    result = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result
