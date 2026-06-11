from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.indiana211 import Resource
from tools.indiana211_schedule import ScheduleWindow, format_minutes, is_24_hour_window


OPTIONAL_FIELD_PROBABILITY = {
    "schedule": 0.70,
    "location": 0.80,
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


def build_user_specs(
    resources: list[Resource],
    count: int,
    seed: int,
    progress_every: int = 100,
) -> list[dict]:
    rng = random.Random(seed)
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
        category = categories[category_index % len(categories)]
        category_index += 1
        resource = rng.choice(by_category[category])
        spec = make_user_spec(resource, category, rng)
        if spec is None:
            continue
        specs.append(spec)
        if progress_every and (len(specs) == 1 or len(specs) % progress_every == 0 or len(specs) == count):
            print(f"[user-specs] selected={len(specs)}/{count} attempts={attempts}")

    if len(specs) < count:
        raise RuntimeError(f"Only generated {len(specs)}/{count} user specs after {attempts} attempts.")
    for idx, spec in enumerate(specs, start=1):
        spec["user_spec_id"] = f"user-spec-{idx:03d}"
    print(f"Selected {len(specs)}/{count} user specs from {len(resources)} resources.")
    return specs


def resources_by_category(resources: list[Resource]) -> dict[str, list[Resource]]:
    by_category: dict[str, list[Resource]] = defaultdict(list)
    for resource in resources:
        for category in resource.service_categories:
            by_category[category].append(resource)
    return dict(by_category)


def make_user_spec(resource: Resource, category: str, rng: random.Random) -> dict | None:
    location = sample_location(resource, rng) if include_field("location", rng) else {}
    schedule = sample_schedule(resource, rng) if include_field("schedule", rng) else {}
    intake_methods = sample_intake(resource, rng) if include_field("intake_methods", rng) else []
    available_documents = sample_available_documents(resource, rng) if include_field("available_documents", rng) else []
    eligibility = sample_eligibility(resource, rng) if include_field("eligibility", rng) else []
    return {
        "user_spec_id": "",
        "source_resource_id": resource.resource_id,
        "service_name": resource.service_name,
        "service_category": category,
        "schedule": schedule,
        "location": location,
        "intake_methods": intake_methods,
        "available_documents": available_documents,
        "eligibility": eligibility,
    }


def include_field(field: str, rng: random.Random) -> bool:
    return rng.random() < OPTIONAL_FIELD_PROBABILITY[field]


def sample_location(resource: Resource, rng: random.Random) -> dict:
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
    return {
        "day": window.day,
        "start_time": format_minutes(start),
        "end_time": format_minutes(end),
    }


def sample_intake(resource: Resource, rng: random.Random) -> list[str]:
    methods = [method for method in resource.intake_methods if method != "empty"]
    if not methods:
        return []
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
