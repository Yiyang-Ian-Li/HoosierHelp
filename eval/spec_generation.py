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

IGNORED_REQUIREMENTS = {"empty", "none", "varies", "unknown"}
CASE_TYPES = ("single", "composite")
PREFERENCE_PROFILES = ("strict", "flexible")
PREFERRED_CANDIDATE_SCAN_LIMIT = 200


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
        preference_profile = PREFERENCE_PROFILES[(len(specs) // len(CASE_TYPES)) % len(PREFERENCE_PROFILES)]
        category = categories[category_index % len(categories)]
        category_index += 1
        resource = rng.choice(by_category[category])
        spec = make_user_spec(
            resource,
            category,
            rng,
            case_type=case_type,
            preference_profile=preference_profile,
            by_category=by_category,
            all_resources=resources,
        )
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
    preference_profile: str = "strict",
    by_category: dict[str, list[Resource]] | None = None,
    all_resources: list[Resource] | None = None,
) -> dict | None:
    if case_type == "composite":
        composite = make_composite_needs(resource, category, rng, preference_profile, by_category or {}, all_resources or [])
        if composite is None:
            return None
        needs = composite
    else:
        needs = [make_need(resource, category, rng, preference_profile, "need-1", all_resources or [])]
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
        "preference_profile": preference_profile,
        "source_resource_id": resource.resource_id,
        "source_resource_ids": [need["ground_truth_resource_id"] for need in needs],
        "needs": needs,
        "ground_truth_resources": ground_truth_resources,
    }


def make_composite_needs(
    resource: Resource,
    category: str,
    rng: random.Random,
    preference_profile: str,
    by_category: dict[str, list[Resource]],
    all_resources: list[Resource],
) -> list[dict] | None:
    second = sample_second_resource_with_shared_location(resource, category, by_category, rng)
    if second is None:
        return None
    second_resource, second_category = second
    shared_context = sample_shared_user_context(resource, second_resource, rng)
    if shared_context is None:
        return None
    if not resource.schedule_windows or not second_resource.schedule_windows:
        return None
    needs = [
        make_need(resource, category, rng, "strict", "need-1", all_resources, shared_context=shared_context, require_schedule=True),
        make_need(
            second_resource,
            second_category,
            rng,
            "strict",
            "need-2",
            all_resources,
            shared_context=shared_context,
            require_schedule=True,
        ),
    ]
    if preference_profile == "flexible":
        add_composite_preferred_constraints(needs, all_resources, rng)
    return needs


def make_need(
    resource: Resource,
    category: str,
    rng: random.Random,
    preference_profile: str,
    need_id: str,
    all_resources: list[Resource],
    shared_context: dict | None = None,
    require_schedule: bool = False,
) -> dict:
    shared_context = shared_context or {}
    location = shared_context.get("location")
    if location is None:
        location = sample_location(resource, rng) if include_field("location", rng) else {}
    schedule = sample_schedule(resource, rng) if (require_schedule or include_field("schedule", rng)) else {}
    intake_methods = shared_context.get("intake_methods")
    if intake_methods is None:
        intake_methods = sample_intake(resource, rng) if include_field("intake_methods", rng) else []
    available_documents = shared_context.get("available_documents")
    if available_documents is None:
        available_documents = sample_available_documents(resource, rng) if include_field("available_documents", rng) else []
    eligibility = shared_context.get("eligibility")
    if eligibility is None:
        eligibility = sample_eligibility(resource, rng) if include_field("eligibility", rng) else []
    need = {
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
    if preference_profile == "flexible":
        preferred = sample_unavailable_preferred_constraints(need, resource, all_resources, rng)
        if preferred is None:
            return need
        need["preferred"] = preferred
    return need


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
        preferred = need.get("preferred")
        if preferred:
            preferred_ids = [
                result.resource.resource_id
                for result in search_resources(index, request_from_tool_args(expected_tool_args_for_need(preferred)), limit=limit)
            ]
            if preferred_ids:
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


def sample_second_resource_with_shared_location(
    first_resource: Resource,
    first_category: str,
    by_category: dict[str, list[Resource]],
    rng: random.Random,
) -> tuple[Resource, str] | None:
    categories = [category for category in sorted(by_category) if category != first_category]
    rng.shuffle(categories)
    for category in categories:
        candidates = [
            item
            for item in by_category[category]
            if item.resource_id != first_resource.resource_id
            and first_resource.schedule_windows
            and item.schedule_windows
            and common_location_options(first_resource, item)
        ]
        rng.shuffle(candidates)
        if candidates:
            return candidates[0], category
    return None


def include_field(field: str, rng: random.Random) -> bool:
    return rng.random() < OPTIONAL_FIELD_PROBABILITY[field]


def sample_shared_user_context(first: Resource, second: Resource, rng: random.Random) -> dict | None:
    location_options = common_location_options(first, second)
    if not location_options:
        return None
    context = {
        "location": rng.choice(location_options),
        "intake_methods": [],
        "available_documents": [],
        "eligibility": [],
    }
    if include_field("intake_methods", rng):
        context["intake_methods"] = sample_common_intake(first, second, rng)
    if include_field("available_documents", rng):
        context["available_documents"] = dedupe(
            concrete_requirements(first.document_requirements) + concrete_requirements(second.document_requirements)
        )
    if include_field("eligibility", rng):
        context["eligibility"] = dedupe(
            concrete_requirements(first.eligibility_tags) + concrete_requirements(second.eligibility_tags)
        )
    return context


def common_location_options(first: Resource, second: Resource) -> list[dict]:
    options = []
    common_counties = sorted(set(first.counties) & set(second.counties))
    if common_counties:
        options.extend({"counties": [county]} for county in common_counties)
    if first.city and first.city == second.city:
        options.append({"cities": [first.city]})
    if first.zipcode and first.zipcode == second.zipcode:
        options.append({"zipcodes": [first.zipcode]})
    return options


def sample_common_intake(first: Resource, second: Resource, rng: random.Random) -> list[str]:
    common = sorted(
        method
        for method in set(first.intake_methods) & set(second.intake_methods)
        if method != "empty"
    )
    return [rng.choice(common)] if common else []


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


def sample_intake(resource: Resource, rng: random.Random) -> list[str]:
    methods = [method for method in resource.intake_methods if method != "empty"]
    if not methods:
        return []
    return [rng.choice(methods)]


def sample_unavailable_preferred_constraints(
    fallback_need: dict,
    resource: Resource,
    all_resources: list[Resource],
    rng: random.Random,
) -> dict | tuple[dict, str] | None:
    return sample_unavailable_preferred_constraints_for_fields(
        fallback_need,
        resource,
        all_resources,
        rng,
        allowed_fields=("location", "schedule", "intake"),
        return_changed_field=False,
    )


def sample_unavailable_preferred_constraints_for_fields(
    fallback_need: dict,
    resource: Resource,
    all_resources: list[Resource],
    rng: random.Random,
    allowed_fields: tuple[str, ...],
    return_changed_field: bool = False,
) -> dict | tuple[dict, str] | None:
    field_samplers = [
        ("location", lambda: sample_wrong_location(fallback_need, resource, all_resources, rng)),
        ("schedule", lambda: sample_wrong_schedule(resource, all_resources, rng)),
        ("intake", lambda: sample_wrong_intake(resource, all_resources, rng)),
    ]
    field_samplers = [(field, sampler) for field, sampler in field_samplers if field in allowed_fields]
    rng.shuffle(field_samplers)
    for field, sampler in field_samplers:
        value = sampler()
        if not value:
            continue
        preferred = copy_need_search_fields(fallback_need)
        if field == "location":
            preferred["location"] = value
        elif field == "schedule":
            preferred["schedule"] = value
        elif field == "intake":
            preferred["intake_methods"] = value
        if return_changed_field:
            return preferred, field
        return preferred
    return None


def add_composite_preferred_constraints(needs: list[dict], all_resources: list[Resource], rng: random.Random) -> None:
    if add_per_need_preferred_constraints(needs, all_resources, rng):
        return
    field_samplers = [
        ("location", lambda: sample_wrong_shared_location(needs, all_resources, rng)),
        ("schedule", lambda: sample_wrong_schedules_for_needs(needs, all_resources, rng)),
        ("intake", lambda: sample_wrong_shared_intake(needs, all_resources, rng)),
    ]
    rng.shuffle(field_samplers)
    for field, sampler in field_samplers:
        value = sampler()
        if not value:
            continue
        for index, need in enumerate(needs):
            preferred = copy_need_search_fields(need)
            if field == "location":
                preferred["location"] = value
            elif field == "schedule":
                preferred["schedule"] = value[index]
            elif field == "intake":
                preferred["intake_methods"] = value
            need["preferred"] = preferred
        return


def add_per_need_preferred_constraints(needs: list[dict], all_resources: list[Resource], rng: random.Random) -> bool:
    resources_by_id = {resource.resource_id: resource for resource in all_resources}
    resources = [resources_by_id.get(need.get("ground_truth_resource_id")) for need in needs]
    if any(resource is None for resource in resources):
        return False
    field_plans = [
        ("schedule", "intake"),
        ("intake", "schedule"),
        ("schedule", "schedule"),
        ("intake", "intake"),
    ]
    rng.shuffle(field_plans)
    field_plans.sort(key=lambda plan: len(set(plan)) == 1)
    for field_plan in field_plans:
        preferred_needs = []
        for need, resource, field in zip(needs, resources, field_plan):
            preferred = sample_unavailable_preferred_constraints_for_fields(
                need,
                resource,
                all_resources,
                rng,
                allowed_fields=(field,),
            )
            if preferred is None:
                preferred_needs = []
                break
            preferred_needs.append(preferred)
        if preferred_needs:
            for need, preferred in zip(needs, preferred_needs):
                need["preferred"] = preferred
            return True
    return False


def copy_need_search_fields(need: dict) -> dict:
    return {
        "service_categories": list(need.get("service_categories") or []),
        "schedule": dict(need.get("schedule") or {}),
        "location": dict(need.get("location") or {}),
        "intake_methods": list(need.get("intake_methods") or []),
        "available_documents": list(need.get("available_documents") or []),
        "eligibility": list(need.get("eligibility") or []),
    }


def sample_wrong_shared_location(needs: list[dict], all_resources: list[Resource], rng: random.Random) -> dict:
    index = ResourceIndex(all_resources)
    fallback_locations = [need.get("location") or {} for need in needs]
    candidates = [
        resource
        for resource in all_resources
        if resource.city
        and resource.zipcode
        and resource.counties
    ]
    rng.shuffle(candidates)
    for resource in candidates[:PREFERRED_CANDIDATE_SCAN_LIMIT]:
        county = rng.choice(resource.counties)
        variants = (
            {"zipcodes": [resource.zipcode]},
            {"cities": [resource.city]},
            {"counties": [county]},
        )
        variant_order = list(variants)
        rng.shuffle(variant_order)
        for location in variant_order:
            if location in fallback_locations:
                continue
            if all(preferred_search_empty(need, {"location": location}, index) for need in needs):
                return location
    return {}


def sample_wrong_schedules_for_needs(needs: list[dict], all_resources: list[Resource], rng: random.Random) -> list[dict]:
    schedules = []
    for need in needs:
        schedule = sample_wrong_schedule_for_need(need, all_resources, rng)
        if not schedule:
            return []
        schedules.append(schedule)
    return schedules


def sample_wrong_schedule_for_need(need: dict, all_resources: list[Resource], rng: random.Random) -> dict:
    index = ResourceIndex(all_resources)
    windows = [window for resource in all_resources for window in resource.schedule_windows]
    rng.shuffle(windows)
    for window in windows[:PREFERRED_CANDIDATE_SCAN_LIMIT]:
        schedule = schedule_window_requirement(window, rng)
        if schedule != (need.get("schedule") or {}) and preferred_search_empty(need, {"schedule": schedule}, index):
            return schedule
    return {}


def sample_wrong_shared_intake(needs: list[dict], all_resources: list[Resource], rng: random.Random) -> list[str]:
    index = ResourceIndex(all_resources)
    fallback_methods = set()
    for need in needs:
        fallback_methods.update(need.get("intake_methods") or [])
    candidates = sorted(
        {
            method
            for resource in all_resources
            for method in resource.intake_methods
            if method != "empty" and method not in fallback_methods
        }
    )
    rng.shuffle(candidates)
    for method in candidates:
        value = [method]
        if all(preferred_search_empty(need, {"intake_methods": value}, index) for need in needs):
            return value
    return []


def preferred_search_empty(need: dict, updates: dict, index: ResourceIndex) -> bool:
    preferred = copy_need_search_fields(need)
    preferred.update(updates)
    return not search_resources(index, request_from_tool_args(expected_tool_args_for_need(preferred)), limit=1)


def sample_wrong_location(fallback_need: dict, resource: Resource, all_resources: list[Resource], rng: random.Random) -> dict:
    fallback_counties = set(resource.counties)
    fallback_city = resource.city
    fallback_zipcode = resource.zipcode
    index = ResourceIndex(all_resources)
    candidates = [
        other
        for other in all_resources
        if other.resource_id != resource.resource_id
        and other.city
        and other.zipcode
        and other.counties
        and other.city != fallback_city
        and other.zipcode != fallback_zipcode
        and not (set(other.counties) & fallback_counties)
    ]
    rng.shuffle(candidates)
    for other in candidates:
        county = rng.choice(other.counties)
        if not wrong_location_variants_empty(fallback_need, other, county, index):
            continue
        style = rng.choice(("zipcode", "city", "county"))
        if style == "zipcode":
            return {"zipcodes": [other.zipcode]}
        if style == "city":
            return {"cities": [other.city]}
        return {"counties": [county]}
    return {}


def wrong_location_variants_empty(fallback_need: dict, other: Resource, county: str, index: ResourceIndex) -> bool:
    variants = (
        {"zipcodes": [other.zipcode]},
        {"cities": [other.city]},
        {"counties": [county]},
        {"cities": [other.city], "counties": [county], "zipcodes": [other.zipcode]},
    )
    for location in variants:
        preferred = copy_need_search_fields(fallback_need)
        preferred["location"] = location
        if search_resources(index, request_from_tool_args(expected_tool_args_for_need(preferred)), limit=1):
            return False
    return True


def sample_wrong_schedule(resource: Resource, all_resources: list[Resource], rng: random.Random) -> dict:
    fallback_windows = {(window.day, window.start_minute, window.end_minute) for window in resource.schedule_windows}
    candidates = [
        window
        for other in all_resources
        if other.resource_id != resource.resource_id
        for window in other.schedule_windows
        if (window.day, window.start_minute, window.end_minute) not in fallback_windows
    ]
    if not candidates:
        return {}
    return schedule_window_requirement(rng.choice(candidates), rng)


def sample_wrong_intake(resource: Resource, all_resources: list[Resource], rng: random.Random) -> list[str]:
    fallback_methods = {method for method in resource.intake_methods if method != "empty"}
    candidates = sorted(
        {
            method
            for other in all_resources
            if other.resource_id != resource.resource_id
            for method in other.intake_methods
            if method != "empty" and method not in fallback_methods
        }
    )
    if not candidates:
        return []
    return [rng.choice(candidates)]


def sample_available_documents(resource: Resource, rng: random.Random) -> list[str]:
    return dedupe(concrete_requirements(resource.document_requirements))


def sample_eligibility(resource: Resource, rng: random.Random) -> list[str]:
    return dedupe(concrete_requirements(resource.eligibility_tags))


def concrete_requirements(values: tuple[str, ...]) -> list[str]:
    return [value for value in values if value not in IGNORED_REQUIREMENTS]


def dedupe(values: list[str]) -> list[str]:
    result = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result
