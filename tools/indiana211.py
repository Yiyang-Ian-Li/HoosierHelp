from __future__ import annotations

import csv
from pathlib import Path

from .indiana211_models import Resource, ResourceIndex, SearchRequest, SearchResult
from .indiana211_schedule import (
    DAY_VALUES,
    is_24_hour_window,
    schedule_windows_from_json,
)


DEFAULT_RESOURCE_INDEX = Path("data/benchmark/filtered_resources_tagged.csv")
DEFAULT_FILTERED_RESOURCE_CSV = Path("data/benchmark/filtered_resources_tagged.csv")

TOOL_DESCRIPTION = (
    "Search filtered Indiana 211 benchmark resources. Every non-empty field is a "
    "hard AND filter; values within one field are OR alternatives. Use only fields "
    "that are present in this schema and that the user explicitly needs satisfied."
)

NO_DOCUMENT_VALUES = {"empty", "none", "varies"}
DAY_PART_RANGES = {
    "morning": (5 * 60, 12 * 60),
    "afternoon": (12 * 60, 17 * 60),
}

def load_resource_index(path: Path | str = DEFAULT_RESOURCE_INDEX) -> ResourceIndex:
    return load_filtered_resources(path)


def load_filtered_resources(path: Path | str = DEFAULT_FILTERED_RESOURCE_CSV) -> ResourceIndex:
    resources = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            resources.append(_resource_from_filtered_row(row))
    return ResourceIndex(resources)


def search_resources_tool_schema(index: ResourceIndex) -> dict:
    properties = {
        "counties": _array_schema(
            "User county names in uppercase, such as MARION or ALLEN."
        ),
        "cities": _array_schema(
            "User city names."
        ),
        "zipcodes": _array_schema(
            "User ZIP codes."
        ),
        "service_categories": _enum_array_schema(
            index.service_categories,
            "Service need categories. Choose the closest match to the user's main need."
        ),
        "schedule": {
            "type": "object",
            "description": (
                "Schedule requirement for this service need. Use either day+time, "
                "or requires_24_hours=true. Use time='any' for day-only needs."
            ),
            "properties": {
                "day": {
                    "type": "string",
                    "enum": list(DAY_VALUES),
                    "description": "Required day: mon/tue/wed/thu/fri/sat/sun.",
                },
                "time": {
                    "type": "string",
                    "enum": ["any", "morning", "afternoon"],
                    "description": "Use any, morning, or afternoon.",
                },
                "requires_24_hours": {
                    "type": "boolean",
                    "description": "Use true only when the user requires 24-hour availability.",
                },
            },
            "additionalProperties": False,
        },
        "limit": {
            "type": "integer",
            "minimum": 1,
            "maximum": 20,
            "description": "Maximum resources to return, from 1 to 20. Choose based on how many options are useful.",
        },
    }
    return {
        "type": "function",
        "name": "search_resources",
        "description": TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": ["service_categories", "schedule", "limit"],
            "additionalProperties": False,
        },
    }


def execute_search_resources(index: ResourceIndex, args: dict, limit: int | None = None) -> dict:
    request = request_from_tool_args(args, limit=limit)
    return search_resources_tool_result(search_resources(index, request))


def search_resources(index: ResourceIndex, request: SearchRequest) -> list[SearchResult]:
    results = []
    for resource in index.resources:
        match = _match_resource(resource, request)
        if match is None:
            continue
        matched, score = match
        results.append(
            SearchResult(
                resource=resource,
                score=score,
                matched_filters=tuple(matched),
            )
        )

    results.sort(
        key=lambda result: (
            -result.score,
            result.resource.service_name.lower(),
            result.resource.agency_name.lower(),
            result.resource.resource_id,
        )
    )
    return results[: request.limit]


def request_from_tool_args(args: dict, limit: int | None = None) -> SearchRequest:
    requested_limit = _bounded_limit(args.get("limit"), fallback=limit)
    return SearchRequest(
        counties=_string_tuple(args.get("counties")),
        cities=_string_tuple(args.get("cities")),
        zipcodes=_string_tuple(args.get("zipcodes")),
        service_categories=_string_tuple(args.get("service_categories")),
        schedule=_schedule_object(args.get("schedule")),
        limit=requested_limit,
    )


def search_resources_tool_result(results: list[SearchResult]) -> dict:
    return {
        "resources": [
            {
                "resource_id": result.resource.resource_id,
                "resource_name": result.resource.resource_name,
                "agency_name": result.resource.agency_name,
                "city": result.resource.city,
                "zipcode": result.resource.zipcode,
                "service_categories": result.resource.service_categories,
                "intake_methods": result.resource.intake_methods,
                "document_requirements": tuple(sorted(concrete_document_requirements(result.resource.document_requirements))),
            }
            for result in results
        ]
    }


def _bounded_limit(value: object, fallback: int | None = None) -> int:
    if fallback is None:
        fallback = 10
    try:
        limit = int(value)
    except (TypeError, ValueError):
        limit = int(fallback)
    return max(1, min(20, limit))


def _match_resource(resource: Resource, request: SearchRequest) -> tuple[list[str], float] | None:
    matched = []
    score = 0.0
    if request.counties:
        if _exact_any(request.counties, resource.service_area):
            matched.append("counties")
            score += 2.0
        else:
            return None
    if request.cities:
        if _exact_any(request.cities, (resource.city,)):
            matched.append("cities")
            score += 1.0
        else:
            return None
    if request.zipcodes:
        if _exact_any(request.zipcodes, (resource.zipcode,)):
            matched.append("zipcodes")
            score += 1.0
        else:
            return None
    if request.service_categories:
        if not _exact_any(request.service_categories, resource.service_categories):
            return None
        matched.append("service_categories")
        score += 1.0
    schedule_check = _match_schedule(resource, request)
    if schedule_check is None:
        return None
    if schedule_check:
        matched.append("schedule")
        score += 1.0
    return matched, score


def _match_schedule(resource: Resource, request: SearchRequest) -> bool | None:
    schedule = request.schedule or {}
    if not schedule:
        return False
    if resource.schedule_status != "structured":
        return None
    windows = resource.schedule_windows
    if schedule.get("requires_24_hours"):
        return True if any(is_24_hour_window(window) for window in windows) else None
    if schedule.get("day"):
        return True if any(_window_satisfies_schedule(window, schedule) for window in windows) else None
    return False


def concrete_document_requirements(values: tuple[str, ...]) -> set[str]:
    return {_norm(value) for value in values if _norm(value) not in NO_DOCUMENT_VALUES}


def _window_satisfies_schedule(window, schedule: dict) -> bool:
    if window.day != schedule.get("day"):
        return False
    time = schedule.get("time", "any")
    if time == "any":
        return True
    if is_24_hour_window(window):
        return True
    start, end = DAY_PART_RANGES[time]
    return window.start_minute < end and window.end_minute > start


def _resource_from_filtered_row(row: dict) -> Resource:
    return Resource(
        resource_id=_clean(row.get("resource_id", "")),
        service_name=_clean(row.get("service_name", "")),
        agency_name=_clean(row.get("agency_name", "")),
        site_name=_clean(row.get("site_name", "")),
        service_categories=tuple(_split_pipe(row.get("service_categories", ""))),
        service_area=tuple(_split_pipe(row.get("service_area", ""))),
        city=_clean(row.get("city", "")),
        state=_clean(row.get("state", "")),
        zipcode=_clean(row.get("zipcode", "")),
        address_1=_clean(row.get("address_1", "")),
        phone=_clean(row.get("phone", "")),
        website=_clean(row.get("website", "")),
        schedule_status=_clean(row.get("schedule_status", "")),
        schedule_windows=schedule_windows_from_json(row),
        intake_methods=tuple(_split_pipe(row.get("intake_methods", ""))),
        document_requirements=tuple(_split_pipe(row.get("document_requirements", ""))),
    )


def _array_schema(description: str) -> dict:
    return {"type": "array", "items": {"type": "string"}, "description": description}


def _enum_array_schema(values: list[str], description: str) -> dict:
    return {
        "type": "array",
        "items": {"type": "string", "enum": values},
        "description": description,
    }


def _exact_any(requested: tuple[str, ...], available: tuple[str, ...]) -> bool:
    req = {_norm(item) for item in requested}
    av = {_norm(item) for item in available}
    return bool(req & av)


def _norm(value: str) -> str:
    return value.lower().strip().replace(".", "").replace("-", "_").replace(" ", "_")


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())


def _schedule_object(value: object) -> dict:
    if not isinstance(value, dict):
        return {}
    if value.get("requires_24_hours") is True:
        return {"requires_24_hours": True}
    day = _norm(str(value.get("day", "")))[:3]
    if day not in DAY_VALUES:
        return {}
    time = _norm(str(value.get("time", "any")))
    if time not in {"any", "morning", "afternoon"}:
        time = "any"
    return {"day": day, "time": time}


def _split_pipe(value: str) -> list[str]:
    return [part.strip() for part in (value or "").split("|") if part.strip()]


def _clean(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
