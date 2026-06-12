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
    "Search filtered Indiana 211 benchmark resources. Service, schedule, location, "
    "intake, documents, and eligibility are hard AND filters. Values inside a field "
    "are acceptable OR alternatives. Location is also OR across zipcodes, cities, "
    "and counties, so include every location the user says they can accept."
)

NO_DOCUMENT_VALUES = {"empty", "none", "varies"}
DEFAULT_RESULT_LIMIT = 10

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
                "Optional schedule requirement. Use day alone when the user only "
                "names a day, day with time when they say they are free at a "
                "specific time, day with start_time/end_time for an availability "
                "window, or requires_24_hours=true for 24-hour availability."
            ),
            "properties": {
                "day": {
                    "type": "string",
                    "enum": list(DAY_VALUES),
                    "description": "Required day: mon/tue/wed/thu/fri/sat/sun.",
                },
                "start_time": {
                    "type": "string",
                    "description": "Start of needed window in 24-hour HH:MM format, such as 09:00.",
                },
                "end_time": {
                    "type": "string",
                    "description": "End of needed window in 24-hour HH:MM format, such as 17:30.",
                },
                "time": {
                    "type": "string",
                    "description": "Specific time the user can go in 24-hour HH:MM format, such as 14:00.",
                },
                "requires_24_hours": {
                    "type": "boolean",
                    "description": "Use true only when the user requires 24-hour availability.",
                },
            },
            "additionalProperties": False,
        },
        "intake_methods": _enum_array_schema(
            index.intake_methods,
            "Optional intake methods the user can use, such as call, walk_in, online, appointment, email, text, or mail.",
        ),
        "available_documents": _enum_array_schema(
            index.document_requirements,
            "Optional documents the user can provide. These are capabilities, not required resource tags.",
        ),
        "eligibility": _enum_array_schema(
            index.eligibility_tags,
            "Optional user eligibility traits the user has, such as low_income, veteran, senior, youth, disability, pregnant, family, homeless, uninsured, or resident.",
        ),
    }
    return {
        "type": "function",
        "name": "search_resources",
        "description": TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": ["service_categories"],
            "additionalProperties": False,
        },
    }


def execute_search_resources(index: ResourceIndex, args: dict, limit: int | None = None) -> dict:
    request = request_from_tool_args(args, limit=limit)
    return search_resources_tool_result(search_resources(index, request, limit=limit))


def search_resources(index: ResourceIndex, request: SearchRequest, limit: int | None = None) -> list[SearchResult]:
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
            result.resource.resource_id,
        )
    )
    return results[:_bounded_limit(limit, DEFAULT_RESULT_LIMIT)]


def request_from_tool_args(args: dict, limit: int | None = None) -> SearchRequest:
    return SearchRequest(
        service_categories=_string_tuple(args.get("service_categories")),
        schedule=_schedule_object(args.get("schedule")),
        counties=_string_tuple(args.get("counties")),
        cities=_string_tuple(args.get("cities")),
        zipcodes=_string_tuple(args.get("zipcodes")),
        intake_methods=_string_tuple(args.get("intake_methods")),
        available_documents=_string_tuple(args.get("available_documents")),
        eligibility=_string_tuple(args.get("eligibility")),
    )


def search_resources_tool_result(results: list[SearchResult]) -> dict:
    return {
        "resources": [
            {
                "resource_id": result.resource.resource_id,
                "resource_name": result.resource.resource_name,
                "city": result.resource.city,
                "zipcode": result.resource.zipcode,
                "service_categories": result.resource.service_categories,
                "intake_methods": result.resource.intake_methods,
                "document_requirements": tuple(sorted(concrete_document_requirements(result.resource.document_requirements))),
                "eligibility": result.resource.eligibility_tags,
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
    if request.service_categories:
        if not _exact_any(request.service_categories, resource.service_categories):
            return None
        matched.append("service_categories")
        score += 3.0
    if request.schedule:
        schedule_check = _match_schedule(resource, request)
        if schedule_check is None:
            return None
        matched.append("schedule")
        score += 2.0
    if request.counties or request.cities or request.zipcodes:
        location_matches = []
        if request.counties and _exact_any(request.counties, resource.counties):
            location_matches.append("counties")
            score += 2.0
        if request.cities and _exact_any(request.cities, (resource.city,)):
            location_matches.append("cities")
            score += 1.0
        if request.zipcodes and _exact_any(request.zipcodes, (resource.zipcode,)):
            location_matches.append("zipcodes")
            score += 1.0
        if not location_matches:
            return None
        matched.extend(location_matches)
    if request.intake_methods:
        if _exact_any(request.intake_methods, resource.intake_methods):
            matched.append("intake_methods")
            score += 1.0
        else:
            return None
    if request.available_documents:
        if not _requirements_satisfied(resource.document_requirements, request.available_documents, NO_DOCUMENT_VALUES):
            return None
        matched.append("available_documents")
        score += 1.0
    if request.eligibility:
        if not _requirements_satisfied(resource.eligibility_tags, request.eligibility, {"none", "unknown"}):
            return None
        matched.append("eligibility")
        score += 1.0
    return matched, score


def _match_schedule(resource: Resource, request: SearchRequest) -> bool | None:
    schedule = request.schedule or {}
    if not schedule:
        return False
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
    if is_24_hour_window(window):
        return True
    start = schedule.get("start_minute")
    end = schedule.get("end_minute")
    time = schedule.get("time_minute")
    if time is not None:
        return window.start_minute <= time <= window.end_minute
    if start is None or end is None:
        return True
    return window.start_minute <= start and window.end_minute >= end


def _requirements_satisfied(required: tuple[str, ...], available: tuple[str, ...], ignored: set[str]) -> bool:
    required_set = {_norm(value) for value in required if _norm(value) not in ignored}
    if not required_set:
        return True
    available_set = {_norm(value) for value in available}
    return required_set <= available_set


def _resource_from_filtered_row(row: dict) -> Resource:
    return Resource(
        resource_id=_clean(row.get("resource_id", "")),
        service_name=_clean(row.get("service_name", "")),
        service_categories=tuple(_split_pipe(row.get("service_categories", ""))),
        counties=tuple(_split_pipe(row.get("counties", row.get("service_area", "")))),
        city=_clean(row.get("city", "")),
        zipcode=_clean(row.get("zipcode", "")),
        schedule_windows=schedule_windows_from_json(row),
        intake_methods=tuple(_split_pipe(row.get("intake_methods", ""))),
        document_requirements=tuple(_split_pipe(row.get("document_requirements", ""))),
        eligibility_tags=tuple(_split_pipe(row.get("eligibility", row.get("eligibility_tags", "")))),
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
    start = _parse_hhmm(value.get("start_time"))
    end = _parse_hhmm(value.get("end_time"))
    time = _parse_hhmm(value.get("time"))
    result = {"day": day}
    if start is not None and end is not None and start == end:
        result["time"] = str(value.get("start_time"))
        result["time_minute"] = start
    elif start is not None and end is not None and start < end:
        result["start_time"] = str(value.get("start_time"))
        result["end_time"] = str(value.get("end_time"))
        result["start_minute"] = start
        result["end_minute"] = end
    elif time is not None:
        result["time"] = str(value.get("time"))
        result["time_minute"] = time
    return result


def _parse_hhmm(value: object) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    parts = text.split(":")
    if len(parts) != 2:
        return None
    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except ValueError:
        return None
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return None
    return hour * 60 + minute


def _split_pipe(value: str) -> list[str]:
    return [part.strip() for part in (value or "").split("|") if part.strip()]


def _clean(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
