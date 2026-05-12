from __future__ import annotations

import csv
from pathlib import Path

from .indiana211_models import Resource, ResourceIndex, SearchRequest, SearchResult
from .indiana211_schedule import (
    DAY_VALUES,
    format_minutes,
    is_24_hour_window,
    parse_24_hour_time,
    schedule_windows_from_json,
)
from .indiana211_tags import (
    DOCUMENT_REQUIREMENT_MEANINGS,
    INTAKE_METHOD_MEANINGS,
)


DEFAULT_RESOURCE_INDEX = Path("data/benchmark/filtered_resources_tagged.csv")
DEFAULT_FILTERED_RESOURCE_CSV = Path("data/benchmark/filtered_resources_tagged.csv")

TOOL_DESCRIPTION = (
    "Search filtered Indiana 211 benchmark resources. Every non-empty field is a "
    "hard AND filter; values within one field are OR alternatives. Use only fields "
    "that are present in this schema and that the user explicitly needs satisfied."
)

DIFFICULTY_TOOL_FIELDS = {
    "easy": {
        "counties",
        "cities",
        "zipcodes",
        "service_categories",
        "intake_methods",
        "limit",
    },
    "medium": {
        "counties",
        "cities",
        "zipcodes",
        "service_categories",
        "available_days",
        "available_time_windows",
        "requires_24_hours",
        "intake_methods",
        "limit",
    },
    "hard": {
        "counties",
        "cities",
        "zipcodes",
        "service_categories",
        "available_days",
        "available_time_windows",
        "requires_24_hours",
        "intake_methods",
        "documents_available",
        "limit",
    },
}

IGNORED_REQUEST_VALUES = {"intake_methods": {"empty"}}
NO_DOCUMENT_VALUES = {"empty", "none", "varies"}

def load_resource_index(path: Path | str = DEFAULT_RESOURCE_INDEX) -> ResourceIndex:
    return load_filtered_resources(path)


def load_filtered_resources(path: Path | str = DEFAULT_FILTERED_RESOURCE_CSV) -> ResourceIndex:
    resources = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            resources.append(_resource_from_filtered_row(row))
    return ResourceIndex(resources)


def search_resources_tool_schema(index: ResourceIndex, difficulty: str | None = None) -> dict:
    allowed_fields = DIFFICULTY_TOOL_FIELDS.get(difficulty or "")
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
        "available_days": _enum_array_schema(
            list(DAY_VALUES),
            "Use when the user only needs availability on one or more days, without a specific time. Use mon/tue/wed/thu/fri/sat/sun.",
        ),
                "available_time_windows": {
                    "type": "array",
                    "description": "Use when the user needs availability on a specific day at a specific time. In this benchmark, provide day and start only.",
                    "items": {
                        "type": "object",
                        "properties": {
                    "day": {
                        "type": "string",
                        "enum": list(DAY_VALUES),
                        "description": "Required day: mon/tue/wed/thu/fri/sat/sun.",
                    },
                            "start": {
                                "type": "string",
                                "description": "Required start time in HH:MM 24-hour format.",
                            },
                        },
                        "required": ["day", "start"],
                        "additionalProperties": False,
            },
        },
        "requires_24_hours": {
            "type": "boolean",
            "description": "Use true only when the user requires 24-hour availability.",
        },
        "intake_methods": _enum_array_schema(
            ["call", "walk_in", "online", "appointment", "email", "text", "mail"],
            "Required intake/contact method.\n"
            + _value_meanings(["call", "walk_in", "online", "appointment", "email", "text", "mail"], INTAKE_METHOD_MEANINGS),
        ),
        "documents_available": _enum_array_schema(
            [
                "none",
                "photo_id",
                "proof_of_income",
                "proof_of_address",
                "lease",
                "insurance_card",
                "social_security",
                "birth_certificate",
                "utility_bill",
            ],
            "Documents the user can provide. Use ['none'] when the user has no documents available.\n"
            + _value_meanings(
                [
                    "none",
                    "photo_id",
                    "proof_of_income",
                    "proof_of_address",
                    "lease",
                    "insurance_card",
                    "social_security",
                    "birth_certificate",
                    "utility_bill",
                ],
                DOCUMENT_REQUIREMENT_MEANINGS,
            ),
        ),
        "limit": {
            "type": "integer",
            "minimum": 1,
            "maximum": 20,
            "description": "Maximum resources to return, from 1 to 20. Choose based on how many options are useful.",
        },
    }
    if allowed_fields is not None:
        properties = {key: value for key, value in properties.items() if key in allowed_fields}
    return {
        "type": "function",
        "name": "search_resources",
        "description": TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": ["limit"],
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
        available_days=_day_tuple(args.get("available_days")),
        available_time_windows=_time_window_tuple(args.get("available_time_windows")),
        requires_24_hours=bool(args.get("requires_24_hours")),
        intake_methods=_string_tuple(args.get("intake_methods")),
        documents_available=_string_tuple(args.get("documents_available")),
        limit=requested_limit,
    )


def search_resources_tool_result(results: list[SearchResult]) -> dict:
    return {
        "resources": [
            {
                "resource_id": result.resource.resource_id,
                "service_name": result.resource.service_name,
                "agency_name": result.resource.agency_name,
                "service_area": result.resource.service_area,
                "city": result.resource.city,
                "zipcode": result.resource.zipcode,
                "address": result.resource.address_1,
                "phone": result.resource.phone,
                "website": result.resource.website,
                "service_categories": result.resource.service_categories,
                "schedule_status": result.resource.schedule_status,
                "schedule_windows": [
                    {
                        "day": window.day,
                        "start": format_minutes(window.start_minute),
                        "end": format_minutes(window.end_minute),
                    }
                    for window in result.resource.schedule_windows
                ],
                "intake_methods": result.resource.intake_methods,
                "document_requirements": result.resource.document_requirements,
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
    intake_check = _match_derived_tag_field("intake_methods", request.intake_methods, resource.intake_methods)
    if intake_check is None:
        return None
    if intake_check:
        matched.append("intake_methods")
        score += 1.0
    document_check = _match_documents_available(request.documents_available, resource.document_requirements)
    if document_check is None:
        return None
    if document_check:
        matched.append("documents_available")
        score += 1.0
    return matched, score


def _match_derived_tag_field(name: str, requested: tuple[str, ...], available: tuple[str, ...]) -> bool | None:
    requested = _remove_ignored_requested_values(name, requested)
    if not requested:
        return False
    if not available:
        return None
    if _exact_any(requested, available):
        return True
    return None


def _match_schedule(resource: Resource, request: SearchRequest) -> bool | None:
    has_constraint = bool(
        request.available_days
        or request.available_time_windows
        or request.requires_24_hours
    )
    if not has_constraint:
        return False
    if resource.schedule_status != "structured":
        return None
    windows = resource.schedule_windows
    if request.requires_24_hours:
        windows = tuple(window for window in windows if is_24_hour_window(window))
    if request.available_days:
        requested_days = set(request.available_days)
        windows = tuple(window for window in windows if window.day in requested_days)
    for requested_window in request.available_time_windows:
        if not any(_window_satisfies_request(window, requested_window) for window in windows):
            return None
    return True if windows else None


def _match_documents_available(
    requested: tuple[str, ...],
    requirements: tuple[str, ...],
) -> bool | None:
    if not requested:
        return False
    requested_set = {_norm(value) for value in requested}
    required = concrete_document_requirements(requirements)
    if "none" in requested_set and len(requested_set) == 1:
        return True if not required else None
    requested_set.discard("none")
    return True if required.issubset(requested_set) else None


def concrete_document_requirements(values: tuple[str, ...]) -> set[str]:
    return {_norm(value) for value in values if _norm(value) not in NO_DOCUMENT_VALUES}


def _window_satisfies_request(window, requested: dict) -> bool:
    if window.day != requested.get("day"):
        return False
    start = parse_24_hour_time(str(requested.get("start", "")))
    if start is None:
        return False
    end = parse_24_hour_time(str(requested.get("end", "")))
    if is_24_hour_window(window):
        return True
    if end is None:
        return window.start_minute <= start < window.end_minute
    return window.start_minute <= start and window.end_minute >= end


def _remove_ignored_requested_values(name: str, requested: tuple[str, ...]) -> tuple[str, ...]:
    ignored = IGNORED_REQUEST_VALUES.get(name, set())
    return tuple(value for value in requested if _norm(value) not in ignored)


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


def _value_meanings(values, meanings: dict[str, str]) -> str:
    return "Candidate meanings: " + "; ".join(
        f"{value} = {meanings[value]}" for value in values
    )


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


def _day_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    days = []
    for item in value:
        if not isinstance(item, str):
            continue
        day = _norm(item)[:3]
        if day in DAY_VALUES and day not in days:
            days.append(day)
    return tuple(days)


def _time_window_tuple(value: object) -> tuple[dict, ...]:
    if not isinstance(value, list):
        return ()
    windows = []
    for item in value:
        if not isinstance(item, dict):
            continue
        day = _norm(str(item.get("day", "")))[:3]
        start = _clean(item.get("start", ""))
        end = _clean(item.get("end", ""))
        if day not in DAY_VALUES or parse_24_hour_time(start) is None:
            continue
        window = {"day": day, "start": start}
        if end and parse_24_hour_time(end) is not None:
            window["end"] = end
        if window not in windows:
            windows.append(window)
    return tuple(windows)


def _split_pipe(value: str) -> list[str]:
    return [part.strip() for part in (value or "").split("|") if part.strip()]


def _clean(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
