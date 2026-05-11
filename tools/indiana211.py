from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from .curated_categories import service_categories_for_raw_subcategories
from .indiana211_models import Resource, ResourceIndex, SearchRequest, SearchResult
from .indiana211_schedule import (
    DAY_VALUES,
    format_minutes,
    is_24_hour_window,
    parse_24_hour_time,
    schedule_status,
    schedule_windows,
    schedule_windows_from_json,
)
from .indiana211_tags import (
    DOCUMENT_REQUIREMENT_MEANINGS,
    ELIGIBILITY_TAG_MEANINGS,
    FEE_OPTION_MEANINGS,
    INTAKE_METHOD_MEANINGS,
    document_requirements,
    eligibility_tags,
    fee_options,
    intake_methods,
)


DEFAULT_RESOURCE_INDEX = Path("data/indiana211/indiana211_resources_raw_all_counties.json")
DEFAULT_FULL_INDIANA_CSV = Path("data/indiana211/indiana211_resources_deduped.csv")

TOOL_DESCRIPTION = (
    "Search Indiana 211 resources. Non-empty fields are hard AND filters; values "
    "within one field are OR alternatives. County filters match resources serving "
    "that county, including statewide resources. Cities and ZIP codes are ranking "
    "signals, not hard filters. Use optional tag filters only for explicit user requirements."
)

DERIVED_TAG_FIELDS = {
    "eligibility_tags",
    "intake_methods",
    "document_requirements",
    "fee_options",
}

IGNORED_REQUEST_VALUES = {
    "eligibility_tags": {"empty", "open"},
    "intake_methods": {"empty"},
    "document_requirements": {"empty", "varies"},
    "fee_options": {"empty", "varies"},
}

def load_resource_index(path: Path | str = DEFAULT_RESOURCE_INDEX) -> ResourceIndex:
    resources = []
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return ResourceIndex(resources)
    if text.startswith("["):
        return ResourceIndex(_resources_from_raw_rows(json.loads(text)))
    for line in text.splitlines():
        if line.strip():
            resources.append(_resource_from_json(json.loads(line)))
    return ResourceIndex(resources)


def load_indiana_csv(path: Path | str = DEFAULT_FULL_INDIANA_CSV) -> ResourceIndex:
    resources = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            service_name = _clean(row.get("service_name", ""))
            slug = re.sub(r"[^a-z0-9]+", "-", service_name.lower()).strip("-")
            resources.append(
                Resource(
                    resource_id=f"in211-{row['agency_id']}-{row['site_id']}-{slug}",
                    service_name=service_name,
                    agency_name=_clean(row.get("agency_name", "")),
                    site_name=_clean(row.get("site_name", "")),
                    taxonomy_categories=tuple(_split(row.get("taxonomy_categories", ""))),
                    subcategories=tuple(_split(row.get("subcategories", ""))),
                    service_categories=service_categories_for_raw_subcategories(
                        _split(row.get("subcategories", ""))
                    ),
                    service_area=tuple(_split(row.get("counties_served", ""))),
                    city=_clean(row.get("city", "")),
                    state=_clean(row.get("state_province", "")),
                    zipcode=_clean(row.get("zipcode", "")),
                    address_1=_clean(row.get("address_1", "")),
                    phone=_clean(row.get("site_number", "")),
                    website=_clean(row.get("service_website", "")),
                    eligibility=_clean(row.get("site_eligibility", "")),
                    site_schedule=_clean(row.get("site_schedule", "")),
                    schedule_status=schedule_status(row.get("site_schedule", "")),
                    schedule_windows=schedule_windows(row.get("site_schedule", "")),
                    site_details=_clean(row.get("site_details", "")),
                    fee_structure=_clean(row.get("fee_structure", "")),
                    documents_required=_clean(row.get("documents_required", "")),
                    eligibility_tags=eligibility_tags(row.get("site_eligibility", "")),
                    intake_methods=intake_methods(row.get("site_details", "")),
                    document_requirements=document_requirements(row.get("documents_required", "")),
                    fee_options=fee_options(row.get("fee_structure", "")),
                )
            )
    return ResourceIndex(resources)


def search_resources_tool_schema(index: ResourceIndex) -> dict:
    return {
        "type": "function",
        "name": "search_resources",
        "description": TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "counties": _array_schema(
                    "User county names in uppercase, such as MARION or ALLEN. Matches resources serving that county, including statewide resources."
                ),
                "cities": _array_schema(
                    "User city names explicitly stated by the user. Ranking boost only."
                ),
                "zipcodes": _array_schema(
                    "User ZIP codes explicitly stated by the user. Ranking boost only."
                ),
                "service_categories": _enum_array_schema(
                    index.service_categories,
                    "Service need categories. Choose the closest match to the user's main need."
                ),
                "eligibility_tags": _enum_array_schema(
                    [
                        "open",
                        "resident",
                        "income",
                        "senior",
                        "children",
                        "disability",
                        "veteran",
                        "pregnant",
                        "homeless",
                    ],
                    "Eligibility requirements explicitly needed by the user. Do not add general user traits as filters unless required.\n"
                    + _value_meanings(
                        [
                            "open",
                            "resident",
                            "income",
                            "senior",
                            "children",
                            "disability",
                            "veteran",
                            "pregnant",
                            "homeless",
                        ],
                        ELIGIBILITY_TAG_MEANINGS,
                    ),
                ),
                "available_days": _enum_array_schema(
                    list(DAY_VALUES),
                    "Days the user explicitly needs availability. Use mon/tue/wed/thu/fri/sat/sun.",
                ),
                "available_at_or_after": {
                    "type": "string",
                    "description": "Earliest clock time the resource must be open until, formatted HH:MM in 24-hour time, such as 17:00. Use only for explicit time needs.",
                },
                "requires_weekend": {
                    "type": "boolean",
                    "description": "Use true only when the user requires Saturday or Sunday availability.",
                },
                "requires_24_hours": {
                    "type": "boolean",
                    "description": "Use true only when the user requires 24-hour availability.",
                },
                "allow_appointment_only": {
                    "type": "boolean",
                    "description": "Use true when appointment-only resources are acceptable for the user's schedule need.",
                },
                "intake_methods": _enum_array_schema(
                    ["call", "walk_in", "online", "appointment", "email", "text", "mail"],
                    "Required intake/contact method. Do not filter on methods the user merely says they can use.\n"
                    + _value_meanings(["call", "walk_in", "online", "appointment", "email", "text", "mail"], INTAKE_METHOD_MEANINGS),
                ),
                "document_requirements": _enum_array_schema(
                    [
                        "none",
                        "varies",
                        "photo_id",
                        "proof_of_income",
                        "proof_of_address",
                        "lease",
                        "insurance_card",
                        "social_security",
                        "birth_certificate",
                        "utility_bill",
                    ],
                    "Document constraints explicitly required by the user.\n"
                    + _value_meanings(
                        [
                            "none",
                            "varies",
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
                "fee_options": _enum_array_schema(
                    ["unknown", "free", "sliding_scale", "varies", "insurance", "payment_required"],
                    "Fee/payment requirements explicitly stated by the user.\n"
                    + _value_meanings(["unknown", "free", "sliding_scale", "varies", "insurance", "payment_required"], FEE_OPTION_MEANINGS),
                ),
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Maximum resources to return, from 1 to 20. Choose based on how many options are useful.",
                },
            },
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
        eligibility_tags=_string_tuple(args.get("eligibility_tags")),
        available_days=_day_tuple(args.get("available_days")),
        available_at_or_after=_clean(args.get("available_at_or_after", "")),
        requires_weekend=bool(args.get("requires_weekend")),
        requires_24_hours=bool(args.get("requires_24_hours")),
        allow_appointment_only=bool(args.get("allow_appointment_only")),
        intake_methods=_string_tuple(args.get("intake_methods")),
        document_requirements=_string_tuple(args.get("document_requirements")),
        fee_options=_string_tuple(args.get("fee_options")),
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
                "eligibility": result.resource.eligibility,
                "schedule": result.resource.site_schedule,
                "schedule_status": result.resource.schedule_status,
                "schedule_windows": [
                    {
                        "day": window.day,
                        "start": format_minutes(window.start_minute),
                        "end": format_minutes(window.end_minute),
                    }
                    for window in result.resource.schedule_windows
                ],
                "intake": result.resource.site_details,
                "fees": result.resource.fee_structure,
                "documents": result.resource.documents_required,
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
    derived_checks = [
        ("eligibility_tags", request.eligibility_tags, resource.eligibility_tags),
        ("intake_methods", request.intake_methods, resource.intake_methods),
        ("document_requirements", request.document_requirements, resource.document_requirements),
        ("fee_options", request.fee_options, resource.fee_options),
    ]
    matched = []
    score = 0.0
    if request.counties:
        if _exact_any(request.counties, resource.service_area):
            matched.append("counties")
            score += 2.0
        elif _exact_any(("STATEWIDE", "ALL"), resource.service_area):
            matched.append("statewide")
            score += 2.0
        else:
            return None
    if request.cities:
        if _exact_any(request.cities, (resource.city,)):
            matched.append("cities")
            score += 1.0
    if request.zipcodes:
        if _exact_any(request.zipcodes, (resource.zipcode,)):
            matched.append("zipcodes")
            score += 1.0
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
    for name, requested, available in derived_checks:
        check = _match_derived_tag_field(name, requested, available)
        if check is None:
            return None
        if check:
            matched.append(name)
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
        or request.available_at_or_after
        or request.requires_weekend
        or request.requires_24_hours
    )
    if not has_constraint:
        return False
    if resource.schedule_status == "appointment_only":
        return True if request.allow_appointment_only else None
    if resource.schedule_status != "structured":
        return None
    windows = resource.schedule_windows
    if request.requires_24_hours:
        windows = tuple(window for window in windows if is_24_hour_window(window))
    requested_days = set(request.available_days)
    if request.requires_weekend:
        requested_days.update(("sat", "sun"))
    if requested_days:
        windows = tuple(window for window in windows if window.day in requested_days)
    requested_time = parse_24_hour_time(request.available_at_or_after)
    if requested_time is not None:
        windows = tuple(
            window
            for window in windows
            if is_24_hour_window(window) or window.end_minute > requested_time
        )
    return True if windows else None


def _remove_ignored_requested_values(name: str, requested: tuple[str, ...]) -> tuple[str, ...]:
    ignored = IGNORED_REQUEST_VALUES.get(name, set())
    return tuple(value for value in requested if _norm(value) not in ignored)


def _resource_from_json(row: dict) -> Resource:
    location = row.get("location") or {}
    contact = row.get("contact") or {}
    return Resource(
        resource_id=str(row["resource_id"]),
        service_name=str(row.get("service_name", "")),
        agency_name=str(row.get("agency_name", "")),
        site_name=str(row.get("site_name", "")),
        taxonomy_categories=tuple(row.get("taxonomy_categories") or ()),
        subcategories=tuple(row.get("subcategories") or ()),
        service_categories=tuple(
            row.get("service_categories")
            or service_categories_for_raw_subcategories(row.get("subcategories") or ())
        ),
        service_area=tuple(row.get("service_area") or ()),
        city=str(location.get("city", "")),
        state=str(location.get("state", "")),
        zipcode=str(location.get("zipcode", "")),
        address_1=str(location.get("address_1", "")),
        phone=str(contact.get("phone", "")),
        website=str(contact.get("website", "")),
        eligibility=str(row.get("eligibility", "")),
        site_schedule=str(row.get("site_schedule", "")),
        schedule_status=str(row.get("schedule_status") or schedule_status(row.get("site_schedule", ""))),
        schedule_windows=schedule_windows_from_json(row),
        site_details=str(row.get("site_details", "")),
        fee_structure=str(row.get("fee_structure", "")),
        documents_required=str(row.get("documents_required", "")),
        eligibility_tags=tuple(row.get("eligibility_tags") or eligibility_tags(row.get("eligibility", ""))),
        intake_methods=tuple(row.get("intake_methods") or intake_methods(row.get("site_details", ""))),
        document_requirements=tuple(row.get("document_requirements") or document_requirements(row.get("documents_required", ""))),
        fee_options=tuple(row.get("fee_options") or fee_options(row.get("fee_structure", ""))),
    )


def _resources_from_raw_rows(rows: list[dict]) -> list[Resource]:
    grouped: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        key = (
            str(row.get("agency_id") or ""),
            str(row.get("site_id") or ""),
            _clean(row.get("service_name", "")),
        )
        if key not in grouped:
            grouped[key] = {**row, "_counties": [], "_categories": [], "_subcategories": []}
        group = grouped[key]
        _append_unique(group["_counties"], _clean(row.get("county", "")))
        _append_unique(group["_categories"], _clean(row.get("taxonomy_category", "")))
        _append_unique(group["_subcategories"], _clean(row.get("subcategory", "")))
    return [_resource_from_raw_group(row) for row in grouped.values()]


def _resource_from_raw_group(row: dict) -> Resource:
    service_name = _clean(row.get("service_name", ""))
    slug = "-".join(part for part in service_name.lower().split() if part)
    return Resource(
        resource_id=f"in211-{row.get('agency_id', '')}-{row.get('site_id', '')}-{slug}",
        service_name=service_name,
        agency_name=_clean(row.get("agency_name", "")),
        site_name=_clean(row.get("site_name", "")),
        taxonomy_categories=tuple(row.get("_categories") or ()),
        subcategories=tuple(row.get("_subcategories") or ()),
        service_categories=service_categories_for_raw_subcategories(row.get("_subcategories") or ()),
        service_area=tuple(row.get("_counties") or ()),
        city=_clean(row.get("city", "")),
        state=_clean(row.get("state_province", "")),
        zipcode=_clean(row.get("zipcode", "")),
        address_1=_clean(row.get("address_1", "")),
        phone=_clean(row.get("site_number", "")),
        website=_clean(row.get("service_website", "")),
        eligibility=_clean(row.get("site_eligibility", "")),
        site_schedule=_clean(row.get("site_schedule", "")),
        schedule_status=schedule_status(row.get("site_schedule", "")),
        schedule_windows=schedule_windows(row.get("site_schedule", "")),
        site_details=_clean(row.get("site_details", "")),
        fee_structure=_clean(row.get("fee_structure", "")),
        documents_required=_clean(row.get("documents_required", "")),
        eligibility_tags=eligibility_tags(row.get("site_eligibility", "")),
        intake_methods=intake_methods(row.get("site_details", "")),
        document_requirements=document_requirements(row.get("documents_required", "")),
        fee_options=fee_options(row.get("fee_structure", "")),
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


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def _split(value: str) -> list[str]:
    return [part.strip() for part in (value or "").split(";") if part.strip()]


def _clean(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
