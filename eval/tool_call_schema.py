from __future__ import annotations

import re
from typing import Any


USER_BEHAVIORS = (
    "normal",
    "rambling",
    "impatience",
    "self_contradictory",
    "unsupported_request",
)

TOOL_NAME = "search_resources"
DAY_ALIASES = {
    "monday": "mon",
    "tuesday": "tue",
    "wednesday": "wed",
    "thursday": "thu",
    "friday": "fri",
    "saturday": "sat",
    "sunday": "sun",
}


def normalize_tool_args(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "service_categories": _string_list(args.get("service_categories")),
        "schedule": normalize_schedule(args.get("schedule") or {}),
        "counties": _location_list(args.get("counties")),
        "cities": _location_list(args.get("cities")),
        "zipcodes": _string_list(args.get("zipcodes")),
        "intake_methods": _string_list(args.get("intake_methods")),
        "available_documents": _document_list(args.get("available_documents")),
        "eligibility": _string_list(args.get("eligibility")),
    }


def normalize_schedule(schedule: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(schedule, dict):
        return {}
    if schedule.get("requires_24_hours"):
        return {"requires_24_hours": True}
    day = _clean(schedule.get("day")).lower()
    day = DAY_ALIASES.get(day, day)
    result: dict[str, Any] = {}
    if day:
        result["day"] = day
        start_time = _clean(schedule.get("start_time"))
        end_time = _clean(schedule.get("end_time"))
        if not (start_time and end_time):
            start_time, end_time = _time_range(schedule.get("time"))
        if start_time and end_time:
            result["start_time"] = start_time
            result["end_time"] = end_time
    return result


def tool_arg_scores(predicted: dict[str, Any] | None, expected: dict[str, Any]) -> dict[str, bool]:
    expected = normalize_tool_args(expected)
    if predicted is None:
        return {
            "valid_tool_call": False,
            "service_match": False,
            "location_match": False,
            "schedule_match": False,
            "intake_match": False,
            "documents_match": False,
            "eligibility_match": False,
            "all_match": False,
        }
    predicted = normalize_tool_args(predicted)
    service = set(predicted["service_categories"]) == set(expected["service_categories"])
    location = all(set(predicted[key]) == set(expected[key]) for key in ("counties", "cities", "zipcodes"))
    schedule = predicted["schedule"] == expected["schedule"]
    intake = set(predicted["intake_methods"]) == set(expected["intake_methods"])
    documents = set(predicted["available_documents"]) == set(expected["available_documents"])
    eligibility = set(predicted["eligibility"]) == set(expected["eligibility"])
    return {
        "valid_tool_call": True,
        "service_match": service,
        "location_match": location,
        "schedule_match": schedule,
        "intake_match": intake,
        "documents_match": documents,
        "eligibility_match": eligibility,
        "all_match": service and location and schedule and intake and documents and eligibility,
    }


def _time_range(value: Any) -> tuple[str, str]:
    text = _clean(value).lower()
    match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(?:-|to)\s*(\d{1,2})(?::(\d{2}))?", text)
    if not match:
        return "", ""
    start_hour = int(match.group(1))
    start_minute = int(match.group(2) or 0)
    end_hour = int(match.group(3))
    end_minute = int(match.group(4) or 0)
    if start_hour < 7 and end_hour <= 12:
        start_hour += 12
        end_hour += 12
    return f"{start_hour:02d}:{start_minute:02d}", f"{end_hour:02d}:{end_minute:02d}"


def _string_list(value: Any) -> list[str]:
    if value is None:
        values = []
    elif isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = [item for item in value if isinstance(item, str)]
    else:
        values = []
    result = []
    for item in values:
        clean = _clean(item)
        if clean and clean not in result:
            result.append(clean)
    return result


def _upper_list(value: Any) -> list[str]:
    return [item.upper() for item in _string_list(value)]


def _location_list(value: Any) -> list[str]:
    normalized = []
    for item in _upper_list(value):
        clean = re.sub(r"[^A-Z0-9]+", " ", item).strip()
        clean = re.sub(r"\s+", " ", clean)
        if clean and clean not in normalized:
            normalized.append(clean)
    return normalized


def _document_list(value: Any) -> list[str]:
    normalized = []
    for item in _string_list(value):
        clean = item.lower()
        if clean in {"empty", "none", "varies"}:
            continue
        if clean and clean not in normalized:
            normalized.append(clean)
    return normalized


def _clean(value: Any) -> str:
    return str(value or "").strip()
