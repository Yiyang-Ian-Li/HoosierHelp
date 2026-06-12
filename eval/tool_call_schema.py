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
        time = _clean(schedule.get("time"))
        if not (start_time and end_time):
            start_time, end_time = _time_range(schedule.get("time"))
        if start_time and end_time and start_time == end_time:
            result["time"] = start_time
        elif start_time and end_time:
            result["start_time"] = start_time
            result["end_time"] = end_time
        elif time:
            parsed = _single_time(time)
            if parsed:
                result["time"] = parsed
    return result


def normalize_tool_calls(calls: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [normalize_tool_args(call) for call in (calls or []) if isinstance(call, dict)]


def score_tool_calls(predicted: list[dict[str, Any]] | None, expected: list[dict[str, Any]]) -> dict[str, Any]:
    expected_norm = normalize_tool_calls(expected)
    predicted_norm = normalize_tool_calls(predicted)
    matched_predicted: set[int] = set()
    per_expected = []
    for expected_call in expected_norm:
        best_index = None
        best_score = None
        for idx, predicted_call in enumerate(predicted_norm):
            if idx in matched_predicted:
                continue
            score = tool_arg_scores(predicted_call, expected_call)
            if best_score is None or int(score["all_match"]) > int(best_score["all_match"]) or field_match_count(score) > field_match_count(best_score):
                best_index = idx
                best_score = score
        if best_index is not None:
            matched_predicted.add(best_index)
            per_expected.append(best_score)
        else:
            per_expected.append(tool_arg_scores(None, expected_call))

    keys = ("service_match", "location_match", "schedule_match", "intake_match", "documents_match", "eligibility_match")
    return {
        "valid_tool_call": bool(predicted_norm),
        "expected_call_count": len(expected_norm),
        "predicted_call_count": len(predicted_norm),
        "tool_call_count_match": len(predicted_norm) == len(expected_norm),
        "per_expected": per_expected,
        **{key: all(bool(item.get(key)) for item in per_expected) if per_expected else False for key in keys},
        "all_match": bool(per_expected) and len(predicted_norm) == len(expected_norm) and all(bool(item.get("all_match")) for item in per_expected),
    }


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


def field_match_count(score: dict[str, Any]) -> int:
    return sum(
        bool(score.get(key))
        for key in ("service_match", "location_match", "schedule_match", "intake_match", "documents_match", "eligibility_match")
    )


def parse_selected_resource_ids(text: str) -> list[str]:
    ids = []
    for match in re.finditer(r"\bin211-[a-z0-9-]+\b", text or "", flags=re.IGNORECASE):
        value = match.group(0)
        if value not in ids:
            ids.append(value)
    return ids


def score_resource_selection(predicted_ids: list[str], expected_ids: list[str]) -> dict[str, Any]:
    predicted = list(dict.fromkeys(predicted_ids or []))
    expected = list(dict.fromkeys(expected_ids or []))
    predicted_set = set(predicted)
    expected_set = set(expected)
    correct = predicted_set & expected_set
    return {
        "expected_resource_count": len(expected),
        "predicted_resource_count": len(predicted),
        "correct_resource_count": len(correct),
        "resource_precision": len(correct) / len(predicted) if predicted else 0.0,
        "resource_recall": len(correct) / len(expected) if expected else 0.0,
        "resource_exact_match": predicted_set == expected_set if expected else not predicted,
    }


def score_resource_selection_by_need(predicted_ids: list[str], acceptable_by_need: list[list[str]]) -> dict[str, Any]:
    predicted = list(dict.fromkeys(predicted_ids or []))
    acceptable_sets = [set(ids) for ids in acceptable_by_need if ids]
    all_acceptable = set().union(*acceptable_sets) if acceptable_sets else set()
    covered = [bool(set(predicted) & acceptable) for acceptable in acceptable_sets]
    invalid = [resource_id for resource_id in predicted if resource_id not in all_acceptable]
    return {
        "expected_resource_count": len(acceptable_sets),
        "predicted_resource_count": len(predicted),
        "correct_resource_count": sum(covered),
        "resource_precision": (len([item for item in predicted if item in all_acceptable]) / len(predicted)) if predicted else 0.0,
        "resource_recall": (sum(covered) / len(acceptable_sets)) if acceptable_sets else 0.0,
        "resource_exact_match": bool(acceptable_sets) and all(covered) and not invalid,
        "acceptable_resource_ids_by_need": [sorted(ids) for ids in acceptable_sets],
        "invalid_resource_ids": invalid,
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


def _single_time(value: Any) -> str:
    text = _clean(value).lower()
    match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?", text)
    if not match:
        return ""
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    if hour < 7:
        hour += 12
    if hour > 23 or minute > 59:
        return ""
    return f"{hour:02d}:{minute:02d}"


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
