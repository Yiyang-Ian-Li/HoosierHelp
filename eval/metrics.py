from __future__ import annotations

import re


RESOURCE_ID_RE = re.compile(r"\bin211-[a-z0-9-]+\b", re.IGNORECASE)


def extract_resource_ids(response: dict) -> list[str]:
    ids = []
    for call in response.get("tool_calls", ()):
        result = call.get("result") or {}
        for resource in result.get("resources", []):
            resource_id = resource.get("resource_id")
            if resource_id and resource_id not in ids:
                ids.append(resource_id)
    for item in response.get("input", []):
        if item_get(item, "type") != "function_call_output":
            continue
        result = parse_tool_output(item_get(item, "output") or "")
        for resource in result.get("resources", []):
            resource_id = resource.get("resource_id")
            if resource_id and resource_id not in ids:
                ids.append(resource_id)
    return ids


def extract_recommended_resource_ids(response: dict) -> list[str]:
    ids = []
    for resource_id in RESOURCE_ID_RE.findall(response.get("output_text", "")):
        normalized = resource_id.lower()
        if normalized not in ids:
            ids.append(normalized)
    return ids


def score_case(card: dict, ground_truth: dict, transcript: list[dict], final_response: dict) -> dict:
    retrieved_ids = extract_resource_ids(final_response)
    recommended_ids = extract_recommended_resource_ids(final_response)
    for resource_id in extract_recommended_resource_ids_from_transcript(transcript):
        if resource_id not in recommended_ids:
            recommended_ids.append(resource_id)
    primary = set(ground_truth.get("primary_gt_resource_ids", []))
    acceptable = set(ground_truth.get("acceptable_gt_resource_ids", []))
    text = transcript_text(transcript)
    asked_items = {
        "location_or_zip": any(term in text for term in ["zip", "county", "city", "where are you", "location"]),
        "urgency": any(term in text for term in ["urgent", "when", "deadline", "appointment", "tonight", "shutoff"]),
        "constraints": any(term in text for term in ["transport", "ride", "bus", "car", "language", "phone", "accessible"]),
        "household_or_eligibility": any(term in text for term in ["household", "children", "older", "pregnant", "veteran", "disability", "income"]),
    }
    return {
        "user_id": card["user_id"],
        "difficulty": card["difficulty"],
        "retrieved_resource_ids": retrieved_ids,
        "recommended_resource_ids": recommended_ids,
        "primary_hit": bool(primary & set(recommended_ids)),
        "acceptable_hit": bool(acceptable & set(recommended_ids)),
        "retrieval_primary_hit": bool(primary & set(retrieved_ids)),
        "retrieval_acceptable_hit": bool(acceptable & set(retrieved_ids)),
        "tool_call_count": count_function_calls(final_response),
        "turn_count": len([turn for turn in transcript if turn["role"] == "user"]),
        "asked_items": asked_items,
        "clarification_score": sum(1 for value in asked_items.values() if value),
    }


def count_function_calls(response: dict) -> int:
    calls = len(response.get("tool_calls", ()))
    calls = max(
        calls,
        sum(1 for item in response.get("input", []) if item_get(item, "type") == "function_call"),
    )
    return calls


def extract_recommended_resource_ids_from_transcript(transcript: list[dict]) -> list[str]:
    ids = []
    for turn in transcript:
        if turn.get("role") != "agent":
            continue
        for resource_id in RESOURCE_ID_RE.findall(str(turn.get("content", ""))):
            normalized = resource_id.lower()
            if normalized not in ids:
                ids.append(normalized)
    return ids


def item_get(item, key: str):
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def parse_tool_output(output: str) -> dict:
    import json

    try:
        data = json.loads(output or "{}")
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def aggregate(scores: list[dict]) -> dict:
    if not scores:
        return {}
    return {
        "cases": len(scores),
        "primary_hit_rate": mean(score["primary_hit"] for score in scores),
        "acceptable_hit_rate": mean(score["acceptable_hit"] for score in scores),
        "retrieval_primary_hit_rate": mean(score["retrieval_primary_hit"] for score in scores),
        "retrieval_acceptable_hit_rate": mean(score["retrieval_acceptable_hit"] for score in scores),
        "average_tool_calls": sum(score["tool_call_count"] for score in scores) / len(scores),
        "average_clarification_score": sum(score["clarification_score"] for score in scores) / len(scores),
        "by_difficulty": {
            difficulty: {
                "cases": len(group),
                "primary_hit_rate": mean(score["primary_hit"] for score in group),
                "acceptable_hit_rate": mean(score["acceptable_hit"] for score in group),
                "retrieval_primary_hit_rate": mean(score["retrieval_primary_hit"] for score in group),
                "retrieval_acceptable_hit_rate": mean(score["retrieval_acceptable_hit"] for score in group),
                "average_clarification_score": sum(score["clarification_score"] for score in group) / len(group),
            }
            for difficulty, group in groups_by_difficulty(scores).items()
        },
    }


def groups_by_difficulty(scores: list[dict]) -> dict[str, list[dict]]:
    groups = {}
    for score in scores:
        groups.setdefault(score["difficulty"], []).append(score)
    return groups


def mean(values) -> float:
    values = list(values)
    return sum(1 for value in values if value) / len(values)


def transcript_text(transcript: list[dict]) -> str:
    return "\n".join(str(turn.get("content", "")) for turn in transcript).lower()
