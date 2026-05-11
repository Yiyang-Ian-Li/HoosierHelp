from __future__ import annotations

import re


RESOURCE_ID_RE = re.compile(r"\bin211-[a-z0-9]+(?:-[a-z0-9]+)*\b", re.IGNORECASE)


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


def extract_recommended_resource_ids(response: dict, transcript: list[dict] | None = None) -> list[str]:
    if transcript:
        transcript_ids = extract_last_recommended_resource_ids_from_transcript(transcript)
        if transcript_ids:
            return transcript_ids
    ids = []
    result = response.get("structured_result") or {}
    for resource_id in result.get("recommended_resource_ids", []) or []:
        if not isinstance(resource_id, str):
            continue
        normalized = resource_id.strip().lower()
        if normalized not in ids:
            ids.append(normalized)
    return ids


def extract_last_recommended_resource_ids_from_transcript(transcript: list[dict]) -> list[str]:
    for turn in reversed(transcript):
        if turn.get("role") != "agent":
            continue
        ids = extract_resource_ids_from_text(str(turn.get("content", "")))
        if ids:
            return ids
    return []


def extract_resource_ids_from_text(text: str) -> list[str]:
    ids = []
    for match in RESOURCE_ID_RE.finditer(text):
        resource_id = match.group(0).lower()
        if resource_id not in ids:
            ids.append(resource_id)
    return ids


def score_case(
    card: dict,
    ground_truth: dict,
    transcript: list[dict],
    final_response: dict,
    user_satisfaction: dict | None = None,
) -> dict:
    retrieved_ids = extract_resource_ids(final_response)
    recommended_ids = extract_recommended_resource_ids(final_response, transcript)
    expected = set(ground_truth.get("ground_truth_resource_ids", []))
    satisfaction = user_satisfaction or {}
    diagnostics = recommendation_diagnostics(transcript, final_response, retrieved_ids)
    ground_truth_hit = bool(expected & set(recommended_ids))
    retrieval_ground_truth_hit = bool(expected & set(retrieved_ids))
    return {
        "user_id": card["user_id"],
        "retrieved_resource_ids": retrieved_ids,
        "recommended_resource_ids": recommended_ids,
        "ground_truth_hit": ground_truth_hit,
        "retrieval_ground_truth_hit": retrieval_ground_truth_hit,
        "tool_call_count": count_function_calls(final_response),
        "turn_count": len([turn for turn in transcript if turn["role"] == "user"]),
        "user_satisfaction": satisfaction,
        "satisfaction": numeric_or_none(satisfaction.get("satisfaction")),
        "got_relevant_help": bool_or_none(satisfaction.get("got_relevant_help")),
        "felt_understood": bool_or_none(satisfaction.get("felt_understood")),
        "actionability": numeric_or_none(satisfaction.get("actionability")),
        **diagnostics,
    }


def recommendation_diagnostics(
    transcript: list[dict],
    final_response: dict,
    retrieved_ids: list[str],
) -> dict:
    agent_turns = [str(turn.get("content", "")) for turn in transcript if turn.get("role") == "agent"]
    recommendation_turns = [text for text in agent_turns if extract_resource_ids_from_text(text)]
    retrieved = {resource_id.lower() for resource_id in retrieved_ids}
    recommended = extract_last_recommended_resource_ids_from_transcript(transcript)
    return {
        "recommendation_turn_count": len(recommendation_turns),
        "multiple_recommendation_turns": len(recommendation_turns) > 1,
        "recommended_ids_not_retrieved": [
            resource_id for resource_id in recommended if resource_id not in retrieved
        ],
    }


def count_function_calls(response: dict) -> int:
    calls = len(response.get("tool_calls", ()))
    calls = max(
        calls,
        sum(1 for item in response.get("input", []) if item_get(item, "type") == "function_call"),
    )
    return calls


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
        "ground_truth_hit_rate": mean(score.get("ground_truth_hit") for score in scores),
        "retrieval_ground_truth_hit_rate": mean(score["retrieval_ground_truth_hit"] for score in scores),
        "average_tool_calls": sum(score["tool_call_count"] for score in scores) / len(scores),
        "average_satisfaction": mean_numeric(score.get("satisfaction") for score in scores),
        "got_relevant_help_rate": mean_optional(score.get("got_relevant_help") for score in scores),
        "felt_understood_rate": mean_optional(score.get("felt_understood") for score in scores),
        "average_actionability": mean_numeric(score.get("actionability") for score in scores),
        "multiple_recommendation_turn_rate": mean(
            score.get("multiple_recommendation_turns") for score in scores
        ),
        "recommended_ids_not_retrieved_rate": mean(
            bool(score.get("recommended_ids_not_retrieved")) for score in scores
        ),
    }


def mean(values) -> float:
    values = list(values)
    return sum(1 for value in values if value) / len(values)


def mean_optional(values) -> float | None:
    values = [value for value in values if value is not None]
    if not values:
        return None
    return mean(values)


def mean_numeric(values) -> float | None:
    values = [value for value in values if isinstance(value, (int, float))]
    if not values:
        return None
    return sum(values) / len(values)


def numeric_or_none(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def bool_or_none(value):
    return value if isinstance(value, bool) else None
