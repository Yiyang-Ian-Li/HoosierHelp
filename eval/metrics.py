from __future__ import annotations

import json
import re


RESOURCE_ID_RE = re.compile(r"\bin211-[a-z0-9]+(?:-[a-z0-9]+)*\b", re.IGNORECASE)
MAX_RECOMMENDATIONS_FOR_SCORING = 3
JSON_DECODER = json.JSONDecoder()


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
    parsed = final_json_from_response(response, transcript)
    if parsed is not None:
        return recommended_ids_from_final_json(parsed)
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
) -> dict:
    retrieved_ids = extract_resource_ids(final_response)
    recommended_ids = extract_recommended_resource_ids(final_response, transcript)
    expected = set(ground_truth.get("ground_truth_resource_ids", []))
    diagnostics = recommendation_diagnostics(transcript, final_response, retrieved_ids)
    id_hit = expected.issubset(set(recommended_ids))
    retrieval_hit = expected.issubset(set(retrieved_ids))
    parsed_final_json = final_json_from_response(final_response, transcript)
    detail_scores = detail_hit_scores_from_parsed_json(
        parsed_final_json,
        ground_truth.get("ground_truth_resources", []),
    )
    json_scores = final_json_scores(final_response, transcript, ground_truth.get("ground_truth_resources", []))
    answer_detail_hit = id_hit and detail_scores["intake_hit"] and detail_scores["document_hit"]
    return {
        "user_id": card["user_id"],
        "retrieved_resource_ids": retrieved_ids,
        "recommended_resource_ids": recommended_ids,
        "id_hit": id_hit,
        "answer_detail_hit": answer_detail_hit,
        "retrieval_hit": retrieval_hit,
        **json_scores,
        **detail_scores,
        "tool_call_count": count_function_calls(final_response),
        "turn_count": len([turn for turn in transcript if turn["role"] == "user"]),
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


def final_recommendation_text(transcript: list[dict], final_response: dict) -> str:
    for turn in reversed(transcript):
        if turn.get("role") != "agent":
            continue
        text = str(turn.get("content", ""))
        if extract_resource_ids_from_text(text):
            return text
    return str(final_response.get("output_text", ""))


def detail_hit_scores(text: str, ground_truth_resources: list[dict]) -> dict:
    parsed = parse_final_json(text)
    return detail_hit_scores_from_parsed_json(parsed, ground_truth_resources)


def detail_hit_scores_from_parsed_json(parsed: dict | None, ground_truth_resources: list[dict]) -> dict:
    if parsed is None:
        return {
            "intake_hit": False,
            "document_hit": False,
            "per_resource_intake_hit": [False for _ in ground_truth_resources],
            "per_resource_document_hit": [False for _ in ground_truth_resources],
        }
    return detail_hit_scores_from_json(parsed, ground_truth_resources)


def final_json_scores(final_response: dict, transcript: list[dict], ground_truth_resources: list[dict]) -> dict:
    parsed = final_json_from_response(final_response, transcript)
    recommendations = parsed.get("recommendations") if parsed else []
    strict_valid = final_json_strict_valid(final_response, transcript)
    parse_mode = final_json_parse_mode(final_response, transcript)
    return {
        "final_json_valid": parsed is not None,
        "final_json_strict_valid": strict_valid,
        "final_json_embedded_valid": parsed is not None and not strict_valid,
        "final_json_parse_mode": parse_mode,
        "recommended_resource_detail_count": len((recommendations or [])[:MAX_RECOMMENDATIONS_FOR_SCORING]),
        "submitted_recommendation_count": len(recommendations or []),
    }


def final_json_from_response(response: dict, transcript: list[dict] | None = None) -> dict | None:
    result = response.get("structured_result") or {}
    if isinstance(result.get("final_json"), dict):
        return result["final_json"]
    if transcript:
        return parse_final_json(final_recommendation_text(transcript, response))
    return parse_final_json(str(response.get("output_text", "")))


def final_json_strict_valid(response: dict, transcript: list[dict] | None = None) -> bool:
    result = response.get("structured_result") or {}
    if "final_json_strict_valid" in result:
        return bool(result.get("final_json_strict_valid"))
    if isinstance(result.get("final_json"), dict):
        return bool(result.get("final_json_valid"))
    text = final_recommendation_text(transcript, response) if transcript else str(response.get("output_text", ""))
    return parse_strict_final_json(text) is not None


def final_json_parse_mode(response: dict, transcript: list[dict] | None = None) -> str:
    result = response.get("structured_result") or {}
    mode = result.get("final_json_parse_mode")
    if isinstance(mode, str) and mode:
        return mode
    if isinstance(result.get("final_json"), dict):
        return "strict" if result.get("final_json_valid") else "embedded"
    text = final_recommendation_text(transcript, response) if transcript else str(response.get("output_text", ""))
    return parse_final_json_mode(text)


def parse_final_json(text: str) -> dict | None:
    return parse_strict_final_json(text) or parse_embedded_final_json(text)


def parse_strict_final_json(text: str) -> dict | None:
    text = strip_react_prefix(text).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if valid_final_json_object(parsed) else None


def parse_embedded_final_json(text: str) -> dict | None:
    text = strip_react_prefix(text)
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = JSON_DECODER.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if valid_final_json_object(parsed):
            return parsed
    return None


def parse_final_json_mode(text: str) -> str:
    if parse_strict_final_json(text) is not None:
        return "strict"
    if parse_embedded_final_json(text) is not None:
        return "embedded"
    return "none"


def valid_final_json_object(parsed) -> bool:
    if not isinstance(parsed, dict):
        return False
    recommendations = parsed.get("recommendations")
    if not isinstance(recommendations, list):
        return False
    for item in recommendations:
        if not isinstance(item, dict):
            return False
        if not isinstance(item.get("resource_id"), str) or not item["resource_id"].strip():
            return False
        if not isinstance(item.get("resource_name"), str) or not item["resource_name"].strip():
            return False
        for key in ("intake_methods", "document_requirements"):
            if not isinstance(item.get(key), list) or not all(isinstance(value, str) for value in item[key]):
                return False
    return True


def strip_react_prefix(text: str) -> str:
    marker = "Answer:"
    if marker in text:
        return text.split(marker, 1)[1]
    return text


def recommended_ids_from_final_json(parsed: dict) -> list[str]:
    ids = []
    for item in (parsed.get("recommendations") or [])[:MAX_RECOMMENDATIONS_FOR_SCORING]:
        resource_id = item.get("resource_id")
        if not isinstance(resource_id, str):
            continue
        normalized = resource_id.strip().lower()
        if normalized and normalized not in ids:
            ids.append(normalized)
    return ids


def detail_hit_scores_from_json(parsed: dict, ground_truth_resources: list[dict]) -> dict:
    by_id = {}
    for item in (parsed.get("recommendations") or [])[:MAX_RECOMMENDATIONS_FOR_SCORING]:
        resource_id = item.get("resource_id")
        if isinstance(resource_id, str):
            by_id[resource_id.strip().lower()] = item
    intake_hits = []
    document_hits = []
    for resource in ground_truth_resources:
        item = by_id.get(str(resource.get("resource_id", "")).lower())
        if not item:
            intake_hits.append(False)
            document_hits.append(False)
            continue
        intake_hits.append(set(resource.get("intake_methods") or []) == set(item.get("intake_methods") or []))
        document_hits.append(
            set(resource.get("document_requirements") or []) == set(item.get("document_requirements") or [])
        )
    return {
        "intake_hit": all(intake_hits) if intake_hits else False,
        "document_hit": all(document_hits) if document_hits else False,
        "per_resource_intake_hit": intake_hits,
        "per_resource_document_hit": document_hits,
    }


def count_function_calls(response: dict) -> int:
    return len(executed_tool_call_records(response))


def executed_tool_call_records(response: dict) -> list[dict]:
    records = []
    seen = set()
    pending_calls = {}
    for item in response.get("input", []) or []:
        item_type = item_get(item, "type")
        if item_type == "function_call":
            call_id = item_get(item, "call_id")
            pending_calls[call_id] = parse_json_object(item_get(item, "arguments") or "{}")
            continue
        if item_type != "function_call_output":
            continue
        output = item_get(item, "output") or ""
        result = parse_tool_output(output)
        if "resources" not in result and "error" in result:
            continue
        key = (item_get(item, "call_id"), output)
        if key in seen:
            continue
        seen.add(key)
        records.append(
            {
                "arguments": pending_calls.get(item_get(item, "call_id"), {}),
                "result": result,
            }
        )
    if records:
        return records
    for call in response.get("tool_calls", ()) or ():
        arguments = call.get("arguments") or {}
        result = call.get("result") or {}
        records.append({"arguments": arguments, "result": result})
    return records


def parse_json_object(text: str) -> dict:
    try:
        parsed = json.loads(text or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


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
        "id_hit_rate": mean(score.get("id_hit") for score in scores),
        "answer_detail_hit_rate": mean(score.get("answer_detail_hit") for score in scores),
        "intake_hit_rate": mean(score.get("intake_hit") for score in scores),
        "document_hit_rate": mean(score.get("document_hit") for score in scores),
        "final_json_valid_rate": mean(score.get("final_json_valid") for score in scores),
        "final_json_strict_valid_rate": mean(score.get("final_json_strict_valid") for score in scores),
        "final_json_embedded_valid_rate": mean(score.get("final_json_embedded_valid") for score in scores),
        "retrieval_hit_rate": mean(score["retrieval_hit"] for score in scores),
        "average_tool_calls": sum(score["tool_call_count"] for score in scores) / len(scores),
        "average_turns": sum(score["turn_count"] for score in scores) / len(scores),
        "multiple_recommendation_turn_rate": mean(
            score.get("multiple_recommendation_turns") for score in scores
        ),
        "recommended_ids_not_retrieved_rate": mean(
            bool(score.get("recommended_ids_not_retrieved")) for score in scores
        ),
    }


def aggregate_breakdown(cases: list[dict], scores: list[dict] | None = None) -> dict:
    if scores is not None:
        cases = [
            {
                **case,
                "score": score,
            }
            for case, score in zip(cases, scores)
        ]
    return {
        "by_case_type": _aggregate_case_groups(
            cases,
            lambda case: str(case.get("card", {}).get("case_type") or "unknown"),
        ),
        "by_trait": _aggregate_case_groups(
            cases,
            lambda case: _case_trait(case),
        ),
    }


def _aggregate_case_groups(cases: list[dict], group_key) -> dict:
    groups = {}
    for case in cases:
        key = group_key(case)
        groups.setdefault(key, []).append(case)
    return {
        key: _aggregate_case_group(group_cases)
        for key, group_cases in sorted(groups.items())
    }


def _aggregate_case_group(cases: list[dict]) -> dict:
    scores = [case["score"] for case in cases]
    case_count = len(cases)
    id_hits = sum(1 for score in scores if score.get("id_hit"))
    answer_detail_hits = sum(1 for score in scores if score.get("answer_detail_hit"))
    retrieval_hits = sum(1 for score in scores if score.get("retrieval_hit"))
    no_match_count = sum(1 for case in cases if case.get("stop_reason") == "no_match")
    return {
        "cases": case_count,
        "id_hits": id_hits,
        "id_hit_rate": id_hits / case_count if case_count else 0,
        "answer_detail_hits": answer_detail_hits,
        "answer_detail_hit_rate": answer_detail_hits / case_count if case_count else 0,
        "retrieval_hits": retrieval_hits,
        "retrieval_hit_rate": retrieval_hits / case_count if case_count else 0,
        "no_match_count": no_match_count,
        "average_turns": sum(score["turn_count"] for score in scores) / case_count if case_count else 0,
    }


def _case_trait(case: dict) -> str:
    return str(case["card"]["traits"][0])


def mean(values) -> float:
    values = list(values)
    return sum(1 for value in values if value) / len(values)
