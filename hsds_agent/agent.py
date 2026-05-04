from __future__ import annotations

import json
import sqlite3

from .llm import LLMClient
from .models import AgentAnswer, ResourceCandidate, SearchRequest
from .tools import search_services


SEARCH_SERVICES_TOOL = {
    "type": "function",
    "function": {
        "name": "search_services",
        "description": (
            "Search the HSDS-style social support resource database for services "
            "matching a user's need, location, language, eligibility, and timing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language service need from the user.",
                },
                "location": {
                    "type": ["string", "null"],
                    "description": "City or ZIP code. Use null if missing.",
                },
                "radius_miles": {
                    "type": "number",
                    "description": "Search radius in miles.",
                    "default": 10,
                },
                "categories": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "food",
                            "housing",
                            "transportation",
                            "mental_health",
                            "legal",
                            "childcare",
                            "benefits",
                        ],
                    },
                    "description": "Normalized service categories.",
                },
                "languages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Requested service languages.",
                },
                "eligibility": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Eligibility constraints such as senior or family.",
                },
                "open_now": {
                    "type": "boolean",
                    "description": "Whether the user specifically needs currently open services.",
                    "default": False,
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 5,
                },
            },
            "required": ["query", "location"],
            "additionalProperties": False,
        },
    },
}


class ResourceAgent:
    def __init__(self, conn: sqlite3.Connection, llm: LLMClient):
        self.conn = conn
        self.llm = llm

    def answer(self, question: str) -> AgentAnswer:
        tool_calls: list[dict] = []
        candidates: list[ResourceCandidate] = []

        def execute_tool(name: str, arguments: dict) -> str:
            if name != "search_services":
                return json.dumps({"error": f"Unknown tool: {name}"})

            request = _search_request_from_args(arguments, question)
            tool_calls.append(
                {
                    "tool": name,
                    "arguments": {
                        "query": request.query,
                        "location": request.location,
                        "radius_miles": request.radius_miles,
                        "categories": list(request.categories),
                        "languages": list(request.languages),
                        "eligibility": list(request.eligibility),
                        "open_now": request.open_now,
                        "limit": request.limit,
                    },
                }
            )

            if request.location is None:
                return json.dumps(
                    {
                        "needs_follow_up": True,
                        "message": "Ask the user for a city or ZIP code.",
                    }
                )

            results = tuple(search_services(self.conn, request))
            candidates.extend(results)
            return json.dumps(
                {
                    "needs_follow_up": False,
                    "resources": [_candidate_to_dict(candidate) for candidate in results],
                    "result_count": len(results),
                }
            )

        answer = self.llm.answer_with_tools(
            question=question,
            tools=[SEARCH_SERVICES_TOOL],
            execute_tool=execute_tool,
        )
        return AgentAnswer(
            answer=answer,
            needs_follow_up=_needs_follow_up(answer, candidates),
            tool_calls=tuple(tool_calls),
            candidates=tuple(candidates),
        )


def _search_request_from_args(arguments: dict, question: str) -> SearchRequest:
    return SearchRequest(
        query=_string(arguments.get("query"), question),
        location=_optional_string(arguments.get("location")),
        radius_miles=_float(arguments.get("radius_miles"), 10.0),
        categories=tuple(_string_list(arguments.get("categories"))),
        languages=tuple(_string_list(arguments.get("languages"))),
        eligibility=tuple(_string_list(arguments.get("eligibility"))),
        open_now=bool(arguments.get("open_now", False)),
        limit=_int(arguments.get("limit"), 5),
    )


def _candidate_to_dict(candidate: ResourceCandidate) -> dict:
    return {
        "service_id": candidate.service_id,
        "service_name": candidate.service_name,
        "organization_name": candidate.organization_name,
        "description": candidate.description,
        "categories": list(candidate.categories),
        "address": candidate.address,
        "city": candidate.city,
        "region": candidate.region,
        "postal_code": candidate.postal_code,
        "distance_miles": candidate.distance_miles,
        "phone": candidate.phone,
        "website": candidate.website,
        "languages": list(candidate.languages),
        "eligibility": candidate.eligibility,
        "schedule": candidate.schedule,
        "score": candidate.score,
        "source_fields": list(candidate.source_fields),
    }


def _needs_follow_up(answer: str, candidates: list[ResourceCandidate]) -> bool:
    if candidates:
        return False
    lowered = answer.lower()
    return "zip" in lowered or "city" in lowered or "location" in lowered


def _string(value: object, default: str) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else default


def _optional_string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _float(value: object, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(1.0, min(100.0, parsed))


def _int(value: object, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(10, parsed))
