from __future__ import annotations

from typing import Any

from tools.indiana211 import search_resources_tool_schema
from tools.indiana211_models import ResourceIndex


SEARCH_RESOURCES_TOOL_NAME = "search_resources"
FINAL_RECOMMENDATION_TOOL_NAME = "final_recommendation"


def agent_tool_schemas(index: ResourceIndex) -> list[dict[str, Any]]:
    return [
        search_resources_tool_schema(index),
        final_recommendation_tool_schema(),
    ]


def final_recommendation_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "name": FINAL_RECOMMENDATION_TOOL_NAME,
        "description": (
            "End the task with the selected Indiana 211 resource_id values after "
            "search_resources has returned matching resources. Use this only as "
            "the final action."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "resource_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Selected resource_id values exactly as returned by search_resources.",
                },
                "message": {
                    "type": "string",
                    "description": "Short final message for the user.",
                },
            },
            "required": ["resource_ids"],
            "additionalProperties": False,
        },
    }


def normalize_final_recommendation_args(args: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(args, dict):
        args = {}
    return {
        "resource_ids": _string_list(args.get("resource_ids")),
        "message": _clean(args.get("message")),
    }


def qwen_tool_schema(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": schema["name"],
            "description": schema.get("description", ""),
            "parameters": schema["parameters"],
        },
    }


def qwen_tool_schemas(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [qwen_tool_schema(schema) for schema in schemas]


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


def _clean(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""
