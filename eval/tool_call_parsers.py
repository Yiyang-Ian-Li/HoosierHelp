from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from eval.tool_call_schema import normalize_tool_args


TOOL_NAME = "search_resources"
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(?P<payload>\{.*?\})\s*</tool_call>", re.DOTALL)


@dataclass(frozen=True)
class ParsedToolCall:
    name: str
    arguments: dict[str, Any]
    parse_mode: str


def parse_qwen_xml_tool_call(text: str) -> ParsedToolCall | None:
    match = TOOL_CALL_RE.search(text)
    if match:
        payload = _json_object(match.group("payload"))
        return _tool_call_from_object(payload, "qwen_xml")
    payload = _json_object(text.strip())
    return _openai_function_call_from_object(payload, "qwen_openai_function_json")


def parse_responses_tool_call(response: Any) -> ParsedToolCall | None:
    for item in list(getattr(response, "output", []) or []):
        item_type = _item_attr(item, "type")
        if item_type not in {"function_call", "tool_call"}:
            continue
        name = _item_attr(item, "name")
        arguments = _json_object(_item_attr(item, "arguments") or "{}")
        if name == TOOL_NAME and isinstance(arguments, dict):
            return ParsedToolCall(TOOL_NAME, normalize_tool_args(arguments), "responses_function_call")
    return None


def clean_tool_call_text(text: str) -> str:
    return TOOL_CALL_RE.sub("", text).strip()


def _tool_call_from_object(obj: Any, parse_mode: str) -> ParsedToolCall | None:
    if not isinstance(obj, dict):
        return None
    if obj.get("name") == TOOL_NAME and isinstance(obj.get("arguments"), dict):
        return ParsedToolCall(TOOL_NAME, normalize_tool_args(obj["arguments"]), parse_mode)
    if obj.get("type") == "tool_call" and obj.get("name") == TOOL_NAME and isinstance(obj.get("arguments"), dict):
        return ParsedToolCall(TOOL_NAME, normalize_tool_args(obj["arguments"]), parse_mode)
    function = obj.get("function")
    if isinstance(function, dict) and function.get("name") == TOOL_NAME and isinstance(function.get("arguments"), dict):
        return ParsedToolCall(TOOL_NAME, normalize_tool_args(function["arguments"]), parse_mode)
    return None


def _openai_function_call_from_object(obj: Any, parse_mode: str) -> ParsedToolCall | None:
    if not isinstance(obj, dict) or obj.get("type") != "function":
        return None
    function = obj.get("function")
    if isinstance(function, dict) and function.get("name") == TOOL_NAME and isinstance(function.get("arguments"), dict):
        return ParsedToolCall(TOOL_NAME, normalize_tool_args(function["arguments"]), parse_mode)
    return None


def _json_object(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _item_attr(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)
