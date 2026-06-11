from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval.tool_call_backends import qwen_tool_schema
from eval.tool_call_prompts import AGENT_SYSTEM_PROMPT
from eval.tool_call_schema import tool_arg_scores
from tools.indiana211 import load_resource_index, search_resources_tool_schema


TOOL_CALL_NAME = "search_resources"
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def tool_schema_from_resources(resources: Path) -> dict[str, Any]:
    return search_resources_tool_schema(load_resource_index(resources))


def render_qwen_prompt(tokenizer, messages: list[dict[str, Any]], tool_schema: dict[str, Any], *, privileged: bool = False, behavior: str | None = None) -> str:
    system_prompt = privileged_agent_prompt(behavior) if privileged else AGENT_SYSTEM_PROMPT
    chat_messages = [{"role": "system", "content": system_prompt}, *messages]
    tools = [qwen_tool_schema(tool_schema)]
    if getattr(tokenizer, "chat_template", None):
        kwargs = {"tokenize": False, "add_generation_prompt": True, "tools": tools}
        try:
            return tokenizer.apply_chat_template(chat_messages, **kwargs, enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(chat_messages, **kwargs)
    rendered = "\n".join(f"{message['role']}: {message.get('content', '')}" for message in chat_messages)
    return f"{rendered}\nassistant:"


def assistant_completion(message: dict[str, Any], raw_output: str | None = None) -> str:
    if raw_output:
        return raw_output.strip()
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        function = tool_calls[0].get("function", {})
        return serialize_qwen_tool_call(function.get("name") or TOOL_CALL_NAME, function.get("arguments") or {})
    return str(message.get("content") or "").strip()


def serialize_qwen_tool_call(name: str, arguments: dict[str, Any]) -> str:
    payload = {"name": name, "arguments": arguments}
    return "<tool_call>\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n</tool_call>"


def privileged_agent_prompt(behavior: str | None) -> str:
    behavior = behavior or "unknown"
    hint = PRIVILEGED_BEHAVIOR_HINTS.get(behavior, PRIVILEGED_BEHAVIOR_HINTS["unknown"])
    return (
        AGENT_SYSTEM_PROMPT
        + "\n\nPrivileged training-only information:\n"
        + f"- The simulated user behavior is `{behavior}`.\n"
        + "- Use this only to interpret the visible conversation and choose the next assistant action.\n"
        + "- Never mention the behavior label, hidden policy, or privileged information to the user.\n"
        + "- Produce exactly the same kind of user-facing assistant output required by the base instructions.\n"
        + "- Do not explain missing fields, do not describe internal reasoning, and do not write diagnostic notes.\n"
        + "- Do not write confirmations, summaries, examples, headings, or 'next step' text.\n"
        + "- If a follow-up is needed, output only one concise question for the current slot group.\n"
        + "- If enough facts are known, output only the required <tool_call> block with no prose before or after it.\n"
        + hint
    )


PRIVILEGED_BEHAVIOR_HINTS = {
    "normal": "\n- The user should be treated as direct and factual. Follow the ordinary slot order.",
    "rambling": (
        "\n- The user may include tangents and irrelevant background."
        "\n- Extract only facts that answer the current slot group."
        "\n- Ignore off-topic questions and do not let tangents change service, location, schedule, intake, documents, or eligibility."
    ),
    "impatience": (
        "\n- The user may sound rushed or annoyed."
        "\n- Keep replies concise and do not mirror frustration."
        "\n- Do not skip required slot groups because the user is impatient."
    ),
    "self_contradictory": (
        "\n- The user may give one direct contradiction inside a slot."
        "\n- If the current slot is contradictory, ask one focused clarification."
        "\n- If the user later gives a normal clarification, preserve that clarified fact in the final tool call instead of clearing the field."
    ),
    "unsupported_request": (
        "\n- The user may ask for money, purchases, direct arrangement, guarantees, or another impossible outcome."
        "\n- Do not promise the impossible outcome."
        "\n- Identify the underlying resource-search need and continue collecting the required slot facts."
    ),
    "unknown": "\n- Infer robustly from the visible conversation and follow the base task instructions.",
}


def score_serialized_tool_call(completion: str, expected: dict[str, Any]) -> dict[str, bool]:
    from eval.tool_call_parsers import parse_qwen_xml_tool_call

    parsed = parse_qwen_xml_tool_call(completion)
    return tool_arg_scores(parsed.arguments if parsed else None, expected)
