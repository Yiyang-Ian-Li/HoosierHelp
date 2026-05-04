from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Protocol


ToolExecutor = Callable[[str, dict], str]


class LLMClient(Protocol):
    def answer_with_tools(
        self, question: str, tools: list[dict], execute_tool: ToolExecutor
    ) -> str:
        ...


class OpenAICompatibleLLM:
    """OpenAI-style chat completion client with tool-call execution."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
    ):
        from openai import OpenAI

        self.model = model
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        if default_headers:
            kwargs["default_headers"] = default_headers
        self.client = OpenAI(**kwargs)

    @classmethod
    def from_env(cls, provider: str = "openrouter") -> "OpenAICompatibleLLM":
        load_dotenv()
        provider = provider.lower()
        if provider == "openai":
            return cls(
                api_key=os.environ["OPENAI_API_KEY"],
                model=os.getenv("HSDS_AGENT_MODEL", "gpt-4.1-mini"),
                base_url=os.getenv("OPENAI_BASE_URL"),
            )

        return cls(
            api_key=os.environ["OPENROUTER_API_KEY"],
            model=os.getenv("HSDS_AGENT_MODEL", "openai/gpt-4.1-mini"),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            default_headers=_openrouter_headers(),
        )

    def answer_with_tools(
        self, question: str, tools: list[dict], execute_tool: ToolExecutor
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a social support resource assistant backed by an "
                    "HSDS-style database. Use the provided tools to retrieve resource "
                    "records before recommending services. Do not invent services, "
                    "eligibility, hours, locations, or contact details. If the user "
                    "does not provide a city or ZIP code, ask a brief follow-up instead "
                    "of guessing. Tell users to confirm details before visiting. The "
                    "current data is synthetic experiment data, not a real referral "
                    "directory."
                ),
            },
            {"role": "user", "content": question},
        ]

        first = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0,
        )
        message = first.choices[0].message
        tool_calls = message.tool_calls or []
        if not tool_calls:
            return message.content or ""

        messages.append(message.model_dump(exclude_none=True))
        for tool_call in tool_calls:
            name = tool_call.function.name
            arguments = _json_object(tool_call.function.arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": execute_tool(name, arguments),
                }
            )

        final = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
        )
        return final.choices[0].message.content or ""


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _json_object(value: str) -> dict:
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _openrouter_headers() -> dict[str, str]:
    headers = {}
    referer = os.getenv("OPENROUTER_HTTP_REFERER")
    title = os.getenv("OPENROUTER_APP_TITLE")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return headers
