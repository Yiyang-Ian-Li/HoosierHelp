from __future__ import annotations

import json
from collections.abc import Callable

from .index import ResourceIndex
from .models import AgentResponse, SearchRequest, SearchResult
from .planner import plan_search
from .tool import (
    request_from_tool_args,
    request_to_tool_call,
    search_resources,
    search_resources_tool_schema,
    tool_result,
    with_limit,
)

Reranker = Callable[[str, list[SearchResult], int], list[SearchResult]]


class Agent211:
    def __init__(
        self,
        index: ResourceIndex,
        client=None,
        model: str | None = None,
        use_openai_tools: bool = False,
        reranker: Reranker | None = None,
        retrieval_limit: int = 30,
    ):
        self.index = index
        self.client = client
        self.model = model
        self.use_openai_tools = use_openai_tools
        self.reranker = reranker
        self.retrieval_limit = retrieval_limit

    def ask(
        self, query: str, request: SearchRequest | None = None, limit: int = 10
    ) -> AgentResponse:
        if self.use_openai_tools and request is None:
            return self._ask_with_openai_tools(query, limit)
        return self._ask_with_heuristic_search(query, request, limit)

    def _ask_with_heuristic_search(
        self, query: str, request: SearchRequest | None, limit: int
    ) -> AgentResponse:
        planned = request or plan_search(
            query, self.index, limit=self.retrieval_limit if self.reranker else limit
        )
        planned = with_limit(
            planned, max(limit, planned.limit, self.retrieval_limit if self.reranker else 0)
        )
        results = search_resources(self.index, planned)
        tool_calls = [request_to_tool_call(planned)]
        if self.reranker:
            results = self.reranker(query, results, limit)
            tool_calls.append(_rerank_call_dict(query, len(results), limit))
        else:
            results = results[:limit]
        return AgentResponse(
            query=query,
            request=planned,
            results=tuple(results),
            answer=format_answer(results),
            tool_calls=tuple(tool_calls),
        )

    def _ask_with_openai_tools(self, query: str, limit: int) -> AgentResponse:
        if self.client is None or not self.model:
            raise ValueError("OpenAI tool calling requires client and model.")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a 211 resource search agent. Call the search_resources "
                    "tool exactly once to retrieve candidate services before answering. "
                    "Use only tool results in the final answer."
                ),
            },
            {"role": "user", "content": query},
        ]
        tool_schema = search_resources_tool_schema(self.index, limit)
        first = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=[tool_schema],
            tool_choice={"type": "function", "function": {"name": "search_resources"}},
            temperature=0,
        )
        message = first.choices[0].message
        tool_calls = message.tool_calls or []
        results: list[SearchResult] = []
        request = SearchRequest(text_query=query, limit=limit)
        recorded_tool_calls = []

        messages.append(_message_to_dict(message))
        for tool_call in tool_calls:
            args = _json_object(tool_call.function.arguments)
            request = request_from_tool_args(args, fallback_query=query, limit=limit)
            request = with_limit(request, self.retrieval_limit if self.reranker else limit)
            results = search_resources(self.index, request)
            recorded_tool_calls.append(request_to_tool_call(request))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result(results), ensure_ascii=False),
                }
            )

        if self.reranker:
            results = self.reranker(query, results, limit)
            recorded_tool_calls.append(_rerank_call_dict(query, len(results), limit))
        else:
            results = results[:limit]

        final = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
        )
        answer = final.choices[0].message.content or format_answer(results)
        return AgentResponse(
            query=query,
            request=request,
            results=tuple(results),
            answer=answer,
            tool_calls=tuple(recorded_tool_calls),
        )


def format_answer(results) -> str:
    if not results:
        return "I could not find a matching resource in the benchmark index."
    lines = ["Top matching resources:"]
    for idx, result in enumerate(results[:5], start=1):
        r = result.resource
        contact = r.phone or r.website or "No contact listed"
        area = ", ".join(r.service_area)
        lines.append(
            f"{idx}. {r.service_name} - {r.agency_name} ({r.city}, {r.state}; {area})"
        )
        lines.append(f"   Contact: {contact}")
        if r.eligibility:
            lines.append(f"   Eligibility: {r.eligibility}")
    return "\n".join(lines)

def _rerank_call_dict(query: str, candidate_count: int, limit: int) -> dict:
    return {
        "tool": "rerank_resources",
        "arguments": {"query": query, "candidate_count": candidate_count, "limit": limit},
    }


def _message_to_dict(message) -> dict:
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    return {
        "role": "assistant",
        "content": getattr(message, "content", None),
        "tool_calls": getattr(message, "tool_calls", None),
    }


def _json_object(value: str) -> dict:
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}

