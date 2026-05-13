from __future__ import annotations

import json

from agent.llm import create_response_with_retries

class Agent:
    def __init__(
        self,
        client,
        model: str,
        tools: list[dict],
        tool_functions: dict,
        instructions: str,
        max_tool_calls: int = 1,
    ):
        self.client = client
        self.model = model
        self.tools = tools
        self.tool_functions = tool_functions
        self.instructions = instructions
        self.max_tool_calls = max_tool_calls
        self.executed_tool_calls = 0

    def ask(
        self,
        query: str,
        history: list | None = None,
        limit: int | None = None,
    ) -> dict:
        input_list = list(history or [])
        input_list.append({"role": "user", "content": query})
        tool_calls = []
        output_text = ""
        token_usage = empty_token_usage()

        while True:
            response = create_response_with_retries(
                self.client,
                model=self.model,
                instructions=self.instructions,
                tools=self.tools,
                input=input_list,
            )
            add_response_usage(token_usage, response)
            output = list(getattr(response, "output", []) or [])
            input_list += output
            output_text = getattr(response, "output_text", "") or ""
            function_calls = [item for item in output if _item_type(item) == "function_call"]

            if not function_calls:
                break
            if self.executed_tool_calls >= self.max_tool_calls:
                for item in function_calls:
                    input_list.append(
                        {
                            "type": "function_call_output",
                            "call_id": _item_attr(item, "call_id"),
                            "output": json.dumps(
                                {
                                    "error": (
                                        "Maximum tool calls reached. Stop calling tools and "
                                        "answer from the information already available."
                                    )
                                }
                            ),
                        }
                    )
                response = create_response_with_retries(
                    self.client,
                    model=self.model,
                    instructions=self.instructions,
                    input=input_list,
                )
                add_response_usage(token_usage, response)
                output = list(getattr(response, "output", []) or [])
                input_list += output
                output_text = getattr(response, "output_text", "") or ""
                break

            for index, item in enumerate(function_calls):
                if self.executed_tool_calls >= self.max_tool_calls:
                    input_list.append(
                        {
                            "type": "function_call_output",
                            "call_id": _item_attr(item, "call_id"),
                            "output": json.dumps(
                                {
                                    "error": (
                                        "Maximum tool calls reached. Stop calling tools and "
                                        "answer from the information already available."
                                    )
                                }
                            ),
                        }
                    )
                    continue
                name = _item_attr(item, "name")
                args = _json_object(_item_attr(item, "arguments") or "{}")
                if name not in self.tool_functions:
                    result = {
                        "error": (
                            f"Unknown tool '{name}'. Use one of the provided tools, "
                            "or stop calling tools and answer with the final JSON."
                        )
                    }
                else:
                    result = self.tool_functions[name](args, limit)
                    self.executed_tool_calls += 1
                tool_calls.append({"tool": name, "arguments": args, "result": result})
                input_list.append(
                    {
                        "type": "function_call_output",
                        "call_id": _item_attr(item, "call_id"),
                        "output": _tool_output_text(result),
                    }
                )
        return {
            "query": query,
            "output_text": output_text,
            "input": input_list,
            "tool_calls": tuple(tool_calls),
            "token_usage": token_usage,
        }


def _item_type(item) -> str | None:
    return _item_attr(item, "type")


def _item_attr(item, name: str):
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def _json_object(value: str) -> dict:
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _tool_output_text(result: object) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False)


def empty_token_usage() -> dict:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def add_response_usage(total: dict, response) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    total["input_tokens"] += int(_usage_attr(usage, "input_tokens") or 0)
    total["output_tokens"] += int(_usage_attr(usage, "output_tokens") or 0)
    total["total_tokens"] += int(_usage_attr(usage, "total_tokens") or 0)


def _usage_attr(usage, name: str):
    if isinstance(usage, dict):
        return usage.get(name)
    return getattr(usage, name, None)
