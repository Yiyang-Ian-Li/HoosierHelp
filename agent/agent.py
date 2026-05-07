from __future__ import annotations

import json

DEFAULT_INSTRUCTIONS = (
    "You are a resource search agent. Call available tools before answering "
    "when tool data is needed. Use only tool results in the final answer."
)
MAX_TOOL_ROUNDS = 8


class Agent:
    def __init__(
        self,
        client,
        model: str,
        tools: list[dict],
        tool_functions: dict,
        instructions: str = DEFAULT_INSTRUCTIONS,
    ):
        self.client = client
        self.model = model
        self.tools = tools
        self.tool_functions = tool_functions
        self.instructions = instructions

    def ask(
        self,
        query: str,
        history: list | None = None,
        limit: int = 10,
    ) -> dict:
        input_list = list(history or [])
        input_list.append({"role": "user", "content": query})
        tool_calls = []
        output_text = ""
        executed_tool_rounds = 0

        while True:
            response = self.client.responses.create(
                model=self.model,
                instructions=self.instructions,
                tools=self.tools,
                input=input_list,
            )
            output = list(getattr(response, "output", []) or [])
            input_list += output
            output_text = getattr(response, "output_text", "") or ""
            function_calls = [item for item in output if _item_type(item) == "function_call"]

            if not function_calls:
                break
            if executed_tool_rounds >= MAX_TOOL_ROUNDS:
                break

            for item in function_calls:
                name = _item_attr(item, "name")
                args = _json_object(_item_attr(item, "arguments") or "{}")
                result = self.tool_functions[name](args, limit)
                tool_calls.append({"tool": name, "arguments": args, "result": result})
                input_list.append(
                    {
                        "type": "function_call_output",
                        "call_id": _item_attr(item, "call_id"),
                        "output": _tool_output_text(result),
                    }
                )
            executed_tool_rounds += 1

        return {
            "query": query,
            "output_text": output_text,
            "input": input_list,
            "tool_calls": tuple(tool_calls),
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
