from __future__ import annotations

import json

AGENT_INSTRUCTIONS = """
You are an Indiana 211 resource retrieval agent. Your job is to understand the
user's situation, call the search tool as needed, and make one final resource
recommendation.

The user may not provide all needed information up front. Ask concise follow-up
questions when important details are missing. You may call the tool multiple
times as you learn more or need to refine the search.

Only give a final recommendation when you are very confident the resource fits
the user's need and situation. A recommendation is final: once you include a
resource ID, the interaction will stop. Do not include resource IDs in tentative
options, clarification questions, or progress updates.

Only recommend concrete resources from tool results. When ready, use this
format:
- Resource name (resource_id): why it fits
Example: - Food Pantry (in211-123-456-food-pantry): close to your county and
offers groceries this week.
Copy the full resource_id exactly as it appears in the tool result.
""".strip()

REACT_INSTRUCTIONS = (
    AGENT_INSTRUCTIONS
    + "\n\n"
    + """
Use a ReAct-style response format. Before each visible assistant reply, write a
brief `Thought:` line that explains what you are doing next. Then write
`Answer:` with the user-facing message.
""".strip()
)
MAX_TOOL_ROUNDS = 8


class Agent:
    def __init__(
        self,
        client,
        model: str,
        tools: list[dict],
        tool_functions: dict,
        instructions: str = AGENT_INSTRUCTIONS,
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
        limit: int | None = None,
    ) -> dict:
        input_list = list(history or [])
        input_list.append({"role": "user", "content": query})
        tool_calls = []
        output_text = ""
        executed_tool_rounds = 0
        token_usage = empty_token_usage()

        while True:
            response = self.client.responses.create(
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
            if executed_tool_rounds >= MAX_TOOL_ROUNDS:
                for item in function_calls:
                    input_list.append(
                        {
                            "type": "function_call_output",
                            "call_id": _item_attr(item, "call_id"),
                            "output": json.dumps(
                                {
                                    "error": (
                                        "Maximum tool rounds reached. Stop calling tools and "
                                        "answer from the information already available."
                                    )
                                }
                            ),
                        }
                    )
                response = self.client.responses.create(
                    model=self.model,
                    instructions=self.instructions,
                    input=input_list,
                )
                add_response_usage(token_usage, response)
                output = list(getattr(response, "output", []) or [])
                input_list += output
                output_text = getattr(response, "output_text", "") or ""
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
    result = _with_empty_result_guidance(result)
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False)


def _with_empty_result_guidance(result: object) -> object:
    if not isinstance(result, dict):
        return result
    resources = result.get("resources")
    if resources != []:
        return result
    return {
        **result,
        "retry_guidance": (
            "No resources matched this exact query. Consider calling the tool again with "
            "appropriately relaxed filters before answering."
        ),
    }


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
