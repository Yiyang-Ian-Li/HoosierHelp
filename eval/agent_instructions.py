from __future__ import annotations


PROTOCOLS = {
    "easy": [
        ("service", "the main kind of help needed", "service_categories"),
        ("location", "exactly one of county, city, or ZIP code", "counties, cities, or zipcodes"),
        ("intake", "required contact or access method", "intake_methods"),
    ],
    "medium": [
        ("service", "the main kind of help needed", "service_categories"),
        ("location", "exactly one of county, city, or ZIP code", "counties, cities, or zipcodes"),
        ("intake", "required contact or access method", "intake_methods"),
        ("schedule", "required day, day/time, or 24-hour availability", "available_days, available_time_windows, or requires_24_hours"),
    ],
    "hard": [
        ("service", "the main kind of help needed", "service_categories"),
        ("location", "exactly one of county, city, or ZIP code", "counties, cities, or zipcodes"),
        ("intake", "required contact or access method", "intake_methods"),
        ("schedule", "required day, day/time, or 24-hour availability", "available_days, available_time_windows, or requires_24_hours"),
        ("documents", "documents the user can provide, or none", "documents_available"),
    ],
}

BASE_INSTRUCTIONS = """
You are an Indiana 211 resource retrieval agent in a benchmark evaluation. Your
job is to understand the user's situation, call the search tool once, and make
one final resource recommendation.

Ask concise follow-up questions until you have all the required information for 
this task. Ask about only one area at a time. Do not ask for information outside 
the required list.

The search tool uses hard filters for every non-empty field. Build one precise
tool call using only the fields listed for this task.

You may call the search tool at most once. After that call, give exactly one
final response: recommend one concrete resource from the results, or explain
that no exact match was found. A recommendation is final: once you include a
resource ID, the interaction will stop. 

Only recommend concrete resources from tool results. When ready, use this
format:
- Resource name (resource_id): why it fits
Example: - Food Pantry (in211-123-456-food-pantry): close to your county and
offers groceries this week.
Copy the full resource_id exactly as it appears in the tool result.
""".strip()

REACT_SUFFIX = """
Use a ReAct-style response format. Before each visible assistant reply, write a
brief `Thought:` line that explains what you are doing next. Then write
`Answer:` with the user-facing message.
""".strip()


def agent_instructions(agent_type: str, difficulty: str | None = None) -> str:
    instructions = benchmark_instructions(difficulty)
    if agent_type == "react":
        return instructions + "\n\n" + REACT_SUFFIX
    return instructions


def benchmark_instructions(difficulty: str | None = None) -> str:
    fields = PROTOCOLS.get(difficulty or "", PROTOCOLS["hard"])
    return (
        BASE_INSTRUCTIONS
        + "\n\n"
        + "Required information for this task:\n"
        + required_information_text(fields)
        + "\n\n"
        + "Tool fields for this task:\n"
        + tool_mapping_text(fields)
    )


def required_information_text(fields: list[tuple[str, str, str]]) -> str:
    return "\n".join(f"- {name}: {description}" for name, description, _ in fields)


def tool_mapping_text(fields: list[tuple[str, str, str]]) -> str:
    return "\n".join(f"- {name} -> {tool_field}" for name, _, tool_field in fields)
