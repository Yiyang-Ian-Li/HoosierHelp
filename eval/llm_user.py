from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any

from agent.llm import (
    create_chat_completion_with_retries,
    create_response_with_retries,
    is_llama_cpp_provider,
    make_openai_client,
)


USER_SYSTEM_PROMPT = """You are simulating a person asking for help finding Indiana community resources.

You must act like the user, not like an assistant. Do not mention that you are simulated, following a behavior pattern, using hidden facts, or obeying instructions.

Use only the facts provided for the current reply. By default, if the agent asks about one or more specific information areas, answer those areas and do not volunteer other valid search facts. When an asked area has multiple provided values, include all of them unless the behavior instruction says to omit or obscure that area. Follow the behavior instructions when they say to be self-contradictory, impatient, tangential, or unrealistic.

Keep the response as a natural user message. When giving a time range, make AM/PM or 24-hour meaning clear for both the start and end time if known. Do not use lists unless the user would naturally list a few items."""


BEHAVIOR_INSTRUCTIONS = {
    "normal": """Behavior:
- Opening: directly state the service need only.
- Follow-ups: answer the information areas the agent asked about completely and directly.
- Do not add unrelated background or extra valid search facts.""",
    "rambling": """Behavior:
- Opening: state the service need, but include extra background noise or an unrelated worry/question.
- Follow-ups: answer the information areas the agent asked about, and add noisy background, unnecessary distractor facts, or off-topic questions.
- Noisy background must not contain any city, county, ZIP code, day, time, intake method, document, eligibility trait, or service need unless it is present in the available facts for this reply.
- Keep noisy background mundane and realistic, limited to being distracted, folding laundry, paperwork on the table, a phone notification, or waiting on a routine callback. Do not mention children, family needs, bills, housing, food, utilities, transportation help, appliance problems, safety concerns, medical issues, legal issues, money needs, jokes, surreal comments, or animal-related tangents.
- Do not provide valid search facts the agent did not ask for.""",
    "self_contradictory": """Behavior:
- Opening: directly state the service need only.
- Follow-ups: answer the information areas the agent asked about.
- One information area is selected as your contradiction slot. Only contradict yourself when answering that selected slot.
- A self-contradiction means you assert a fact or requirement and also deny that same fact or requirement in the same reply.
- If the available facts contain multiple acceptable options, those options are normal constraints, not self-contradictions.
- If the agent asks again to clarify or confirm the contradiction, answer normally with the real fact.
- Do not present the contradiction as a correction. Do not use words such as actually, sorry, or I mean.""",
    "impatience": """Behavior:
- Opening: directly state the service need only.
- Follow-ups: answer the information areas the agent asked about, but sound impatient, rushed, or frustrated about the number of questions.
- You may complain that there are too many questions, say you are in a hurry, or ask the agent to move faster.
- Do not intentionally omit asked information that is available.
- Do not refuse to provide asked information unless the asked information is not in the available facts.
- Do not provide valid search facts the agent did not ask for.""",
    "unsupported_request": """Behavior:
- Opening: express the real service need through a concrete request the agent cannot fulfill directly, such as asking the agent for money, asking the agent to order/pay for something, asking the agent to make a purchase, asking the agent to directly provide an item/service, or asking the agent to personally arrange the outcome.
- The impossible request should be closely related to the real service need and may be distracting. For example, if the hidden need is food assistance, the user might say they are hungry and ask for $100, ask the agent to order food from a restaurant, or ask the agent to buy groceries.
- Follow-ups: answer the information areas the agent asked about, and sometimes repeat or rephrase the impossible request.
- The impossible request is not a valid search fact and should not replace the hidden facts.
- Do not provide valid search facts the agent did not ask for.""",
}

SLOT_GROUPS = (
    ("schedule",),
    ("location", "intake"),
    ("documents", "eligibility"),
)


USER_GENERATION_TOKEN_LIMIT = 8192


@dataclass
class LLMSimulatedUser:
    spec: dict[str, Any]
    user_behavior: str
    provider: str
    model: str
    seed: int | None = None
    temperature: float = 0.0
    turn: int = 0
    client: Any = None
    contradiction_area: str | None = None
    contradiction_used: bool = False
    provided_areas: set[str] = field(default_factory=set)
    rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(f"{self.spec['user_spec_id']}::{self.user_behavior}::{self.seed or 0}")
        self.contradiction_area = self._select_contradiction_area()
        self.client = self.client or make_openai_client(self.provider)

    def opening(self) -> str:
        self.turn += 1
        self.provided_areas.add("service")
        return self._generate([], ["service"], opening=True)

    def respond(self, messages: list[dict[str, Any]], agent_message: str) -> str:
        self.turn += 1
        requested = requested_areas(agent_message)
        if not requested:
            requested = [next_unanswered_area(messages)]
        requested = self._current_slot_group(requested)
        return self._generate(messages, requested, opening=False)

    def _generate(self, messages: list[dict[str, Any]], requested: list[str], opening: bool) -> str:
        visible_areas, turn_instruction = self._turn_plan(requested, opening)
        behavior_key = self.user_behavior
        context = "\n\n".join(
            (
                BEHAVIOR_INSTRUCTIONS[behavior_key],
                f"Current turn type: {'opening' if opening else 'follow-up'}.",
                f"Information areas requested this turn: {', '.join(requested)}.",
                turn_instruction,
                "Facts available for this reply:\n" + json.dumps(hidden_user_facts(self.spec, visible_areas), ensure_ascii=False, indent=2),
                "Do not state, hint at, or invent valid search facts that are not included in the facts available for this reply.",
                "Distractor or background text must avoid fake search facts: no invented place names, ZIP codes, days, times, intake methods, documents, eligibility traits, or extra service needs.",
                "Do not mention information areas that are not available for this reply, even to say no preference, none, unknown, or no information.",
                "If an available requested area is an empty list or empty object, answer naturally that you have no specific information, no requirement, or no preference for that area.",
                "Write exactly one user reply. Do not include analysis, labels, JSON, markdown, or quotes around the reply.",
            )
        )
        if is_llama_cpp_provider(self.provider):
            text = self._generate_chat_completion(messages, context)
        else:
            text = self._generate_responses_api(messages, context)
        self.provided_areas.update(area for area in visible_areas if area != "service")
        return text.strip()

    def _generate_responses_api(self, messages: list[dict[str, Any]], context: str) -> str:
        response = create_response_with_retries(
            self.client,
            model=self.model,
            instructions=USER_SYSTEM_PROMPT,
            input=[*llm_role_mapped_messages(messages), {"role": "user", "content": context}],
            temperature=self.temperature,
            max_output_tokens=USER_GENERATION_TOKEN_LIMIT,
        )
        return getattr(response, "output_text", "") or ""

    def _generate_chat_completion(self, messages: list[dict[str, Any]], context: str) -> str:
        response = create_chat_completion_with_retries(
            self.client,
            model=self.model,
            messages=[
                {"role": "system", "content": USER_SYSTEM_PROMPT},
                *llm_role_mapped_messages(messages),
                {"role": "user", "content": context},
            ],
            temperature=self.temperature,
            max_tokens=USER_GENERATION_TOKEN_LIMIT,
        )
        choice = response.choices[0] if response.choices else None
        message = choice.message if choice is not None else None
        return getattr(message, "content", None) or ""

    def _current_slot_group(self, requested: list[str]) -> list[str]:
        requested_set = set(requested)
        for group in SLOT_GROUPS:
            if all(area in self.provided_areas for area in group):
                continue
            if requested_set.intersection(group) or requested_set.difference({"service"}):
                return list(group)
        return requested

    def _turn_plan(self, requested: list[str], opening: bool) -> tuple[list[str], str]:
        if opening:
            return ["service"], "Opening instruction: mention only the service need from the available facts."

        requested = _dedupe(requested)
        if self.user_behavior == "impatience":
            return requested, (
                "Impatience instruction: answer all requested areas using the available facts, but sound rushed, annoyed, "
                "or frustrated by the number of questions. You may complain or ask the agent to hurry, but do not "
                "intentionally omit asked information that is available."
            )

        if self.user_behavior == "self_contradictory":
            if (
                self.contradiction_area
                and not self.contradiction_used
                and self.contradiction_area in requested
            ):
                self.contradiction_used = True
                return requested, self_contradiction_instruction(self.contradiction_area)
            return requested, self_contradiction_resolved_instruction(self.contradiction_area)

        return requested, "Reply instruction: answer the requested areas using only the available facts."

    def _select_contradiction_area(self) -> str | None:
        if self.user_behavior != "self_contradictory":
            return None
        candidates = [
            area
            for area in ("schedule", "location", "intake", "documents", "eligibility")
            if has_concrete_area_fact(self.spec, area)
        ]
        return self.rng.choice(candidates) if candidates else None


def hidden_user_facts(spec: dict[str, Any], areas: list[str]) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    selected = set(areas)
    needs = normalized_needs(spec)
    if "service" in selected:
        facts["service_needs"] = [
            {
                "need_id": need.get("need_id"),
                "service_categories": need.get("service_categories") or [],
            }
            for need in needs
        ]
        facts["case_type"] = spec.get("case_type") or ("composite" if len(needs) > 1 else "single")
        facts["instruction"] = "Mention the user's real-world need naturally; do not mention category labels unless that is the natural wording."
    if len(needs) > 1 and selected.intersection({"schedule", "location", "intake", "documents", "eligibility"}):
        facts["note"] = "There are two separate needs. When the requested information differs by need, answer for both needs."
    if "schedule" in selected:
        facts["schedule"] = facts_for_needs(needs, "schedule")
    if "location" in selected:
        facts["location"] = facts_for_needs(needs, "location")
    if "intake" in selected:
        facts["intake_methods"] = facts_for_needs(needs, "intake_methods")
    if "documents" in selected:
        facts["available_documents"] = facts_for_needs(needs, "available_documents")
    if "eligibility" in selected:
        facts["eligibility"] = facts_for_needs(needs, "eligibility")
    return facts


def has_concrete_area_fact(spec: dict[str, Any], area: str) -> bool:
    needs = normalized_needs(spec)
    if area == "schedule":
        return any(bool(need.get("schedule")) for need in needs)
    if area == "location":
        return any(
            any((need.get("location") or {}).get(key) for key in ("counties", "cities", "zipcodes"))
            for need in needs
        )
    if area == "intake":
        return any(bool(need.get("intake_methods")) for need in needs)
    if area == "documents":
        return any(bool(need.get("available_documents")) for need in needs)
    if area == "eligibility":
        return any(bool(need.get("eligibility")) for need in needs)
    return False


def normalized_needs(spec: dict[str, Any]) -> list[dict[str, Any]]:
    needs = spec.get("needs")
    if isinstance(needs, list) and needs:
        return [need for need in needs if isinstance(need, dict)]
    return [
        {
            "need_id": "need-1",
            "service_categories": [spec.get("service_category")] if spec.get("service_category") else [],
            "schedule": spec.get("schedule") or {},
            "location": spec.get("location") or {},
            "intake_methods": spec.get("intake_methods") or [],
            "available_documents": spec.get("available_documents") or [],
            "eligibility": spec.get("eligibility") or [],
        }
    ]


def facts_for_needs(needs: list[dict[str, Any]], key: str) -> Any:
    if len(needs) == 1:
        return needs[0].get(key) or ([] if key != "schedule" and key != "location" else {})
    return [
        {
            "need_id": need.get("need_id"),
            "service_categories": need.get("service_categories") or [],
            key: need.get(key) or ([] if key != "schedule" and key != "location" else {}),
        }
        for need in needs
    ]


def self_contradiction_instruction(target: str | None) -> str:
    if not target:
        return self_contradiction_resolved_instruction(target)
    examples = {
        "schedule": "It has to be Monday from 3:00 PM to 4:30 PM, but I cannot do Monday from 3:00 PM to 4:30 PM.",
        "location": "It needs to be in Grant County, but it cannot be in Grant County.",
        "intake": "It has to be by phone, but I cannot use phone intake.",
        "documents": "I can bring photo ID, but I cannot bring photo ID.",
        "eligibility": "I qualify as a senior, but I do not qualify as a senior.",
    }
    example = examples.get(target, "I need that, but I do not need that.")
    return (
        f"Self-contradictory instruction: the predetermined contradiction slot is {target}. "
        f"If this turn asks about {target}, and the transcript does not already contain a contradiction for {target}, "
        f"answer the {target} question with one direct self-contradiction. Use the real available fact for that area, "
        "then deny that same fact or requirement in the same sentence. Use a direct conflict like 'X works for me, but I cannot do X' "
        "or 'I can bring X, but I do not have X.' "
        f"For the style of contradiction, use this pattern as a guide but adapt it to the available facts: {example} "
        "If this turn does not ask about the predetermined contradiction slot, answer normally. "
        "If the agent is asking again to clarify or confirm a previous contradiction, answer normally with the real available fact and do not repeat the contradiction. "
        "Do not turn acceptable alternatives into a contradiction. Do not use uncertainty like not sure or maybe. "
        "Use but to connect the conflict. The reply is invalid if it contains actually, sorry, I mean, not sure, maybe, or after all."
    )


def self_contradiction_resolved_instruction(target: str | None) -> str:
    if not target:
        return (
            "Self-contradictory instruction: this user has no concrete slot fact available for a contradiction. "
            "Answer the requested areas normally using only the available facts."
        )
    return (
        f"Self-contradictory instruction: the predetermined contradiction slot is {target}, "
        "but the contradiction should not be used on this turn. Answer the requested areas normally "
        "using only the real available facts. If the agent is asking to clarify or confirm a previous "
        "contradiction, give the real fact only. Do not repeat, introduce, mention, explain, or apologize "
        "for any contradiction."
    )


def requested_areas(text: str) -> list[str]:
    clean = text.lower()
    areas = []
    if _has_any(clean, ("when", "schedule", "availability", "available", "day", "time", "hour", "morning", "afternoon", "evening", "weekend")):
        areas.append("schedule")
    if _has_any(clean, ("where", "location", "city", "county", "zip", "zipcode", "near", "area", "address")):
        areas.append("location")
    if _has_any(clean, ("intake", "walk", "walk-in", "walk in", "call", "phone", "online", "appointment", "email", "text", "mail", "access")):
        areas.append("intake")
    if _has_any(clean, ("document", "paperwork", "bring", "photo id", "id ", "proof", "license", "birth certificate", "social security", "utility bill")):
        areas.append("documents")
    if _has_any(clean, ("eligible", "eligibility", "qualify", "income", "veteran", "senior", "youth", "pregnant", "disabled", "disability", "resident", "homeless", "medicaid", "uninsured")):
        areas.append("eligibility")
    if _has_any(clean, ("what kind of help", "what do you need", "looking for")) and not areas:
        areas.append("service")
    return _dedupe(areas)


def next_unanswered_area(messages: list[dict[str, Any]]) -> str:
    user_text = "\n".join(str(message.get("content", "")) for message in messages if message.get("role") == "user").lower()
    for area, markers in (
        ("schedule", ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "24-hour", "24 hour")),
        ("location", ("county", "city", "zip")),
        ("intake", ("calling", "walk", "online", "appointment", "email", "text", "mail")),
        ("documents", ("document", "paperwork", "photo id", "proof", "license", "utility bill")),
        ("eligibility", ("eligibility", "low income", "senior", "veteran", "pregnant", "homeless", "uninsured")),
    ):
        if not _has_any(user_text, markers):
            return area
    return "service"


def llm_role_mapped_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    mapped = []
    for message in messages[-10:]:
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        role = message.get("role")
        if role == "assistant":
            mapped.append({"role": "user", "content": content})
        elif role == "user":
            mapped.append({"role": "assistant", "content": content})
    return mapped


def _has_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in text for pattern in patterns)


def _dedupe(values: list[str]) -> list[str]:
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
