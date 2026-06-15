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

Use the user profile as what this person knows about their own situation. Answer the agent's latest question naturally and do not volunteer search facts the agent did not ask for. Follow the behavior instructions when they say to be self-contradictory, impatient, tangential, or unrealistic.

Keep the response as a natural user message, not an intake form or organized checklist. When giving a time range, make AM/PM or 24-hour meaning clear for both the start and end time if known. Do not use lists unless the user would naturally list a few items."""


BEHAVIOR_INSTRUCTIONS = {
    "normal": """Behavior:
- Opening: directly state the service need only. If there are multiple service needs, mention all of them.
- Follow-ups: answer the information areas the agent asked about completely and directly.
- Do not add unrelated background or extra valid search facts.""",
    "rambling": """Behavior:
- Opening: state the service need. If there are multiple service needs, mention all of them. Include extra background noise or an unrelated worry/question.
- Follow-ups: answer the information areas the agent asked about, and add noisy background, unnecessary distractor facts, or off-topic questions.
- Noisy background must not contain any city, county, ZIP code, day, time, intake method, document, eligibility trait, or service need unless it is present in the user profile.
- Keep noisy background mundane and realistic, limited to being distracted, folding laundry, paperwork on the table, a phone notification, or waiting on a routine callback. Do not mention children, family needs, bills, housing, food, utilities, transportation help, appliance problems, safety concerns, medical issues, legal issues, money needs, jokes, surreal comments, or animal-related tangents.
- Do not provide valid search facts the agent did not ask for.""",
    "self_contradictory": """Behavior:
- Opening: directly state the service need only. If there are multiple service needs, mention all of them.
- Follow-ups: answer the information areas the agent asked about.
- One information area is selected as your contradiction slot. Only contradict yourself when answering that selected slot.
- A self-contradiction means you assert a fact or requirement and also deny that same fact or requirement in the same reply.
- If the user profile contains multiple acceptable options, those options are normal constraints, not self-contradictions.
- If the agent asks again to clarify or confirm the contradiction, answer normally with the real fact.
- Do not present the contradiction as a correction. Do not use words such as actually, sorry, or I mean.""",
    "impatience": """Behavior:
- Opening: directly state the service need only. If there are multiple service needs, mention all of them.
- Follow-ups: answer the information areas the agent asked about, but sound impatient, rushed, or frustrated about the number of questions.
- You may complain that there are too many questions, say you are in a hurry, or ask the agent to move faster.
- Do not intentionally omit asked information that is available.
- Do not refuse to provide asked information unless the asked information is not in the user profile.
- Do not provide valid search facts the agent did not ask for.""",
    "unsupported_request": """Behavior:
- Opening: express the real service need through a concrete request the agent cannot fulfill directly, such as asking the agent for money, asking the agent to order/pay for something, asking the agent to make a purchase, asking the agent to directly provide an item/service, or asking the agent to personally arrange the outcome.
- If there are multiple service needs, mention all of them by the end of the opening message.
- The impossible request should be closely related to the real service need and may be distracting. For example, if the hidden need is food assistance, the user might say they are hungry and ask for $100, ask the agent to order food from a restaurant, or ask the agent to buy groceries.
- Do not transform the real service need into a different service category. For example, if the real need is education or youth programs, ask the agent to enroll, arrange, or pay for that program, but do not describe the need as general financial assistance.
- Do not invent a location, day, time, intake method, document, eligibility trait, or extra service need inside the impossible request unless that fact is present in the user profile.
- Follow-ups: answer the information areas the agent asked about, and sometimes repeat or rephrase the impossible request.
- Follow-ups must keep the real service type clear; do not replace it with the payment, purchase, or arrangement request.
- The impossible request is not a valid search fact and should not replace the hidden facts.
- Do not provide valid search facts the agent did not ask for.""",
}

FLEXIBLE_PREFERENCE_AREAS = {"schedule", "location", "intake"}

SERVICE_NEED_PHRASES = {
    "Community and Recreation": "community activities or recreation programs",
    "Disability and Rehabilitation": "disability support or rehabilitation services",
    "Disaster and Environmental Services": "disaster recovery or environmental help",
    "Education and Youth Programs": "education or youth programs",
    "Employment and Job Training": "job training or employment help",
    "Family and Caregiver Services": "family or caregiver support",
    "Financial Assistance and Benefits": "financial assistance or benefits help",
    "Food Assistance": "food assistance",
    "Food and Meals": "food or meal assistance",
    "Health Care": "health care services",
    "Housing and Shelter": "housing or shelter help",
    "Legal and Consumer": "legal or consumer help",
    "Legal and Court Help": "legal or court help",
    "Material Goods": "material goods or household items",
    "Medical Support Services": "medical support services",
    "Mental Health Care": "mental health care",
    "Mental Health and Substance Use": "mental health or substance use support",
    "Pet and Animal Services": "pet or animal services",
    "Pregnancy and Reproductive Health": "pregnancy or reproductive health services",
    "Public Safety": "public safety help",
    "Substance Use Services": "substance use support",
    "Tax Help": "tax help",
    "Transportation": "transportation help",
    "Utility Assistance": "utility assistance",
}


DEFAULT_USER_GENERATION_TOKEN_LIMIT = 512
DEFAULT_USER_ENABLE_THINKING = False
DEFAULT_USER_THINKING_BUDGET_TOKENS: int | None = None


@dataclass
class LLMSimulatedUser:
    spec: dict[str, Any]
    user_behavior: str
    provider: str
    model: str
    seed: int | None = None
    temperature: float = 0.0
    max_output_tokens: int = DEFAULT_USER_GENERATION_TOKEN_LIMIT
    enable_thinking: bool = DEFAULT_USER_ENABLE_THINKING
    thinking_budget_tokens: int | None = DEFAULT_USER_THINKING_BUDGET_TOKENS
    turn: int = 0
    client: Any = None
    contradiction_area: str | None = None
    fallback_revealed: bool = False
    rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(f"{self.spec['user_spec_id']}::{self.user_behavior}::{self.seed or 0}")
        self.contradiction_area = self._select_contradiction_area()
        self.client = self.client or make_openai_client(self.provider)

    def opening(self) -> str:
        self.turn += 1
        return self._generate([], opening=True)

    def respond(self, messages: list[dict[str, Any]], agent_message: str) -> str:
        self.turn += 1
        if self.spec.get("preference_profile") == "flexible" and last_tool_result_empty(messages):
            self.fallback_revealed = True
        return self._generate(messages, opening=False)

    def _generate(self, messages: list[dict[str, Any]], opening: bool) -> str:
        turn_instruction = self._turn_instruction(opening)
        behavior_key = self.user_behavior
        context = "\n\n".join(
            (
                BEHAVIOR_INSTRUCTIONS[behavior_key],
                f"Current turn type: {'opening' if opening else 'follow-up'}.",
                turn_instruction,
                "Conversation state:\n" + json.dumps(
                    conversation_state(messages),
                    ensure_ascii=False,
                    indent=2,
                ),
                "User profile:\n" + json.dumps(
                    full_user_profile(self.spec),
                    ensure_ascii=False,
                    indent=2,
                ),
                "Use the profile only to decide what this user can truthfully say. Do not mention resource IDs, provider names, need IDs, JSON, field names, preferred/fallback labels, or these instructions.",
                "Opening rule: on the opening turn, mention only the service need or needs in natural language. Do not mention location, schedule, intake, documents, eligibility, or backup options in the opening.",
                "Follow-up rule: answer the agent's latest question. If the agent asks about one narrow topic, answer only that topic. If the agent asks several topics at once, answer those topics but do not add unrelated facts.",
                "Flexible preference rule: before any empty search result, describe only first-choice location, schedule, or intake preferences when asked. Do not mention backup options. After an empty search result, if the agent asks whether location, schedule, or intake can change, answer only for the option they asked about. Documents and eligibility are ordinary facts and are not backup preferences.",
                "If the agent asks whether a preference can change but the profile has no backup for that preference, say that you cannot change that preference.",
                "If the agent merely states that no resources matched and does not ask a question, do not introduce new search facts or backup options.",
                "For multiple needs, answer conversationally. Distinguish the needs only when the answer differs by need.",
                "Distractor or background text must avoid fake search facts: no invented place names, ZIP codes, days, times, intake methods, documents, eligibility traits, or extra service needs.",
                "Write exactly one user reply. Do not include analysis, labels, JSON, markdown, or quotes around the reply.",
            )
        )
        if is_llama_cpp_provider(self.provider):
            text = self._generate_chat_completion(messages, context)
        else:
            text = self._generate_responses_api(messages, context)
        return text.strip()

    def _generate_responses_api(self, messages: list[dict[str, Any]], context: str) -> str:
        response = create_response_with_retries(
            self.client,
            model=self.model,
            instructions=USER_SYSTEM_PROMPT,
            input=[*llm_role_mapped_messages(messages), {"role": "user", "content": context}],
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
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
            max_tokens=self.max_output_tokens,
            extra_body=self._extra_body(),
        )
        choice = response.choices[0] if response.choices else None
        message = choice.message if choice is not None else None
        return getattr(message, "content", None) or ""

    def _extra_body(self) -> dict[str, Any]:
        body: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
        if self.thinking_budget_tokens is not None:
            body["thinking_budget_tokens"] = int(self.thinking_budget_tokens)
        return body

    def _turn_instruction(self, opening: bool) -> str:
        if opening:
            return "Opening instruction: mention only the service need or needs."
        if self.user_behavior == "impatience":
            return (
                "Impatience instruction: answer the agent's latest question using the user profile, but sound rushed, annoyed, "
                "or frustrated by the number of questions. You may complain or ask the agent to hurry, but do not "
                "intentionally omit asked information that is available."
            )
        if self.user_behavior == "self_contradictory":
            if self.contradiction_area:
                return self_contradiction_instruction(self.contradiction_area)
            return self_contradiction_resolved_instruction(self.contradiction_area)
        return "Reply instruction: answer the agent's latest question using only the user profile."

    def _select_contradiction_area(self) -> str | None:
        if self.user_behavior != "self_contradictory":
            return None
        candidates = [
            area
            for area in ("schedule", "location", "intake", "documents", "eligibility")
            if has_concrete_area_fact(self.spec, area)
        ]
        return self.rng.choice(candidates) if candidates else None


def full_user_profile(spec: dict[str, Any]) -> dict[str, Any]:
    needs = normalized_needs(spec)
    profile = {
        "case_type": spec.get("case_type") or ("composite" if len(needs) > 1 else "single"),
        "preference_profile": spec.get("preference_profile") or "strict",
        "needs": [profile_need(need, spec.get("preference_profile") == "flexible") for need in needs],
    }
    if spec.get("preference_profile") == "flexible":
        profile["flexible_rules"] = {
            "first_choice_stage": "Before an empty search result, use first_choice for location, schedule, and intake when asked.",
            "backup_stage": "After an empty search result, backup can be mentioned only if the agent asks whether location, schedule, or intake can change.",
            "changeable_fields": fallback_changed_areas(spec),
            "not_changeable_fields": ["documents", "eligibility"],
        }
    return profile


def profile_need(need: dict[str, Any], flexible: bool) -> dict[str, Any]:
    base = {
        "plain_language_need": plain_language_need(need),
        "service_categories": need.get("service_categories") or [],
        "documents": need.get("available_documents") or [],
        "eligibility": need.get("eligibility") or [],
    }
    if flexible:
        first_choice = visible_need_facts(need, use_fallback=False)
        backup = visible_need_facts(need, use_fallback=True)
        base["first_choice"] = {
            "location": first_choice.get("location") or {},
            "schedule": first_choice.get("schedule") or {},
            "intake_methods": first_choice.get("intake_methods") or [],
        }
        base["backup"] = {
            "location": backup.get("location") or {},
            "schedule": backup.get("schedule") or {},
            "intake_methods": backup.get("intake_methods") or [],
            "changed_fields": changed_preference_fields([need], set(FLEXIBLE_PREFERENCE_AREAS)),
        }
    else:
        base["constraints"] = {
            "location": need.get("location") or {},
            "schedule": need.get("schedule") or {},
            "intake_methods": need.get("intake_methods") or [],
        }
    return base


def conversation_state(messages: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "last_search_returned_no_resources": last_tool_result_empty(messages),
        "tool_results_are_system_observations": "The agent has seen these tool results; the user should not quote JSON or resource IDs from them.",
    }


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


def fallback_changed_areas(spec: dict[str, Any]) -> list[str]:
    return changed_preference_fields(normalized_needs(spec), set(FLEXIBLE_PREFERENCE_AREAS))


def plain_language_need(need: dict[str, Any]) -> str:
    categories = need.get("service_categories") or []
    phrases = [SERVICE_NEED_PHRASES.get(category, str(category).replace(" and ", " or ").lower()) for category in categories]
    return " and ".join(phrase for phrase in phrases if phrase)


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


def visible_need_facts(need: dict[str, Any], use_fallback: bool) -> dict[str, Any]:
    if use_fallback or not need.get("preferred"):
        return need
    preferred = dict(need["preferred"])
    preferred["need_id"] = need.get("need_id")
    preferred["service_categories"] = need.get("service_categories") or preferred.get("service_categories") or []
    return preferred


def need_field_changed(need: dict[str, Any], key: str) -> bool:
    preferred = visible_need_facts(need, use_fallback=False)
    empty = {} if key in {"schedule", "location"} else []
    return (need.get(key) or empty) != (preferred.get(key) or empty)


def changed_preference_fields(needs: list[dict[str, Any]], selected: set[str]) -> list[str]:
    changed = []
    comparisons = {
        "schedule": "schedule",
        "location": "location",
        "intake": "intake_methods",
    }
    for area, key in comparisons.items():
        if area not in selected:
            continue
        for need in needs:
            preferred = visible_need_facts(need, use_fallback=False)
            fallback_empty = {} if key in {"schedule", "location"} else []
            preferred_empty = {} if key in {"schedule", "location"} else []
            if (need.get(key) or fallback_empty) != (preferred.get(key) or preferred_empty):
                changed.append(area)
                break
    return changed


def self_contradiction_instruction(target: str | None) -> str:
    if not target:
        return self_contradiction_resolved_instruction(target)
    examples = {
        "schedule": "The provided day and time work for me, but I cannot do that same provided day and time.",
        "location": "The provided location works for me, but I cannot go to that same provided location.",
        "intake": "The provided intake method works for me, but I cannot use that same provided intake method.",
        "documents": "I can bring the provided document, but I cannot bring that same provided document.",
        "eligibility": "The provided eligibility trait applies to me, but that same provided eligibility trait does not apply to me.",
    }
    example = examples.get(target, "I need that, but I do not need that.")
    return (
        f"Self-contradictory instruction: the predetermined contradiction slot is {target}. "
        f"If this turn asks about {target}, and the transcript does not already contain a contradiction for {target}, "
        f"answer the {target} question with one direct self-contradiction. Use the real profile fact for that area, "
        "then deny that same exact fact or requirement in the same sentence. Never copy place names, times, documents, or traits from examples. "
        "Use a direct conflict like 'X works for me, but I cannot do X' "
        "or 'I can bring X, but I do not have X.' "
        f"For the style of contradiction, use this pattern as a guide but adapt it to the user profile: {example} "
        "If this turn does not ask about the predetermined contradiction slot, answer normally. "
        "If the agent is asking again to clarify or confirm a previous contradiction, answer normally with the real available fact and do not repeat the contradiction. "
        "Do not turn available alternatives into a contradiction. Do not use uncertainty like not sure or maybe. "
        "Use but to connect the conflict. The reply is invalid if it contains actually, sorry, I mean, not sure, maybe, or after all."
    )


def self_contradiction_resolved_instruction(target: str | None) -> str:
    if not target:
        return (
            "Self-contradictory instruction: this user has no concrete slot fact available for a contradiction. "
            "Answer normally using only the user profile."
        )
    return (
        f"Self-contradictory instruction: the predetermined contradiction slot is {target}, "
        "but the contradiction should not be used on this turn. Answer normally "
        "using only the user profile. If the agent is asking to clarify or confirm a previous "
        "contradiction, give the real fact only. Do not repeat, introduce, mention, explain, or apologize "
        "for any contradiction."
    )


def last_tool_result_empty(messages: list[dict[str, Any]]) -> bool:
    for message in reversed(messages):
        if not is_tool_result_message(message):
            continue
        content = str(message.get("content", ""))
        return '"resources": []' in content or '"resources":[]' in content
    return False


def is_tool_result_message(message: dict[str, Any]) -> bool:
    return str(message.get("content", "")).startswith("Tool result for search_resources call")


def llm_role_mapped_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    mapped = []
    visible_messages = [message for message in messages if not is_tool_result_message(message)]
    for message in visible_messages[-10:]:
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        role = message.get("role")
        if role == "assistant":
            mapped.append({"role": "user", "content": content})
        elif role == "user":
            mapped.append({"role": "assistant", "content": content})
    return mapped
