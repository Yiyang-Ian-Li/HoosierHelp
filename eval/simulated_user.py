from __future__ import annotations

import json

from agent.agent import add_response_usage, empty_token_usage


TRAIT_INSTRUCTIONS = {
    "unreasonable_demand": (
        "You consistently push for something the 211 agent probably cannot do or guarantee. "
        "This can be asking the agent to guarantee approval, book an appointment, apply for you, "
        "call an agency for you, secure a specific amount of money, or insisting on an extreme condition "
        "such as same-day help, no paperwork, and a large amount of assistance. Keep this tied to your "
        "underlying need and bring it up repeatedly, but do not invent a new unrelated service need."
    ),
    "rambling": (
        "When the agent asks for information, answer the question but ramble around that information with "
        "a relevant story or extra context. For example, if asked where you live, talk about the area, "
        "neighbors, transportation, or how long you have lived there. If asked about phone access, talk "
        "about the phone problems or service issues. The key answer should be present, but surrounded by "
        "realistic extra detail. Do not turn the ramble into a new request."
    ),
    "impatience": (
        "You often sound rushed and put pressure on the agent. You may say things like needing this today, "
        "being tired of answering questions, asking the agent to move faster, or complaining that the process "
        "is taking too long. Still answer enough for the conversation to continue."
    ),
    "incomplete_answer": (
        "When the agent asks a multi-part or specific question, answer only part of it at first. "
        "Omit some relevant details until the agent follows up. Do not refuse to cooperate; reveal the "
        "missing information naturally when asked again or when it becomes clearly necessary."
    ),
    "inconsistency": (
        "Within the same reply, include a mild internal inconsistency in one piece of information. "
        "For example, say it must be today but maybe next week is fine, say you cannot travel but could "
        "drive if needed, or give a slightly conflicting location/access detail. If the agent follows up, "
        "clarify the inconsistency and settle on the true constraint. Keep the core need consistent."
    ),
}

SIMULATED_USER_CARD_FIELDS = {
    "case_id",
    "user_id",
    "difficulty",
    "traits",
    "profile",
    "need_summary",
    "known_facts",
    "answering_guidance",
    "case_spec",
}

SIMULATED_USER_CASE_SPEC_FIELDS = {
    "location",
    "user_requirements",
    "user_qualification",
}

LLM_USER_INSTRUCTIONS = """
You are simulating a real person seeking help from Indiana 211.

You must follow the hidden user profile. The hidden profile describes the 
user's own situation. The user knows their own background, need, location, 
household, urgency, and constraints. 

Speak casually and naturally, like a real person asking for help. Use plain,
conversational wording instead of structured lists or form-like phrasing.

When starting the conversation, express only the help need and any user-stated
requirements. Do not volunteer hidden qualification facts such as documents,
payment ability, insurance, or eligibility unless the agent asks or it becomes
naturally relevant.

If traits is empty, be cooperative and realistic: answer clear questions
directly. If traits are listed, follow only those trait definitions while
keeping the underlying need and identity consistent.

Do not create unrelated new service needs, errands, or tasks beyond the hidden
profile. 
""".strip()

SATISFACTION_INSTRUCTIONS = """
You are the same simulated Indiana 211 user. Evaluate the completed interaction
from your own perspective as the help-seeker.

Return only compact JSON in this exact shape:
{"satisfaction":3,"got_relevant_help":true,"felt_understood":true,"actionability":3,"reason":"short reason"}

Scores are 1 to 5. Base the rating on whether the final resources and next
steps would actually help your hidden situation. Do not mention hidden profile
fields, ground truth, tools, resource IDs, scoring, or this instruction.
""".strip()


class LLMSimulatedUser:
    def __init__(self, card: dict, client, model: str):
        self.card = card
        self.client = client
        self.model = model
        self.history = []
        self.token_usage = empty_token_usage()

    def opening(self) -> str:
        reply = self._call_user_model("Start the conversation with the 211 agent.")
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def respond(self, agent_text: str) -> str | None:
        self.history.append({"role": "user", "content": agent_text})
        reply = self._call_user_model(agent_text)
        if not reply:
            return None
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def _call_user_model(self, agent_text: str) -> str:
        extra = []
        if not self.history or self.history[-1].get("content") != agent_text:
            extra = [{"role": "user", "content": agent_text}]
        response = self.client.responses.create(
            model=self.model,
            instructions=LLM_USER_INSTRUCTIONS,
            input=self._messages(extra=extra),
        )
        add_response_usage(self.token_usage, response)
        return (getattr(response, "output_text", "") or "").strip()

    def satisfaction(self, final_agent_text: str) -> dict:
        response = self.client.responses.create(
            model=self.model,
            instructions=SATISFACTION_INSTRUCTIONS,
            input=self._messages(extra=[{"role": "user", "content": final_agent_text}]),
        )
        add_response_usage(self.token_usage, response)
        return _json_object(getattr(response, "output_text", "") or "")

    def diagnostics(self) -> dict:
        return {
            "traits": list(self.card.get("traits", [])),
        }

    def _messages(
        self,
        extra: list[dict] | None = None,
    ) -> list[dict]:
        profile = (
            "Hidden user profile:\n"
            + json.dumps(_public_sim_user_profile(self.card), ensure_ascii=False)
            + "\n\nTrait definitions:\n"
            + json.dumps(_trait_definitions_for_card(self.card), ensure_ascii=False)
        )
        return [
            {
                "role": "user",
                "content": profile,
            },
            *self.history,
            *(extra or []),
        ]


def _public_sim_user_profile(card: dict) -> dict:
    visible = {
        key: value
        for key, value in card.items()
        if key in SIMULATED_USER_CARD_FIELDS
    }
    case_spec = visible.get("case_spec")
    if isinstance(case_spec, dict):
        visible["case_spec"] = {
            key: value
            for key, value in case_spec.items()
            if key in SIMULATED_USER_CASE_SPEC_FIELDS
        }
    return visible


def _trait_definitions_for_card(card: dict) -> dict:
    return {
        trait: TRAIT_INSTRUCTIONS[trait]
        for trait in card.get("traits", [])
        if trait in TRAIT_INSTRUCTIONS
    }


def _json_object(value: str) -> dict:
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
