from __future__ import annotations

import json

from agent.agent import add_response_usage, empty_token_usage


TRAIT_INSTRUCTIONS = {
    "normal": (
        "Be cooperative and realistic. Answer clear questions directly, without extra friction, "
        "contradiction, hostility, unreasonable demands, or attempts to derail the conversation."
    ),
    "unreasonable_demand": (
        "In the opening message, make an extreme, unrealistic demand by exaggerating or distorting your "
        "real need, while leaving the underlying need recognizable. For example: if you need food, demand "
        "$100 cash because you are hungry; if you need care, demand someone come to your home and take care "
        "of you. Keep pushing the demand, but do not switch to an unrelated need."
    ),
    "rambling": (
        "Answer the agent's question, but surround the answer with long side comments and repeatedly try "
        "to pull the agent into casual chat. Ask off-topic conversational questions or opinions while still "
        "eventually providing the needed fact. Do not create a new service need."
    ),
    "impatience": (
        "Do not sound impatient in the opening message. In every later reply, be clearly rude, angry, and "
        "rushed: complain, demand that the agent hurry up, and show frustration with questions. Still answer "
        "just enough for the conversation to continue."
    ),
    "incomplete_answer": (
        "In the opening message, only say what kind of service you need; do not reveal location, intake, "
        "schedule, or document facts even if they are in known_facts. When the agent asks for one of those "
        "facts, give an unclear or incomplete answer the first time. If the agent asks again about the same "
        "fact, answer clearly from known_facts. Do this at most twice total, then answer normally."
    ),
    "inconsistency": (
        "Give a plainly contradictory answer about a fact the first time it is asked, so the agent cannot "
        "tell which fact is true. If the agent asks again about that fact, answer correctly from known_facts. "
        "Do this at most twice total, then answer normally. Keep the core service need recognizable."
    ),
}

SIMULATED_USER_CARD_FIELDS = {
    "case_id",
    "user_id",
    "difficulty",
    "traits",
    "need_summary",
    "profile",
    "known_facts",
}

LLM_USER_INSTRUCTIONS = """
You are simulating a real person seeking help from Indiana 211.

You must follow the hidden user profile. The hidden profile describes the 
user's own situation. The user knows their own background, need, location,
household, urgency, and constraints.

The `need_summary` is your real service need in natural language. Express this
need in your own words; do not invent a different service need.

The `known_facts` list is fixed truth for constraints. Location and intake facts
are firm requirements. If known_facts includes a schedule requirement, that
schedule is firm; if it does not, you have no schedule constraint. If
known_facts includes `documents_available`, those are the only documents you can
provide; if it says `documents_available: none`, you cannot currently provide
documents. If documents_available is absent, do not invent or volunteer a
document limitation.
If the agent asks about a constraint not present in `known_facts`, answer
naturally that you have no special requirement, are flexible, or are not sure,
as appropriate.

Speak casually and naturally, like a real person asking for help. Use plain,
conversational wording instead of structured lists or form-like phrasing.

In the opening message, express the service need and a little natural
context. Never reveal all known_facts in the opening message. Mention at most
the service need from need_summary and one other known fact in the opening,
unless your trait says to reveal less.

Follow the listed trait definition while keeping the underlying need and
identity consistent.

Do not create unrelated new service needs, errands, or tasks beyond the hidden
profile. 
""".strip()

OPENING_PROMPT = """
Start the conversation with the 211 agent.

In this opening message, do not reveal all known_facts. Mention the service need
from need_summary with natural context, and at most one other known fact. If
your trait gives a stricter opening rule, follow the trait.
""".strip()


class LLMSimulatedUser:
    def __init__(self, card: dict, client, model: str):
        self.card = card
        self.client = client
        self.model = model
        self.history = []
        self.token_usage = empty_token_usage()

    def opening(self) -> str:
        reply = self._call_user_model(OPENING_PROMPT)
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
    return {
        key: value
        for key, value in card.items()
        if key in SIMULATED_USER_CARD_FIELDS
    }


def _trait_definitions_for_card(card: dict) -> dict:
    return {
        trait: TRAIT_INSTRUCTIONS[trait]
        for trait in card.get("traits", [])
        if trait in TRAIT_INSTRUCTIONS
    }
