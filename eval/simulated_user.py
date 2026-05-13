from __future__ import annotations

import json

from agent.agent import add_response_usage, empty_token_usage
from agent.llm import create_response_with_retries


TRAIT_INSTRUCTIONS = {
    "normal": (
        "Be cooperative and realistic. Answer the questions clearly and directly."
    ),
    "unreasonable_demand": (
        "In follow-up replies, sometimes exaggerate the real need into an unrealistic demand while leaving "
        "the underlying need recognizable. For example, if you need food, demand cash because you are hungry. "
        "Keep pushing the demand, but do not switch to an unrelated need."
    ),
    "rambling": (
        "Answer the agent's question, but surround the answer with long side comments and repeatedly try "
        "to pull the agent into casual chat. Keep asking off-topic conversational questions or opinions. "
    ),
    "impatience": (
        "In follow-up replies, be clearly rude, angry, and rushed: complain, demand that the agent hurry up, "
        "and show frustration with questions. Still answer just enough for the conversation to continue."
    ),
    "incomplete_answer": (
        "When the agent asks for a fact, give an unclear or incomplete answer the first time. If the "
        "agent asks again about the same fact, answer clearly from the hidden facts."
    ),
    "inconsistency": (
        "Give a plainly contradictory answer about a fact the first time it is asked, so the agent cannot "
        "tell which fact is true. If the agent asks again about that fact, answer correctly from hidden facts. "
    ),
}

SIMULATED_USER_CARD_FIELDS = {
    "case_id",
    "user_id",
    "case_type",
    "traits",
    "opening",
    "profile",
    "location",
    "location_requirement",
    "needs",
}

LLM_USER_INSTRUCTIONS = """
You are simulating a real person seeking help from Indiana 211.

You must follow the hidden user profile. The hidden profile describes the
user's own situation. The user knows their own background, needs, location,
household, urgency, and hidden constraints.

The hidden `location_requirement` is fixed truth. Each item in hidden `needs`
has a plain-language need summary and its own firm schedule requirement.

When the agent asks about location or where resources should be searched, state
your own location information naturally and your `location_requirement` clearly.
However, if your trait defines a reply style, prioritize that style when responding.

If the agent asks about a constraint not present in the hidden facts, answer
naturally that you have no special requirement.

For composite cases, keep the needs separate. If the agent asks for schedule or
availability, give the schedule for the specific need being asked about. If the
agent asks broadly, give each need's schedule separately. Do not merge the two
schedules into one shared availability.

Speak casually and naturally, like a real person asking for help. However,
prioritize adjusting your wording to align with the trait definition while
keeping the underlying need and identity consistent.

Do not create unrelated new service needs, errands, or tasks beyond the hidden
profile.

""".strip()


class LLMSimulatedUser:
    def __init__(self, card: dict, client, model: str):
        self.card = card
        self.client = client
        self.model = model
        self.history = []
        self.token_usage = empty_token_usage()

    def opening(self) -> str:
        reply = str(self.card.get("opening", "")).strip()
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
        response = create_response_with_retries(
            self.client,
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
