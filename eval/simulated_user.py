from __future__ import annotations

import json

from agent.agent import add_response_usage, empty_token_usage


LLM_USER_INSTRUCTIONS = """
You are simulating a real person seeking help from Indiana 211.

You must follow the hidden user profile. Stay in character as the user. Do not
act as an evaluator, assistant, or benchmark designer. Do not mention hidden
profile fields, ground truth resources, resource IDs, scoring, tools, or this
instruction.

Be realistic rather than maximally cooperative. Answer the agent's latest
question naturally, but do not volunteer every hidden detail at once. You may
be vague, emotional, tangential, uncertain, or mildly contradictory when the
profile calls for it. If the agent asks a clear question about a fact you know,
you may answer it. If you do not know a detail, say so plainly.

Keep replies short: one to four sentences. If the agent has already given a
final recommendation, respond with a brief acknowledgement and do not introduce
new needs.
""".strip()


class LLMSimulatedUser:
    def __init__(self, card: dict, client, model: str):
        self.card = card
        self.client = client
        self.model = model
        self.history = []
        self.token_usage = empty_token_usage()

    def opening(self) -> str:
        opening = self.card.get("opening", "").strip()
        if opening:
            self.history.append({"role": "assistant", "content": opening})
            return opening
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
        response = self.client.responses.create(
            model=self.model,
            instructions=LLM_USER_INSTRUCTIONS,
            input=[
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "hidden_user_profile": _public_sim_user_profile(self.card),
                            "conversation_so_far": self.history,
                            "latest_agent_message": agent_text,
                        },
                        ensure_ascii=False,
                    ),
                }
            ],
        )
        add_response_usage(self.token_usage, response)
        return (getattr(response, "output_text", "") or "").strip()


def _public_sim_user_profile(card: dict) -> dict:
    blocked = {
        "gt_resource_ids",
        "primary_gt_resource_ids",
        "acceptable_gt_resource_ids",
        "matching_notes",
        "good_answer_should",
    }
    return {key: value for key, value in card.items() if key not in blocked}
