import tempfile
import unittest
from pathlib import Path

from hsds_agent.agent import ResourceAgent
from hsds_agent.database import connect, initialize, seed_from_json


class FakeLLM:
    def __init__(self, arguments):
        self.arguments = arguments
        self.used_tools = False

    def answer_with_tools(self, question, tools, execute_tool):
        self.used_tools = True
        result = execute_tool("search_services", self.arguments)
        if "needs_follow_up" in result and "true" in result.lower():
            return "What city or ZIP code should I search near?"
        return f"LLM answer used tool result: {result}"


class ResourceAgentTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        db_path = Path(self.temp_dir.name) / "test.sqlite"
        self.conn = connect(db_path)
        initialize(self.conn)
        seed_from_json(self.conn)
        self.agent = ResourceAgent(
            self.conn,
            FakeLLM(
                {
                    "query": "free food",
                    "location": "bloomington",
                    "categories": ["food"],
                    "limit": 5,
                }
            ),
        )

    def tearDown(self):
        self.conn.close()
        self.temp_dir.cleanup()

    def test_food_question_returns_food_resource(self):
        answer = self.agent.answer("I need free food near Bloomington")

        self.assertFalse(answer.needs_follow_up)
        self.assertIn("Community Food Pantry", answer.answer)
        self.assertTrue(answer.candidates)

    def test_missing_location_asks_follow_up(self):
        agent = ResourceAgent(
            self.conn,
            FakeLLM({"query": "rent help", "location": None, "categories": ["housing"]}),
        )
        answer = agent.answer("I need help with rent")

        self.assertTrue(answer.needs_follow_up)
        self.assertIn("city or ZIP code", answer.answer)

    def test_spanish_senior_transportation(self):
        agent = ResourceAgent(
            self.conn,
            FakeLLM(
                {
                    "query": "senior transportation",
                    "location": "47401",
                    "categories": ["transportation"],
                    "languages": ["Spanish"],
                    "eligibility": ["senior"],
                    "limit": 5,
                }
            ),
        )
        answer = agent.answer(
            "Where can a Spanish-speaking senior get transportation near 47401?"
        )

        self.assertFalse(answer.needs_follow_up)
        self.assertIn("Senior Medical Rides", answer.answer)

    def test_llm_uses_search_services_tool(self):
        llm = FakeLLM(
            {
                "query": "tenant eviction legal help",
                "location": "bloomington",
                "categories": ["legal"],
                "limit": 1,
            }
        )
        answer = ResourceAgent(self.conn, llm).answer("Can I get tenant help?")

        self.assertTrue(llm.used_tools)
        self.assertIn("Eviction Prevention Legal Clinic", answer.answer)
        self.assertEqual(answer.tool_calls[0]["arguments"]["categories"], ["legal"])


if __name__ == "__main__":
    unittest.main()
