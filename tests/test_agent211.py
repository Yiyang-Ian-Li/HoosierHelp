import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agent211 import (
    Agent211,
    SearchRequest,
    evaluate_cases,
    load_benchmark_cases,
    load_resource_index,
)


class Agent211Test(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.index_path = Path(self.temp_dir.name) / "resources.jsonl"
        rows = [
            {
                "resource_id": "food-1",
                "service_name": "Food Pantry",
                "agency_name": "Marion Food Help",
                "site_name": "Marion Food Help",
                "benchmark_categories": ["Food"],
                "source_subcategories": ["Food"],
                "curated_subcategories": ["Food"],
                "service_area": ["MARION"],
                "location": {
                    "address_1": "1 Main St",
                    "address_2": "",
                    "city": "Indianapolis",
                    "state": "IN",
                    "zipcode": "46204",
                    "latitude": "",
                    "longitude": "",
                },
                "contact": {"phone": "317-555-0000", "website": "", "email": ""},
                "eligibility": "Open to Marion County residents",
                "application_process": "Call for pantry hours.",
                "fees": "Free",
                "documents_required": "Photo ID",
                "search_text": "food pantry groceries Marion Indianapolis",
            },
            {
                "resource_id": "legal-1",
                "service_name": "Legal Aid",
                "agency_name": "Tenant Legal Clinic",
                "site_name": "Tenant Legal Clinic",
                "benchmark_categories": ["Legal, Employment & Consumer Help"],
                "source_subcategories": ["Legal Services"],
                "curated_subcategories": ["Legal Services"],
                "service_area": ["MARION"],
                "location": {
                    "address_1": "2 Main St",
                    "address_2": "",
                    "city": "Indianapolis",
                    "state": "IN",
                    "zipcode": "46204",
                    "latitude": "",
                    "longitude": "",
                },
                "contact": {"phone": "317-555-1111", "website": "", "email": ""},
                "eligibility": "Tenants with civil legal issues",
                "application_process": "Call for intake.",
                "fees": "Free",
                "documents_required": "Lease documents",
                "search_text": "legal aid tenant eviction lawyer Marion Indianapolis",
            },
        ]
        self.index_path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )
        self.index = load_resource_index(self.index_path)
        self.agent = Agent211(self.index)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_agent_retrieves_food_resource(self):
        response = self.agent.ask("I need a food pantry in Marion County")

        self.assertEqual(response.results[0].resource.resource_id, "food-1")
        self.assertEqual(response.tool_calls[0]["tool"], "search_resources")

    def test_agent_accepts_planner(self):
        response = Agent211(self.index).ask(
            "My landlord is evicting me",
            request=SearchRequest(
                text_query="My landlord is evicting me",
                curated_subcategories=("Legal Services",),
                limit=3,
            ),
        )

        self.assertEqual(response.results[0].resource.resource_id, "legal-1")

    def test_agent_can_rerank(self):
        def reranker(query, results, limit):
            return sorted(
                results,
                key=lambda result: result.resource.resource_id == "legal-1",
                reverse=True,
            )[:limit]

        response = Agent211(self.index, reranker=reranker).ask(
            "I need help",
            request=SearchRequest(text_query="I need help", counties=("MARION",), limit=10),
            limit=2,
        )

        self.assertEqual(response.results[0].resource.resource_id, "legal-1")
        self.assertEqual(response.tool_calls[1]["tool"], "rerank_resources")

    def test_openai_tool_calling_path_executes_search_tool(self):
        client = FakeOpenAIClient(
            {
                "query": "food pantry",
                "counties": ["MARION"],
                "curated_subcategories": ["Food"],
                "limit": 3,
            }
        )
        response = Agent211(
            self.index,
            client=client,
            model="test-model",
            use_openai_tools=True,
        ).ask("I need a food pantry in Marion County")

        self.assertTrue(client.first_call_tools)
        self.assertEqual(response.results[0].resource.resource_id, "food-1")
        self.assertEqual(response.tool_calls[0]["tool"], "search_resources")

    def test_evaluator_scores_gt_hit(self):
        benchmark_path = Path(self.temp_dir.name) / "benchmark.jsonl"
        benchmark_path.write_text(
            json.dumps(
                {
                    "query_id": "q1",
                    "user_query": "I need a food pantry in Marion County",
                    "primary_gt_resource_ids": ["food-1"],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        cases = load_benchmark_cases(benchmark_path)
        summary = evaluate_cases(self.agent, cases, limit=3)

        self.assertEqual(summary.case_count, 1)
        self.assertEqual(summary.recall_at_1, 1.0)
        self.assertEqual(summary.mrr, 1.0)


if __name__ == "__main__":
    unittest.main()


class FakeOpenAIClient:
    def __init__(self, tool_args):
        self.tool_args = tool_args
        self.calls = 0
        self.first_call_tools = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            self.first_call_tools = kwargs.get("tools")
            message = SimpleNamespace(
                content=None,
                tool_calls=[
                    SimpleNamespace(
                        id="call_1",
                        function=SimpleNamespace(
                            name="search_resources",
                            arguments=json.dumps(self.tool_args),
                        ),
                    )
                ],
                model_dump=lambda exclude_none=True: {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "search_resources",
                                "arguments": json.dumps(self.tool_args),
                            },
                        }
                    ],
                },
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])
        message = SimpleNamespace(content="Final answer from tool result.")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])
