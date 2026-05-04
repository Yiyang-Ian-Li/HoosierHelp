# 211-Agent

Prototype agent framework for answering questions about social support resources
from an HSDS-style database.

This first version is intentionally local and lightweight:

- SQLite database seeded from synthetic HSDS-style JSON
- constrained `search_services` tool for resource retrieval
- OpenAI-style LLM tool calling through OpenRouter or OpenAI
- CLI for quick experiments
- unit tests using Python's standard library

## Quick Start

Put your OpenRouter key in `.env`:

```bash
OPENROUTER_API_KEY="..."
HSDS_AGENT_MODEL="openai/gpt-4.1-mini"
```

```bash
uv run python -m hsds_agent.cli seed
uv run python -m hsds_agent.cli ask "I need free food near Bloomington this week"
uv run python -m hsds_agent.cli ask "Where can a Spanish-speaking senior get transportation near 47401?"
uv run python -m unittest discover -s tests
```

By default, the SQLite database is created at `./data/hsds_agent.sqlite`.

## LLM Providers

The agent uses an OpenAI-style chat completion API for every `ask` request.
OpenRouter is the default provider, using `openai/gpt-4.1-mini` unless
`HSDS_AGENT_MODEL` is set.

OpenAI:

```bash
export OPENAI_API_KEY="..."
export HSDS_AGENT_MODEL="gpt-4.1-mini"
uv run python -m hsds_agent.cli ask "Find free legal help for eviction near Bloomington" --provider openai
```

OpenRouter:

```bash
export OPENROUTER_API_KEY="..."
export HSDS_AGENT_MODEL="openai/gpt-4.1-mini"
uv run python -m hsds_agent.cli ask "Find food help near 47401 for a Spanish speaker"
```

Provider configuration:

- `OPENAI_API_KEY`, `OPENAI_BASE_URL`
- `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`
- `HSDS_AGENT_MODEL`
- optional OpenRouter headers: `OPENROUTER_HTTP_REFERER`,
  `OPENROUTER_APP_TITLE`

## Pipeline

```text
User question
  -> send question + search_services tool schema to OpenAI-compatible LLM
  -> LLM emits a search_services tool call with structured arguments
  -> Python validates/normalizes tool arguments
  -> Python queries/ranks HSDS-style records from SQLite
  -> Python returns structured tool result to the LLM
  -> LLM writes final answer using only the tool result
  -> LLM asks a follow-up if location or other required constraints are missing
```

## How The LLM Uses Tools

The LLM never writes SQL and does not inspect the database schema directly.
Instead, [agent.py](./hsds_agent/agent.py) exposes one OpenAI-style function
tool:

```text
search_services({
  query,
  location,
  radius_miles,
  categories,
  languages,
  eligibility,
  open_now,
  limit
})
```

The model decides the arguments. The Python agent executes the tool by building
a `SearchRequest`, querying SQLite through [tools.py](./hsds_agent/tools.py),
and returning JSON records with names, descriptions, location, phone, website,
hours, eligibility, language, distance, and source fields. The model then writes
the final user-facing answer from those records.

## Experiment Data

The current fixture is [data/sample_hsds.json](./data/sample_hsds.json). It is
synthetic HSDS-style data, hand-built to exercise common 211/social-support
queries without using real referral data.

It includes:

- organizations
- services
- locations
- service-at-location relationships
- phones
- schedules

The services cover food, shelter, childcare, transportation, legal aid, mental
health, and benefits navigation. The records intentionally include useful test
constraints such as Spanish language support, senior eligibility, free/sliding
scale fees, multiple ZIP codes, schedules, and distance ranking around
Bloomington, Indiana.

This data is for experiments only. Real deployments should ingest current HSDS
exports from a publisher, keep source/freshness metadata, and make confirmation
warnings visible in every answer.

## Project Layout

```text
data/sample_hsds.json          Synthetic HSDS-style experiment data
hsds_agent/database.py         SQLite schema, seeding, query helpers
hsds_agent/llm.py              OpenAI-compatible LLM wrapper
hsds_agent/tools.py            Agent-callable tools
hsds_agent/agent.py            Workflow orchestration and response generation
hsds_agent/cli.py              Command-line interface
tests/test_agent.py            Regression tests
```

## Notes

The sample data is synthetic. It is useful for experiments and evaluation
fixtures, but not for real referrals. Real deployments should load current data
from an HSDS publisher, preserve source metadata, and show freshness/confirmation
warnings in the final answer.
