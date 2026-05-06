# 211-Agent

Benchmark-oriented prototype for studying whether an LLM/tool-calling agent can
find appropriate human-service resources from a prepared 211-style resource
index.

The current code path is intentionally simple and reproducible:

```text
user query
  -> OpenAI tool interface exposes search_resources
  -> model calls search_resources with county/city/need filters
  -> local tool filters and ranks records from resource_index JSONL
  -> tool result is returned to the model
  -> optional LLM reranker reorders retrieved candidates
  -> agent returns ranked resources
  -> evaluator compares retrieved resource_ids with benchmark GT
```

The Python package is named `agent211` because Python imports cannot start with
a digit. The CLI command is `211agent`.

The search tool is intentionally isolated in:

```text
agent211/tool.py
```

That file contains the model-facing tool description, OpenAI tool schema,
argument parsing, and filtering. The tool behaves like a website filter form:
non-empty structured fields filter results. `text_query` is accepted for future
retriever experiments but is currently ignored by this filter-only tool.

## Data

The agent expects a benchmark-ready resource index:

```text
data/indiana211/benchmark_curated/resource_index_curated.jsonl
```

Each JSONL row is one resource with:

- `resource_id`
- service, agency, and site names
- benchmark categories and source subcategories
- service area counties
- address and contact information
- eligibility, application process, fees, documents
- `search_text`

The curated cleaning notebook is:

```text
notebooks/indiana211_benchmark_cleaning.ipynb
```

## Ask One Query

By default the CLI uses the OpenAI tool-calling interface. Configure one of:

```bash
export OPENROUTER_API_KEY="..."
# or
export OPENAI_API_KEY="..."
```

`.env` is loaded automatically from the repo root. To override the model:

```bash
AGENT211_MODEL="openai/gpt-4.1-mini"
```

```bash
uv run 211agent ask "I need a food pantry in Marion County"
```

Show raw ranked results:

```bash
uv run 211agent ask "I need help with utility bills in Lake County" --json
```

Use the no-network heuristic baseline only for debugging:

```bash
uv run 211agent ask "I need help with utility bills in Lake County" --planner heuristic --json
```

Enable optional LLM reranking:

```bash
uv run 211agent ask "I need legal help with an eviction near Indianapolis" --rerank --json
```

## Evaluate A Benchmark

Benchmark JSONL format:

```json
{
  "query_id": "q001",
  "user_query": "I need a food pantry in Marion County",
  "primary_gt_resource_ids": ["in211-..."],
  "acceptable_gt_resource_ids": ["in211-..."],
  "hard_negative_resource_ids": ["in211-..."],
  "difficulty": "A"
}
```

Run evaluation:

```bash
uv run 211agent eval data/benchmarks/stage1_queries.jsonl --limit 10
```

With optional reranking:

```bash
uv run 211agent eval data/benchmarks/stage1_queries.jsonl --limit 10 --rerank
```

Metrics reported:

- `recall_at_1`
- `recall_at_3`
- `recall_at_5`
- `mrr`

If your benchmark includes explicit normalized constraints and you want to test
retrieval separately from query understanding, pass:

```bash
uv run 211agent eval data/benchmarks/stage1_queries.jsonl --use-constraints
```

Supported constraint keys:

```json
{
  "constraints": {
    "counties": ["MARION"],
    "cities": ["Indianapolis"],
    "benchmark_categories": ["Food"],
    "subcategories": ["Food"]
  }
}
```

## Playground

Run:

```bash
python3 playground.py "I need a food pantry in South Bend"
```

By default it uses the full Indiana 211 deduped CSV and the no-network heuristic
planner. You can edit the variables at the top of `playground.py`, or pass flags:

```bash
python3 playground.py --data curated --planner llm "I need help with utilities in Lake County"
```

Use `--planner llm` only when you want to make a real OpenAI/OpenRouter tool
calling request.

## Entrypoints

Both forms work:

```bash
uv run 211agent ask "I need food in Marion County"
uv run python main.py ask "I need food in Marion County"
```

## Tests

```bash
uv run python -m unittest discover -s tests
```
