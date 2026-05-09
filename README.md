# Agent

Local resource search agent for Indiana 211 data.

Run it by editing `CONFIG` in `main.py`, then executing:

```bash
python3 main.py
```

By default this uses the OpenAI Responses API:

```bash
export OPENAI_API_KEY="..."
```

`OPENROUTER_API_KEY` can still be configured in `main.py` by setting
`"provider": "openrouter"` if the selected OpenRouter model supports the
Responses-style endpoint.

## Data

The retained Indiana 211 source files are:

```text
data/indiana211/indiana211_resources_raw_all_counties.json
data/indiana211/indiana211_resources_deduped.csv
data/indiana211/indiana211_resource_county_rows.csv
data/indiana211/indiana211_counties.csv
```

LLM simulated-user benchmark data lives under:

```text
data/benchmark/user_cards.json
data/benchmark/ground_truth.json
data/benchmark/dataset_report.md
```

Experiment outputs live outside `data/`:

```text
experiments/<timestamp>__agent-react__agentmodel-...__usermodel-...__n.../
experiments/<run>/conversations/<user_id>.json
```

## Code

Core files:

```text
main.py
agent/agent.py
agent/llm.py
tools/indiana211.py
eval/run_eval.py
eval/analyze_run.py
eval/build_benchmark_data.py
```

`agent/agent.py` is a generic Responses API function-calling loop. It receives
tool schemas and a plain `tool_functions` dictionary, executes requested
function calls, appends `function_call_output`, and asks the model again.

`tools/indiana211.py` keeps the Indiana 211 data loading, `search_resources`
schema, argument parsing, and filtering logic together. Requested fields are
filters: a resource must match every non-empty field to be returned.

## LLM Simulated Eval

Generate or refresh tau-bench-style hidden user cards and ground truth:

```bash
.venv/bin/python -m eval.build_benchmark_data --cases 200
```

Run an OpenAI evaluation with an LLM simulated user:

```bash
.venv/bin/python -m eval.run_eval --provider openai --model gpt-4.1-mini --users data/benchmark/user_cards.json --ground-truth data/benchmark/ground_truth.json --max-turns 8 --agent-type default --sim-user-model gpt-4.1-mini --jobs 8
```

Use `--agent-type react` to evaluate the ReAct variant, which asks the model to
emit a `Thought:` line before `Answer:`.

Analyze an existing run:

```bash
python3 -m eval.analyze_run experiments/<run-id>
```

## Tests

```bash
python3 -m unittest discover -s tests
```
