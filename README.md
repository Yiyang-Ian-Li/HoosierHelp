# Agent

Local resource search agent for Indiana 211 data.

Run evaluation by editing `CONFIG` in `main.py`, then executing:

```bash
python3 main.py
```

For a single ad hoc query, edit `CONFIG` in `playground.py` and run:

```bash
python3 playground.py
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

Benchmark case specs live under:

```text
data/benchmark/case_specs.json
data/benchmark/user_cards.json
data/benchmark/dataset_report.md
```

`case_specs.json` includes the deterministic source data and singleton
`ground_truth_resource_ids`. `user_cards.json` is generated from those specs for LLM
simulated eval.

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
data/benchmark_builder/build_benchmark_data.py
```

`agent/agent.py` is a generic Responses API function-calling loop. It receives
tool schemas and a plain `tool_functions` dictionary, executes requested
function calls, appends `function_call_output`, and asks the model again.

`tools/indiana211.py` exposes the Indiana 211 `search_resources` tool, with
models, tag parsing, and schedule parsing split into adjacent helper modules.
County and service category narrow retrieval, county also includes statewide
resources, and city/ZIP are ranking signals. Optional tag fields should only be
used for constraints the user explicitly makes required. Missing document data
is treated as `none`; missing fee data is exposed as `fee_options=unknown`.
Schedule filtering uses natural fields such as `available_days`,
`available_at_or_after`, `requires_weekend`, `requires_24_hours`, and
`allow_appointment_only`; the old `schedule_tags` filter is intentionally not
kept.

## LLM Simulated Eval

Generate or refresh deterministic case specs:

```bash
.venv/bin/python data/benchmark_builder/build_benchmark_data.py --easy 50 --medium 100 --hard 50
```

Generate LLM simulated-user cards from the case specs:

```bash
.venv/bin/python data/benchmark_builder/build_user_cards.py --model gpt-4.1-mini
```

Run an OpenAI evaluation with an LLM simulated user:

```bash
python3 main.py
```

`main.py` is the eval entrypoint. Its `CONFIG` controls provider, agent
type/model, user model, data paths, turn limit, and parallel jobs.

Eval stops on the first agent message that includes a concrete resource ID
(`in211-...`). That message is treated as the agent's single final
recommendation. Before that point, the agent may ask follow-up questions and
make multiple tool calls, but should not cite resource IDs tentatively.
Retrieval metrics still read tool results. After the final recommendation, the
simulated user gives a structured satisfaction rating.

Analyze an existing run:

```bash
python3 -m eval.analyze_run experiments/<run-id>
```

## Tests

```bash
python3 -m unittest discover -s tests
```
