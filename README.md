# Agent

Local resource search agent for Indiana 211 data.

Run evaluation by editing `CONFIG` in `main.py`, then executing:

```bash
uv run python main.py
```

For a single ad hoc query, edit `CONFIG` in `playground.py` and run:

```bash
uv run python playground.py
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
data/benchmark/filtered_resources_raw.csv
data/benchmark/filtered_resources_tagged.csv
```

`case_specs.json` includes the deterministic source data, single/composite
needs, and one or two `ground_truth_resource_ids`. `user_cards.json` is
generated from those specs for LLM simulated eval.

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
eval/agent_instructions.py
eval/run_eval.py
eval/analyze_run.py
data/benchmark_builder/resource_filter.py
data/benchmark_builder/build_user_specs.py
data/benchmark_builder/build_user_cards.py
```

`agent/agent.py` is a generic Responses API function-calling loop. It receives
tool schemas and a plain `tool_functions` dictionary, executes requested
function calls, appends `function_call_output`, and asks the model again.
Benchmark-specific agent instructions live in `eval/agent_instructions.py`.

`tools/indiana211.py` exposes the Indiana 211 `search_resources` tool, with
models, tag parsing, and schedule parsing split into adjacent helper modules.
The default tool index is `data/benchmark/filtered_resources_tagged.csv`.
County, city, ZIP, service category, and schedule fields are hard filters when
provided. Intake methods and document requirements are returned for the final
recommendation but are not search filters. Schedule filtering uses one
`schedule` object with either `{day, time}` or `{requires_24_hours: true}`.

Benchmark cases are now generated as `single` or `composite`. A single case has
one service need, one schedule requirement, and one location requirement. A
composite case has two service needs with separate non-overlapping schedule
requirements and a shared location. Ground truth resources include the intake
methods and document requirements the final answer should report.

## LLM Simulated Eval

Generate or refresh deterministic case specs:

```bash
uv run python data/benchmark_builder/build_user_specs.py --single 150 --composite 150
```

Generate LLM simulated-user cards from the case specs:

```bash
uv run python data/benchmark_builder/build_user_cards.py --model openai/gpt-4.1
```

Resource and user-spec distribution notebooks:

```text
data/benchmark_builder/analysis/resources_stat.ipynb
data/benchmark_builder/analysis/user_stat.ipynb
```

Run an OpenAI evaluation with an LLM simulated user:

```bash
uv run python main.py
```

`main.py` is the eval entrypoint. Its `CONFIG` controls provider, agent
type/model, user model, data paths, turn limit, and parallel jobs.
Set `case_type` to `single`, `composite`, or `all` in `main.py`. The same
option is available from the CLI:

```bash
uv run python -m eval.run_eval --case-type single
uv run python -m eval.run_eval --case-type composite
```

Eval stops on the first agent message that contains valid final JSON with a
`recommendations` list. The agent may ask follow-up questions and can execute
up to three real tool calls per case. ID hit scoring uses the first three final
recommendations; intake methods and document requirements are tracked as
auxiliary detail metrics. Retrieval metrics read tool results.

Analyze an existing run:

```bash
uv run python -m eval.analyze_run experiments/<run-id>
```

## Tests

```bash
uv run python -m unittest discover -s tests
```
