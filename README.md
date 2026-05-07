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

Generated benchmark data, curated benchmark pools, reports, and cleaning
notebooks are intentionally removed.

## Code

Core files:

```text
main.py
agent/agent.py
agent/llm.py
tools/indiana211.py
```

`agent/agent.py` is a generic Responses API function-calling loop. It receives
tool schemas and a plain `tool_functions` dictionary, executes requested
function calls, appends `function_call_output`, and asks the model again.

`tools/indiana211.py` keeps the Indiana 211 data loading, `search_resources`
schema, argument parsing, and filtering logic together. Requested fields are
filters: a resource must match every non-empty field to be returned.

## Tests

```bash
python3 -m unittest discover -s tests
```
