# HoosierHelp

HoosierHelp is an evaluation benchmark for Indiana 211-style resource-search
agents. The benchmark focuses on a narrow search workflow: after a multi-turn
conversation with a simulated user, the agent should emit the correct
`search_resources` tool call, inspect the returned resources, and select the
right resource IDs.

The evaluation scores both tool-call arguments and final resource selection.
This keeps the benchmark focused on whether the agent collected and preserved
the right user facts: service need, location, schedule, intake methods,
documents, and eligibility constraints, then used the tool results correctly.

## What Is Evaluated

Each case contains hidden user facts, expected `search_resources` calls, and
ground-truth resources. Cases cover single-need and composite two-need users,
all-hard constraints, and acceptable-alternative constraints. Acceptable
alternatives include cases such as a preferred ZIP code with a broader city or
county also acceptable.

The simulated user reveals facts through one of several behavior modes:

```text
normal
rambling
impatience
self_contradictory
unsupported_request
```

The agent is evaluated on whether its tool calls match the expected normalized
arguments and whether the final answer selects acceptable returned resource
IDs. Field-level scores, executed tool results, final selections, and full
transcripts are saved for analysis.

## Main Components

```text
eval/                         Online evaluation harness
  tool_call_eval.py            Evaluation runner library
  tool_call_backends.py        Local HF and Responses API backends
  tool_call_parsers.py         Tool-call parsers
  tool_call_schema.py          Normalization and scoring
  llm_user.py                  LLM simulated user
  spec_generation.py           Reproducible runtime user-spec generation

tools/                        Indiana 211 resource schema and indexing
data/benchmark/               Tagged benchmark resource files
data/benchmark_builder/       Offline resource filtering and analysis scripts
scripts/                      Evaluation utilities and provider smoke tests
archive/training_baselines/   Archived training code, data, and OPSD runs
```

## Data

Primary resource files live under `data/benchmark/`:

```text
filtered_resources_tagged.csv
filtered_resources_raw.csv
```

User specs are generated at evaluation time from `sample_seed` and
`sample_count`. This avoids stale local case files while preserving
reproducibility: for the same seed, a larger sample count includes the smaller
sample count as its prefix. Each run writes the generated specs to
`generated_specs.json` inside the output directory.

`sample_count` controls the number of base user specs. The evaluator then
expands each spec across every entry in `user_behaviors`, so total planned
conversations are `sample_count * len(user_behaviors)`.

## Running Evaluation

Default local backbone: Qwen3.6 35B-A3B through `llama.cpp` server.

Start the server in one terminal:

```bash
export QWEN36_35B_A3B_GGUF=/path/to/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf

llama-server \
  -m "$QWEN36_35B_A3B_GGUF" \
  -ngl 99 \
  -c 8192 \
  --reasoning-format deepseek \
  --host 127.0.0.1 \
  --port 8080
```

Edit `LLM_CONFIG` and `RUN_CONFIG` in `main.py`, then run:

```bash
uv run python main.py
```

`LLM_CONFIG` selects the agent and user models. `RUN_CONFIG` controls run shape
such as sample count, sample seed, behavior set, and parallelism.

Override `LLAMA_CPP_BASE_URL` if the server is not listening at
`http://127.0.0.1:8080/v1`.

By default, both the agent and user simulator use the same `llama.cpp` server.
When `user_provider="llama_cpp"` is used without `user_model`, the evaluator
reuses the agent `model`. Generation token limits are configured in
`LLM_CONFIG` and recorded in each run's `args.json`.

For `llama.cpp` thinking models, `agent_enable_thinking` and
`agent_thinking_budget_tokens` can be set per run. On an 8k-context local
server, keep the user simulator thinking disabled and use a small agent budget
such as `256` when comparing thinking mode; larger budgets can push multi-turn
conversations over the context limit.

OpenRouter Responses API:

Set `backend="responses"`, `provider="openrouter"`,
`model="openai/gpt-4.1-mini"`, `user_provider="openrouter"`, and
`user_model="openai/gpt-4.1-mini"` in `LLM_CONFIG`, then run `uv run python
main.py`.

If `output_dir` is omitted, the evaluator creates a timestamped directory
under `experiments/tool_call_eval/`. Each run writes `args.json`,
`records.jsonl`, and `summary.json`.

For OpenRouter connectivity:

```bash
uv run python scripts/smoke_openrouter.py
```

## Archived Training Work

Training baselines and their experiment artifacts are archived under
`archive/training_baselines/` so the active project surface stays focused on
the evaluation benchmark.

## Configuration

`main.py` is the config-driven local entrypoint. It starts from `EvalConfig`
and applies only the explicit values in `LLM_CONFIG` and `RUN_CONFIG`, so
parser and argument handling do not live in the evaluation runner.

`tool_result_limit` is not part of `main.py`; it is a tool/eval protocol value
defined by `tools.indiana211.DEFAULT_RESULT_LIMIT`.
