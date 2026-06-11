# HoosierHelp

HoosierHelp is an evaluation and training sandbox for Indiana 211-style
resource-search agents. The current benchmark focuses on one narrow capability:
after a multi-turn conversation with a simulated user, the agent should emit the
correct `search_resources` tool call.

The evaluation scores tool-call arguments only. It does not execute the search
tool and does not evaluate final resource recommendations. This keeps the
benchmark focused on whether the agent collected and preserved the right user
facts: service need, location, schedule, intake methods, documents, and
eligibility constraints.

## What Is Evaluated

Each case contains hidden user facts and an expected `search_resources` call.
The simulated user reveals those facts through one of several behavior modes:

```text
normal
rambling
impatience
self_contradictory
unsupported_request
```

The agent is evaluated on whether its final tool call matches the expected
normalized arguments. Field-level scores and full transcripts are saved for
analysis.

## Main Components

```text
eval/                         Online evaluation harness
  tool_call_eval.py            Main eval entrypoint
  tool_call_backends.py        Local HF and Responses API backends
  tool_call_parsers.py         Tool-call parsers
  tool_call_schema.py          Normalization and scoring
  llm_user.py                  LLM simulated user

train/                        Training data and baseline methods
  build_turn_samples.py        Build assistant-turn training samples
  build_teacher_samples.py     Generate privileged teacher completions
  opsd.py                     On-policy self-distillation baseline
  grpo.py                     GRPO baseline

tools/                        Indiana 211 resource schema and indexing
data/benchmark/               Benchmark specs and tagged resources
experiments/                  Eval and training outputs
```

## Data

Primary benchmark files live under `data/benchmark/`:

```text
case_specs.json
user_specs_train_100.json
user_specs_dev_50.json
user_specs_test_50.json
filtered_resources_tagged.csv
filtered_resources_raw.csv
```

The train/dev/test user spec files are disjoint by source resource, so training
and held-out evaluation can use different underlying cases.

## Running Evaluation

Local Qwen baseline:

```bash
uv run python -m eval.tool_call_eval \
  --backend local \
  --model Qwen/Qwen3-4B-Instruct-2507
```

OpenAI or OpenRouter Responses API:

```bash
export OPENAI_API_KEY="..."
uv run python -m eval.tool_call_eval --backend responses --provider openai --model gpt-4.1-mini

export OPENROUTER_API_KEY="..."
uv run python -m eval.tool_call_eval --backend responses --provider openrouter --model openai/gpt-4.1-mini
```

If `--output-dir` is omitted, the evaluator creates a timestamped directory
under `experiments/tool_call_eval/`. Each run writes `args.json`,
`records.jsonl`, and `summary.json`.

For OpenRouter connectivity:

```bash
uv run python scripts/smoke_openrouter.py
```

## Training Baselines

The repo currently includes two local training baselines:

- OPSD: a privileged self-teacher sees the user behavior label and provides the
  target distribution for the same model under the normal prompt.
- GRPO: samples final tool-call states and optimizes a soft reward over valid
  tool-call fields.

Typical workflow:

```bash
uv run python -m train.build_turn_samples --help
uv run python -m train.build_teacher_samples --help
uv run python -m train.opsd --help
uv run python -m train.grpo --help
```

Adapters produced by training can be evaluated by passing the adapter path to
`eval.tool_call_eval`.

## Configuration

`main.py` is a small config-driven wrapper around `eval.tool_call_eval` for
repeatable local runs. Direct CLI usage is usually better for experiments that
need explicit paths or model settings.
