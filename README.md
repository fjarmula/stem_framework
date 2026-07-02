# Stem Framework

This project experiments with a "stem cell" agent architecture: start with a minimal agent, expose it to task pressure, and let it differentiate into a more specialized phenotype by changing its genome.

The current benchmark is intentionally not a calculator benchmark. It uses three stateful domains:

- `trading_floor`: parse market CSV files, exchange rules, and portfolios; emit a legal ledger.
- `security_sandbox`: inspect local toy source fixtures and isolate the candidate vector that reaches a protected branch.
- `matrix_database`: load graph records, traverse relation chains, filter properties, and emit answer sets with path traces.

The important shift is that benchmark grading is deterministic. The v2 benchmark no longer relies on an LLM judge for pass/fail. It compares emitted artifacts against private verifier files under `benchmarks/private/`.

## What This Demonstrates

The stem agent starts with no benchmark organs. During training, each failed domain causes one domain-specific organ to be enabled:

```text
trade_001  -> trading_floor_solver
sec_001    -> security_sandbox_solver
matrix_001 -> matrix_database_solver
```

The saved mature phenotype can then solve the validation tasks and any compatible inference task for domains it has acquired.

This is structural differentiation, not gradient training. The current implementation proves capability gating, deterministic evaluation, and before/after comparison. It does not yet prove that the agent invents entirely new algorithms from scratch.

## Project Layout

```text
.
├── benchmarks/              # Public task artifacts and private expected verifier outputs
├── examples/inference/      # Hand-run inference examples
├── prompts/                 # LLM prompts for evolution, validation, and fallback evaluation
├── src/
│   ├── core/                # StemAgent and genome models
│   ├── evaluation/          # Deterministic benchmark verifier and environment simulator
│   ├── evolution/           # Differentiation engine and manager
│   ├── execution/           # Runtime tool registry
│   ├── regulatory/          # Mutation and generated-tool validation
│   ├── services/            # LLM, prompt, and task loading services
│   ├── inference.py         # Run a saved genome
│   └── training.py          # Train/evaluate the stem agent
├── config.yaml
├── mature_cell.json
├── stem_cell.json
└── tasks_v2.json
```

## Setup

Use Python 3.12+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` in the repository root:

```bash
GEMINI_API_KEY=your_key_here
```

The default `config.yaml` uses Gemini through Google's OpenAI-compatible endpoint:

```yaml
llm:
  provider: "gemini"
  model: "gemini-3.1-flash-lite"
  api_key_env: "GEMINI_API_KEY"
  base_url: "https://generativelanguage.googleapis.com/v1beta/openai/"
  structured_output_mode: "json"
  tool_call_mode: "manual"
```

To use OpenAI, change the provider, model, key env var, and base URL in `config.yaml`, then set `OPENAI_API_KEY` in `.env`.

## Train

```bash
python -m src.training
```

The run has three stages:

1. Baseline stem evaluation on validation tasks.
2. Evolution on training tasks.
3. Final mature phenotype evaluation on validation tasks.

Expected shape:

```text
Stem   0/3 attempts  ->  0%
Mature 3/3 attempts  ->  100%
```

Training logs are written under `logs/experiment_*`. Each generation has:

- `genome.json`
- `trace.json`

The final genome is saved to `mature_cell.json`.

## Inference

Run the saved mature genome against the validation benchmark:

```bash
python -m src.inference --benchmark validation
```

Run all train and validation episodes:

```bash
python -m src.inference --benchmark all
```

Run a single prompt file:

```bash
python -m src.inference --task-file examples/inference/pass_trade_003.txt
```

That task should pass. It is a new trading-floor task compatible with the acquired `trading_floor_solver`.

Run the deliberate fail fixture:

```bash
python -m src.inference --task-file examples/inference/fail_biology_001.txt
```

That task should fail because the mature genome has no `biology_lab` organ.

## How The Verifier Works

For v2 benchmark prompts, `EnvironmentSimulator` calls `verify_stateful_episode()` before any LLM judge path.

The verifier:

1. Parses the rendered benchmark prompt.
2. Reads the private expected artifact for known domains.
3. Parses the agent output JSON.
4. Compares exact final artifacts and required traces.

For unsupported domains, verification fails deterministically.

For non-v2 tasks, the older LLM-as-judge fallback still exists.

## Capability Boundaries

The mature genome does not get one universal benchmark tool. It has three separate organs:

- `trading_floor_solver`
- `security_sandbox_solver`
- `matrix_database_solver`

The agent routes v2 tasks by parsing `domain_id`. If there is no matching acquired organ, it returns a failure instead of trying to solve the v2 benchmark through general LLM reasoning.

Local gating check:

```text
stem       -> fails all benchmark domains
trade_only -> passes trading only
trade_sec  -> passes trading + security only
full       -> passes trading + security + matrix
```

## Current Limitations

The organs are currently pre-registered deterministic tools. Evolution enables them after domain pressure, but it does not yet synthesize robust new organs from scratch.

Each domain has only one training episode and one validation episode. This makes learning fast and still somewhat scripted. The next improvement is to add multiple train episodes per domain and require repeated passes before stabilizing an organ.

The parsers are narrow by design:

- Trading expects the current rule wording and buy-only close portfolios.
- Security expects toy source fixtures with candidate vectors.
- Matrix expects the current query text style.

These limits are acceptable for the current benchmark scaffold, but they should be broadened before presenting the system as a general stem-agent framework.
