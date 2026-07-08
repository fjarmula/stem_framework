# Stem Framework

This project experiments with a "stem cell" agent architecture: start with a minimal agent, expose it to task pressure, and let it differentiate into a more specialized phenotype by changing its genome and compiling new runtime organs.

The current benchmark is intentionally not a calculator benchmark. It uses three stateful domains:

- `trading_floor`: parse market CSV files, exchange rules, and portfolios; emit a legal ledger.
- `security_sandbox`: inspect local toy source fixtures and isolate the candidate vector that reaches a protected branch.
- `matrix_database`: load graph records, traverse relation chains, filter properties, and emit answer sets with path traces.

The important shift is that benchmark grading is deterministic and stateful. The v2 benchmark no longer relies on an LLM judge for pass/fail. It runs multi-turn episodes, requires a physical runtime organ invocation, records disk traces, and compares emitted artifacts against private verifier files under `benchmarks/private/`.

## What This Demonstrates

The stem agent starts with no benchmark organs. In Stage 1 it still attempts every observation turn with general reasoning, but because it has no runtime organ it cannot produce a verifiable physical trace and should score `0%`.

During training, failed episodes pressure the evolution engine to produce a `TransformationPlan` containing a compile-ready Python source string. That source is written into `src/compiled_skills/`, registered into the active runtime belt, and routed deterministically by `domain_id`.

When a generated organ crashes or produces a logically inconsistent artifact, the failure is recorded as a phenotypic scar and fed back into the next mutation prompt. This is structural differentiation, not gradient training: selection pressure changes the executable body of the agent rather than only rewriting a persona prompt.

## Project Layout

```text
.
├── benchmarks/              # Public task artifacts and private expected verifier outputs
├── examples/inference/      # Hand-run inference examples
├── prompts/                 # LLM prompts for evolution and regulatory validation
├── src/
│   ├── core/                # StemAgent and genome models
│   ├── compiled_skills/     # Generated runtime organs created during evolution
│   ├── evaluation/          # Simulator plus split stateful runner/verifier modules
│   ├── evolution/           # Differentiation engine and manager
│   ├── execution/           # Runtime tool registry
│   ├── regulatory/          # Mutation and generated-tool validation
│   ├── services/            # LLM, prompt, and task loading services
│   ├── cli.py               # Shared argparse definitions and CLI helpers
│   ├── inference.py         # Run a saved genome
│   └── training.py          # Train/evaluate the stem agent
├── tasks/                   # Canonical per-task manifests and task-owned contracts
└── config.yaml
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

Run only one benchmark domain to control API spend:

```bash
python -m src.training --domain trading
```

Supported domain aliases:

- `trading`, `trade`, `trading_floor`
- `security`, `security_sandbox`
- `matrix`, `matrix_database`
- `all`

Limit evolutionary epochs for short clinical trials:

```bash
python -m src.training --domain trading --max-epochs 6
```

For `--domain trading`, the run trains on `trade_001`, validates the baseline on `trade_002`, evolves against `trade_001`, then validates the evolved phenotype on `trade_002` again.

The run has three stages:

1. Baseline stem evaluation on validation tasks. The stem cell attempts all turns but has no acquired runtime organ, so stateful benchmark outputs are marked unverifiable.
2. Evolution on training tasks.
3. Final mature phenotype evaluation on validation tasks.

Expected early-stage shape:

```text
Stem   0/N attempts  ->  0%
Mature 0..N/N attempts
```

Mature success is intentionally not guaranteed. A generated organ can be rejected by the regulatory validator, crash during runtime, or fail the deterministic verifier. Those failures are training signal, not simulator success.

Current observed status:

- `security_sandbox` has demonstrated the intended loop: the stem baseline fails, training compiles a new runtime organ from `sec_001`, and the evolved phenotype can pass held-out `sec_002`.
- `matrix_database` now exposes a task-owned `query_contract` in the first turn. The contract declares the seed, relation chain, step-specific filters, and path output shape without moving graph rules into the runner.
- `trading_floor` now exposes a task-owned `trading_contract` in the first turn. The contract declares targets, thresholds, fees, quantity fields, cooldown policy, and ledger row keys without moving trading rules into the architecture.

That means the core runtime loop and task-owned public contracts are in place. Remaining convergence failures should be treated as organ synthesis or task-contract quality issues, not as permission to add domain logic to `src/core`, `src/evolution`, or `src/evaluation`.

Training logs are written under `logs/experiment_*`. Each generation has:

- `genome.json`
- `trace.json`

Phenotypic scars are written to:

- `phenotypic_scars.json`

The final genome is saved to `mature_cell.json` only when all validation episodes pass.

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

That task should pass only if `mature_cell.json` was produced by a successful run that acquired a compatible compiled trading organ.

Run the deliberate fail fixture:

```bash
python -m src.inference --task-file examples/inference/fail_biology_001.txt
```

That task should fail because the mature genome has no `biology_lab` organ.

## How The Stateful Runtime Works

For v2 benchmark prompts, `EnvironmentSimulator` runs an explicit multi-turn environment episode. The agent does not receive the whole solution space on turn 1. The runner itself is domain-agnostic: task payloads provide an `artifact_manifest` that declares which public artifacts are released on each turn and how they are loaded.

The stateful runtime is split into focused modules under `src/evaluation/`: `stateful_contract.py` owns shared types and prompt parsing, `stateful_runner.py` owns the physical turn loop, `stateful_verifier.py` owns generic verifier dispatch, and `stateful_formatting.py` owns console output. `stateful_benchmark.py` remains as a compatibility facade for existing imports.

Each turn:

1. Parses the rendered benchmark prompt.
2. Replays the task's `artifact_manifest` to build one `observation_delta`.
3. Calls `StemAgent.execute_episode_turn()`.
4. Requires an acquired compiled organ to be invoked for the output to be verifiable.
5. Writes observation, action, and result trace files under the episode workspace.
6. Carries the last valid JSON object returned by the agent forward as opaque `memory` in the next observation payload.

After the final turn, the verifier:

1. Resolves the full task definition from the configured task source.
2. Parses the organ output JSON.
3. Dispatches to the task-owned private verifier if one is declared.
4. Falls back to exact expected JSON comparison if no verifier module is declared.
5. Emits deterministic failure tags such as `unverifiable_inference`, `missing_physical_trace`, `runtime_exception`, `ledger_mismatch`, or `multi_turn_collapse`.

For tasks without verifier artifacts, verification fails deterministically.

Non-v2 tasks are outside the MVP simulator contract and fail deterministically.

## Task Manifest Contract

The framework stays domain-independent by keeping task-specific facts in `tasks/` and benchmark artifacts, not in the runner, agent, or mutation engine. `config.yaml` controls the task source:

```yaml
experiments:
  dir: "tasks"
```

The loader expects a directory of per-episode task manifests. The old monolithic manifest format has been removed from the MVP to prevent duplicated task contracts from drifting.

A good v2 task should provide:

- `artifact_manifest`: the public observation schedule, including exactly which artifact slices appear on each turn.
- `output_contract`: final artifact shape, including nested object keys and row schemas when the verifier expects structured records.
- `clinical_probes`: cheap public invariants that catch partial artifacts before private verifier comparison.
- `private_verifier_artifacts`: verifier-only expected data, verifier module path, and optional `submission_source`; this is read by the framework registry and not rendered into the public prompt.
- Optional `structured_contract` manifest loads: task-owned machine-readable rules injected into `observation_delta`, such as traversal contracts, state transition contracts, row schemas, field mappings, thresholds, and output shape hints.
- Optional public probe loaders such as `python_function_probe`: deterministic public behavior observations declared by the task, not hard-coded into the runner.
- Optional selection metadata such as `probe_input_key`, `probe_result_key`, and `selection_match`: task-owned hints that let generated organs select public probe rows without memorizing domain-specific field names.

The security benchmark is the reference example for public probes and dynamic proof-key metadata. Matrix and trading use the same separation principle through `query_contract` and `trading_contract`: rules live in task payloads and artifacts, while the runtime only transports and verifies them.

`stateful_verifier.py` is now a generic dispatcher: it checks physical traces, validates the public output contract, writes the selected submission slice, and invokes the task-owned verifier module. Domain-specific scoring belongs under `benchmarks/private/<domain>/<episode>/verify.py`.

## Runtime Organ Contract

Generated organs are written to `src/compiled_skills/` at runtime and must match a genome capability name. The repository keeps only `src/compiled_skills/__init__.py`; generated organ files are ignored so training runs do not become source changes.

The validator accepts either:

- a module-level `run(observation: str) -> str`
- a class named exactly like the capability with `execute(self, observation: dict)`

Organs are routed by capability metadata:

```text
required_context includes domain_id:<domain>
```

The regulatory validator blocks static benchmark shortcuts, hard-coded benchmark paths, network/subprocess behavior, unrestricted dynamic execution, and file-stream writes. Stateful persistence should be returned in the transaction dictionary as `memory`, `state_trace`, or `internal_state`; the agent injects that payload into the next turn.

## Current Limitations

The generated organs are still LLM-authored Python snippets. The framework now makes those snippets real executable organs and records their failures, but convergence is not guaranteed.

Each domain has only one training episode and one validation episode. This makes learning fast and still somewhat scripted. The next improvement is to add multiple train episodes per domain and require repeated passes before stabilizing an organ.

The current verifier domains are narrow by design:

- Trading expects the current rule wording and buy-only close portfolios.
- Security expects toy source fixtures with candidate vectors.
- Matrix expects the current query text style.

The regulatory validator is intentionally conservative. It may reject useful code if the generated organ appears to hard-code benchmark fixtures or attempts unsafe state persistence.
