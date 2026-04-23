# Stem Agent: Autonomous Genomic Evolution

> "A stem cell doesn't know what it will become. It reads signals from its environment and transforms... What if AI
> agents worked the same way?"

This project implements a **Stem Agent** architecture—a minimal, undifferentiated AI agent that specializes into a "
mature" phenotype through environmental pressure and genomic mutation. Instead of hand-coding specific agents for
specific tasks, we start with a "Stem Cell" and let it evolve the tools and protocols it needs to survive.

In this implementation, the agent is evolved to solve tasks using a `python_interpreter` tool, but the framework is
designed to be extensible to any set of tools or capabilities.

This implementation is based on the concepts explored in:
**[Genomic Evolution of Autonomous Agents (Arxiv: 2603.22359)](https://arxiv.org/pdf/2603.22359)**

---

## Core Concepts

### 1. The Genome

The agent's entire identity is defined by its **Genome** (`src/core/genome.py`). It contains the agent's:

- **Persona & Role**: Its internal self-conception.
- **Reasoning Protocol**: How it approaches problems (e.g., Zero-shot vs. Tool-verified).
- **Capabilities**: The specific tools (organs) it has expressed.
- **Constraints**: Its regulatory boundaries.

### 2. The Differentiation Loop

The agent undergoes "differentiation" in the `DifferentiationManager`. When the environment (the simulator) signals a
failure, the **Evolution Engine** analyzes the gaps and proposes a mutation to the Genome.

### 3. Biological Safeguards

- **Regulatory Validator (The Immune System)**: Every mutation is inspected by a validator. If a mutation proposes
  non-existent tools or logical contradictions, the "immune system" rejects it.
- **Homeostasis (Rollback)**: If an agent evolves a new trait but still fails the task, it can "pull back" and revert to
  its last known stable state (`agent.rollback()`).

---

## File Structure

```text
.
├── tasks.yaml           # Evolution and Validation task sets
├── requirements.txt     # Dependencies
├── mature_agent.json    # Example of final evolved agent genome
├── prompts/             # System instructions for different modules
└── src
    ├── training.py      # Entry point for experiments
    ├── inference.py     # Core for agent-user interaction
    ├── core/            # StemAgent and Genome definitions
    ├── evolution/       # Mutation engine and lifecycle management
    ├── evaluation/      # Environment simulator and feedback logic
    ├── regulatory/      # Safety and implementation validators
    ├── execution/       # Physical tool registry (e.g., Python Interpreter)
    ├── utils/           # config management
    └── services/        # LLM, Prompts, and Task loading
```

## Getting Started

### 1. Prerequisites

* Python 3.12+
* OpenAI API Key (for LLM interactions)

### 2. Installation

```bash
# Clone the repository and navigate to the project directory
git clone https://github.com/fjarmula/stem_framework.git
cd stem_framework
# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Create a .env file in the root directory with your OpenAI API key:

```bash
OPENAI_API_KEY=your_api_key_here
```

*(Optional)* You can adjust the `config.yaml` to set parameters for the experiment, such as the number of generations,
turns per generation, and the model to use.

### 4. Running the Experiment and Inference

The system runs in three stages: Baseline (Stem Cell), Evolution (Differentiation), and Evaluation (Specialized Agent).

```bash
python -m src.training
```

The training logs are saved in `logs/` by default with `genome.json` and `trace.json` for each generation (epoch).

The model is then saved as `mature_agent.json` at the end of the experiment, which contains the final evolved genome of
the agent (sample agent is already attached in the repo).
To use it for a specific task, you can load the genome and run inference:

```bash
python -m src.inference
```

## Insights and Observations

* **Emergent Reliability** - In initial tests, the Stem Cell (Gen 1) often attempts "mental math" or guesses, leading to
  failures. Through evolution, it consistently develops a ***Deterministic Reasoning Phenotype***, mandating the use of
  the
  `python_interpreter` for all calculations.
* **Safeguard Efficiency** - The Regulatory Validator prevents "hallucinated evolution"—stopping the agent from claiming
  capabilities that the physical system cannot support.
* **Convergence** - By Gen 3-4, the agent typically converges on a stable phenotype that reliably solves the task,
  demonstrating
  the effectiveness of the differentiation loop (but sometimes it might depend on task difficulty).

## Future Work

### 1. Generative Tool Synthesis

* Currently, the agent learns to use a predefined set of tools. The next frontier is Self-Synthesis:
* Identifying a functional gap (e.g., "I need a way to parse PDFs").
* The agent writes the Python tool itself.
* The Regulatory Checkpoint evaluates the new code for safety before permanently integrating it into the `TOOL_MAPPING`.

### 2. Unit Test Feedback Loop

Moving away from "LLM-as-a-Judge" feedback. I aim to implement a system where the agent's success is judged by hard
unit tests. The agent would receive the `AssertionError or traceback` as a "chemical signal" to guide its next mutation.

### 3. Rollback

Implementing the conditions under which the agent can "pull back" to a previous stable state if a new mutation leads to
failure. But not always - sometimes the agent might need to persist through a few failures to reach a breakthrough
phenotype.
Example: Model learns how to use a tool that is necessary for the task, but it takes a few iterations to get it right.
If it rolls back too early, it might never discover the tool's potential.

