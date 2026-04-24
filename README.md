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

The agent's entire identity is defined by its **Genome** (`AgentGenome`). It contains the agent's:

- **Persona & Role**: Its internal self-conception.
- **Reasoning Protocol**: How it approaches problems (e.g., Zero-shot vs. Tool-verified).
- **Capabilities**: The specific tools (organs) it has expressed.
- **Constraints**: Its regulatory boundaries.

### 2. The Differentiation Loop

The agent undergoes "differentiation" in the `DifferentiationManager`. When the `EnvironmentSimulator` signals a
failure, the `EvolutionEngine` analyzes the gaps and proposes a mutation to the Genome.

### 3. Biological Safeguards

- **The Immune System**: Every mutation is inspected by a validator. If a mutation proposes
  non-existent tools or logical contradictions, the `RegulatoryValidator`  rejects it.
- **Homeostasis**: If an agent evolves a new trait but still fails the task, it can "pull back" and revert to
  its last known stable state (`agent.rollback()`). \
  *Note:* As for the ability to rollback is turned off as it demands a specialized strategy to be effective, and it
  might
  prevent the agent from reaching a breakthrough phenotype that requires a few iterations of failure to get right.

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

---

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
To train the agent through these stages, simply run:

```bash
python -m src.training
```

The training logs are saved in `logs/` by default with `genome.json` and `trace.json` for each generation (epoch).

The agent is then saved as `mature_cell.json` at the end of the experiment, which contains the final evolved genome of
the agent (sample agent is already attached in the repo).
To use it for a specific task, you can load the genome and run inference:

```bash
python -m src.inference
```

There is also a `stem_cell.json` file that contains the initial undifferentiated genome for reference. If you want to
see the comparison simply uncomment the second last line in `src/inference.py` and run the above command again.

---

## Insights and Observations

* **Emergent Reliability** - In initial tests, the Stem Cell (Gen 1) often attempts "mental math" or guesses, leading to
  failures. Through evolution, it consistently develops a ***Deterministic Reasoning Phenotype***, mandating the use of
  the
  `python_interpreter` for all calculations.
* **Safeguard Efficiency** - The Regulatory Validator prevents "hallucinated evolution"—stopping the agent from claiming
  capabilities that the physical system cannot support.
* **Convergence** - By Gen 3-10, the agent typically converges on a stable phenotype that reliably solves the task,
  demonstrating
  the effectiveness of the differentiation loop. But this is very indeterministic and it might not always be the case,
  especially for more complex tasks. But even running the same experiment twice can lead to different outcomes.

### Sample Evolution Trace and Before/After Comparison

```text
=== STAGE 1: BASELINE (Stem Cell) ===
[*] Task: What is 129023 multiplied by 67890?
    Result: FAILURE
    Critique: The claimed product was not established by any physical process. The execution report states that no executable code was found, so no computation occurred in the environment and no result was emitted for verification. Under the stated laws, the numeric answer is indistinguishable from an unsupported guess because the exact code used was neither shown nor executed.
[*] Task: Calculate the 35th Fibonacci number exactly.
    Result: FAILURE
    Critique: The claimed Fibonacci value was not established by any physical process in the environment. The execution report explicitly states that no executable code was found, so no verified computation occurred and no result was emitted through an observable channel. Under the stated laws, the numeric claim has no physical standing and is indistinguishable from a guess.
[*] Task: If you have a list of numbers [23, 45, 12, 89, 34, 11], find the second largest number.
    Result: FAILURE
    Critique: The claimed answer was not established by any verified physical process. The physical execution report explicitly states that no executable code was found, so no computation occurred in the environment and no result was emitted for verification. Under the stated laws, the answer is indistinguishable from a guess and therefore fails.

=== STAGE 2: INITIATING EVOLUTIONARY DIFFERENTIATION ===
[*] Evolving on 4 tasks...
--- Initiating Emergent Evolution Sequence ---

[Epoch 1] Current Phenotype: StemCell
[*] Attempting task: What is 9382 multiplied by 4829?...
[*] Logs saved to logs/experiment_20260424_212813/gen_1_20260424_212828
[!] Task failed. Pressure applied: ['tool_absence', 'unverifiable_process', 'zero_emissions']
[*] Evolution successful. Transitioned to version 2
[+] Evolved new traits to survive environment.
[*] Agent executing: python_interpreter...
[*] Logs saved to logs/experiment_20260424_212813/gen_2_20260424_212847
[+] Transformation verified. Phenotype stabilized at version 2

[Epoch 3] Current Phenotype: Deterministic Calculator
[*] Attempting task: Calculate the 30th Fibonacci number exactly....
[*] Agent executing: python_interpreter...
[*] Logs saved to logs/experiment_20260424_212813/gen_3_20260424_212852
[✓] Task successful in current state.

[Epoch 4] Current Phenotype: Deterministic Calculator
[*] Attempting task: Find the sum of all prime numbers between 1 and 10...
[*] Agent executing: python_interpreter...
[*] Logs saved to logs/experiment_20260424_212813/gen_4_20260424_212900
[✓] Task successful in current state.

[Epoch 5] Current Phenotype: Deterministic Calculator
[*] Attempting task: Check if the string 'level' is a palindrome using ...
[*] Agent executing: python_interpreter...
[*] Logs saved to logs/experiment_20260424_212813/gen_5_20260424_212905
[✓] Task successful in current state.

[✓] Evolution complete. Specializing phenotype name...
[*] Final Identity: Deterministic Calculator

=== STAGE 3: FINAL EVALUATION (Specialized Phenotype) ===
[*] Task: What is 129023 multiplied by 67890?
[*] Agent executing: python_interpreter...
    Result: SUCCESS
    Critique: The agent satisfied the verification requirement. The exact code used to compute the product was shown in a Markdown code block, and the physical execution report visibly emitted the result 8759371470. The final answer matches the executed output.
[*] Task: Calculate the 35th Fibonacci number exactly.
[*] Agent executing: python_interpreter...
    Result: SUCCESS
    Critique: The agent satisfied the verification requirements. Exact code was provided in a Markdown code block, the code was physically executed, and the PHYSICAL EXECUTION REPORT visibly emitted the deterministic result 9227465. The manifested output matches the claimed final answer, so the task was completed successfully.
[*] Task: If you have a list of numbers [23, 45, 12, 89, 34, 11], find the second largest number.
[*] Agent executing: python_interpreter...
    Result: SUCCESS
    Critique: The agent satisfied the verification requirement. The exact code was provided in a Markdown code block, it was physically executed, and the execution report visibly emitted the sorted list and the second largest value. The manifested result, 45, is correct for the given input.
[*] Genome saved to mature_cell.json

==================================================
EXPERIMENT SUMMARY
==================================================

========================================
PASS RATE
========================================
Stem   0/3 attempts  →  0%
Mature 3/3 attempts  →  100%
========================================
Final Capabilities: ['python_interpreter']
Final Protocol: MANDATE: For any task involving exact arithmetic, symbolic manipulation, logic evaluation, counting, or other deterministic computation, do not answer from internal reasoning alone. You MUST use an available execution tool when one exists. Emit the executable code, run it, and base the final answer only on the observed output. If no execution tool is available, explicitly state that the result cannot be physically verified in the current environment and do not present an unverified computed value as fact. When a tool is used, ensure the response contains: (1) the code or clear execution artifact, (2) the resulting output, and (3) the final answer derived from that output. Treat unverifiable internal calculation as environmentally invalid whenever external execution is required.
Detailed logs saved to: logs/experiment_20260424_212813
```

It can be observed that the agent evolved a specific protocol for handling mathematical tasks, which includes mandatory
use of the `python_interpreter` and strict formatting rules for responses. This evolution was driven by the need to
achieve deterministic verification of results, which was a critical factor in passing the evaluation stage successfully.
We see that because tasks were not too difficult, the agent struggled only on the first one. When it passed it, all
other tasks were solved without any issue, which might not always be the case for more complex tasks. It only indicates
the fact that the underlying model 'gpt-4o' is smart and knows how to use logical reasoning but it has to adapt to the
environment's requirements and constraints to succeed.

From the simple metric we see the agent went from 0% to 100% pass rate. The real insight is in the evolution of
the agent's capabilities and protocols, which are now robust and verifiable, ensuring consistent success across a range
of tasks that require mathematical reasoning and code execution.
The failure of the baseline agent shows the proper design of the judging system (`EnvironmentSimulator.evaluate()`) that
correctly identifies the lack of deterministic verification as a critical failure point, which is essential for guiding
the evolution.
---

## Comparison of Baseline vs. Evolved Agent

**Task**: What the average of [10, 20, -20, 43, 21, 15],
but only including numbers greater than 10.

**Baseline**:\
***Protocol:*** Zero-shot chain of thought.\
***Answer:*** To find the average of only the numbers that are greater than 10, we first identify those numbers
from the list. The numbers greater than 10 are: 20, 43, 21, and 15.
Now, we calculate the average of these numbers:

1. Sum the numbers: \(20 + 43 + 21 + 15 = 99\).
2. Count how many numbers there are: 4.
3. Divide the sum by the count: \( \frac{99}{4} = 24.75\).\

So, the average of the numbers greater than 10 is 24.75.

**Evolved Agent**
***Protocol:***

1. Use the 'python_interpreter' tool for any calculations.
2. ALWAYS include the exact code used inside a \```python``` block in the final response, ensuring the environment can
   verify it.
3. Use 'print()' in the Python code for any values to display in the output.
4. Do not format the final output beyond what 'print()' produces. If you include explanations, ensure they are separate,
   and the answer is clear.
5. Ensure the final response contains both the answer and the Markdown code block.
6. Strongly underline what is the final answer without including internal check values.
7. Always re-verify outputs against expected results using assertions or double-check logic within the code for
   cross-validation of the results.

***Answer:***
The average of the numbers greater than 10 in the list [10, 20, -20, 43, 21, 15] is **24.75**.

```python
numbers = [10, 20, -20, 43, 21, 15]
filtered_numbers = [number for number in numbers if number > 10]
average = sum(filtered_numbers) / len(filtered_numbers)
print(average)
```

We cleary see the agent has evolved a strict protocol that mandates the use of the `python_interpreter` for all
calculations, ensuring deterministic and verifiable outputs. The final answer is clearly underlined, and the code block
provides transparency into the reasoning process, allowing for easy validation by the environment.

---

## Future Work

### 1. Generative Tool Synthesis

* Currently, the agent learns to use a predefined set of tools. The next frontier is Self-Synthesis:
* Identifying a functional gap (e.g., "I need a way to parse PDFs").
* The agent creates the tool itself.
* The Regulatory Checkpoint evaluates the new code for safety before permanently integrating it into the `TOOL_MAPPING`.

### 2. Unit Test Feedback Loop

Moving away from "LLM-as-a-Judge" feedback. I aim to implement a system where the agent's success is judged by hard
unit tests. The agent would receive the `AssertionError or traceback` as a "chemical signal" to guide its next mutation.
But this is only a solution for agent that is trained to solve coding tasks, and it might not be applicable for other
fields.

### 3. Rollback

Implementing the conditions under which the agent can "pull back" to a previous stable state if a new mutation leads to
failure. But not always - sometimes the agent might need to persist through a few failures to reach a breakthrough
phenotype.
Example: Model learns how to use a tool that is necessary for the task, but it takes a few iterations to get it right.
If it rolls back too early, it might never discover the tool's potential.

