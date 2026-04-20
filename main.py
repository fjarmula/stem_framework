import asyncio
import os
import json
from dotenv import load_dotenv
from tasks import TASKS

# Core Framework
from core.stem import StemAgent
from evolution.engine import EvolutionEngine
from evolution.manager import DifferentiationManager
from regulatory.validator import RegulatoryValidator
from evaluation.simulator import EnvironmentSimulator

# Load configuration
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")


async def run_experiment():
    if not API_KEY:
        print("[!] Error: OPENAI_API_KEY not found in .env file.")
        return

    # 1. Initialize the Ecosystem
    agent = StemAgent(api_key=API_KEY)
    engine = EvolutionEngine(api_key=API_KEY)
    validator = RegulatoryValidator(api_key=API_KEY)
    simulator = EnvironmentSimulator(api_key=API_KEY)

    manager = DifferentiationManager(
        engine=engine,
        auditor=validator,
        environment_simulator=simulator,
        log_dir="logs/experiment_v1"
    )

    # 2. Define the "Environmental Stressor" (The Task Suite)
    # We start with a task that requires a tool the agent doesn't have yet.
    task_suite = TASKS

    print("=== STAGE 1: BASELINE (Stem Cell) ===")
    baseline_task = task_suite[0]
    print(f"[*] Task: {baseline_task}")

    initial_output = await agent.execute_task(baseline_task)
    initial_feedback = await simulator.evaluate(baseline_task, initial_output)

    print(f"Result: {'SUCCESS' if initial_feedback.success else 'FAILURE'}")
    print(f"Critique: {initial_feedback.critique}")

    # 3. STAGE 2: EVOLUTIONARY PRESSURE
    print("\n=== STAGE 2: INITIATING EVOLUTIONARY DIFFERENTIATION ===")
    evolved_agent = await manager.evolve_to_maturity(
        agent,
        task_suite=task_suite.copy(),
        max_generations=3
    )

    # 4. STAGE 3: POST-EVOLUTION EVALUATION
    print("\n=== STAGE 3: FINAL EVALUATION (Specialized Phenotype) ===")
    final_output = await evolved_agent.execute_task(baseline_task)
    final_feedback = await simulator.evaluate(baseline_task, final_output)

    # 5. FINAL COMPARISON
    print("\n" + "=" * 50)
    print("EXPERIMENT SUMMARY")
    print("=" * 50)
    print(f"Baseline (Gen 1) Success: {initial_feedback.success}")
    print(f"Evolved (Gen {evolved_agent.genome.version}) Success: {final_feedback.success}")
    print(f"New Capabilities: {[c.name for c in evolved_agent.genome.capabilities]}")
    print(f"Detailed logs saved to: {manager.log_dir}")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(run_experiment())
