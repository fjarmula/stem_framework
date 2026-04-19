# main.py
import asyncio
from core.stem import StemAgent
from evolution.engine import EvolutionEngine
from evolution.manager import DifferentiationManager
from regulatory.validator import RegulatoryValidator
from evaluation.simulator import EnvironmentSimulator
from tasks import TASKS
from dotenv import load_dotenv
import os


async def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    # Initialize components
    agent = StemAgent(api_key=api_key)
    engine = EvolutionEngine(api_key=api_key)
    validator = RegulatoryValidator(api_key=api_key)
    env_sim = EnvironmentSimulator(api_key=api_key)

    manager = DifferentiationManager(engine, validator, env_sim)

    # Record initial performance (before evolution)
    print("=== INITIAL AGENT EVALUATION ===")
    initial_results = []
    for task in TASKS:
        output = await agent.execute_task(task)
        feedback = await env_sim.evaluate(task, output)
        initial_results.append({"task": task, "success": feedback.success, "critique": feedback.critique})

    # Evolve the agent
    print("\n=== STARTING EVOLUTION ===")
    evolved_agent = await manager.evolve_to_maturity(agent, TASKS.copy(), max_generations=5)

    # Record final performance (after evolution)
    print("\n=== EVOLVED AGENT EVALUATION ===")
    final_results = []
    for task in TASKS:
        output = await evolved_agent.execute_task(task)
        feedback = await env_sim.evaluate(task, output)
        final_results.append({"task": task, "success": feedback.success, "critique": feedback.critique})

    # Print comparison
    print("\n=== BEFORE vs AFTER ===")
    for i, task in enumerate(TASKS):
        print(f"Task {i + 1}: {task[:50]}...")
        print(f"  Before: {'Success' if initial_results[i]['success'] else 'Failure'}")
        print(f"  After:  {'Success' if final_results[i]['success'] else 'Failure'}")
        if not final_results[i]['success']:
            print(f"  Critique: {final_results[i]['critique']}")


if __name__ == "__main__":
    asyncio.run(main())
