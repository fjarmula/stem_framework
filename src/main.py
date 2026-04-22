import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv
from src.core.agent import StemAgent
from src.evolution.engine import EvolutionEngine
from src.evolution.manager import DifferentiationManager
from src.regulatory.validator import RegulatoryValidator
from src.evaluation.simulator import EnvironmentSimulator
from src.services.llm import LLMService
from src.services.prompts import PromptManager
from src.services.task_loader import TaskLoader
from src.config import config

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")


async def run_experiment():
    if not API_KEY:
        print("[!] Error: OPENAI_API_KEY not found in .env file.")
        return

    llm = LLMService(api_key=API_KEY)
    prompt_manager = PromptManager()
    loader = TaskLoader()
    evolution_tasks = loader.evolution_tasks

    if not loader.validation_tasks:
        print("[!] Error: No validation tasks found in tasks.yaml")
        return
    test_task = loader.validation_tasks[0]

    agent = StemAgent(llm=llm)
    engine = EvolutionEngine(llm=llm, prompt_manager=prompt_manager)
    validator = RegulatoryValidator(llm=llm, prompt_manager=prompt_manager)
    simulator = EnvironmentSimulator(llm=llm, prompt_manager=prompt_manager)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = config["logging"]["base_dir"]
    experiment_log_dir = f"{base_log_dir}/experiment_{timestamp}"

    manager = DifferentiationManager(
        engine=engine,
        auditor=validator,
        environment_simulator=simulator,
        log_dir=experiment_log_dir
    )

    print("=== STAGE 1: BASELINE (Stem Cell) ===")
    print(f"[*] Task: {test_task}")

    initial_output, _ = await agent.execute_task(test_task)
    initial_feedback = await simulator.evaluate(test_task, initial_output)

    print(f"Result: {'SUCCESS' if initial_feedback.success else 'FAILURE'}")
    print(f"Critique: {initial_feedback.critique}")

    print("\n=== STAGE 2: INITIATING EVOLUTIONARY DIFFERENTIATION ===")
    print(f"[*] Evolving on {len(evolution_tasks)} tasks...")

    evolved_agent = await manager.evolve_to_maturity(
        agent,
        task_suite=evolution_tasks,
        max_generations=config["evolution"]["max_generations"]
    )

    print("\n=== STAGE 3: FINAL EVALUATION (Specialized Phenotype) ===")
    print(f"[*] Re-testing: {test_task}")

    final_output, _ = await evolved_agent.execute_task(test_task)
    final_feedback = await simulator.evaluate(test_task, final_output)

    print(f"Result: {'SUCCESS' if final_feedback.success else 'FAILURE'}")
    
    print("\n" + "=" * 50)
    print("EXPERIMENT SUMMARY")
    print("=" * 50)
    print(f"TEST TASK: {test_task}")
    print("-" * 50)
    print(f"Baseline (Gen 1) Success: {initial_feedback.success}")
    print(f"Evolved (Gen {evolved_agent.genome.version}) Success: {final_feedback.success}")

    caps = [c.name for c in evolved_agent.genome.capabilities]
    print(f"Final Capabilities: {caps if caps else 'None (General Reasoning)'}")
    print(f"Final Protocol: {evolved_agent.genome.reasoning_protocol}")
    print(f"Detailed logs saved to: {manager.log_dir}")
    print("=" * 50)


if __name__ == "__main__":
    try:
        asyncio.run(run_experiment())
    except KeyboardInterrupt:
        print("\n[!] Experiment halted by user.")
