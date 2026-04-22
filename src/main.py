import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv
from tasks import TASKS, TEST_TASK
from src.core.agent import StemAgent
from src.evolution.engine import EvolutionEngine
from src.evolution.manager import DifferentiationManager
from src.regulatory.validator import RegulatoryValidator
from src.evaluation.simulator import EnvironmentSimulator
from src.config import config
from src.services.llm import LLMService
from src.services.prompts import PromptManager

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")


async def run_experiment():
    if not API_KEY:
        print("[!] Error: OPENAI_API_KEY not found in .env file.")
        return

    llm = LLMService(api_key=API_KEY)
    prompt_manager = PromptManager()

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

    task_suite = TASKS.copy()  # make a copy to avoid modifying the original list
    test_task = TEST_TASK
    print("=== STAGE 1: BASELINE (Stem Cell) ===")
    print(f"[*] Task: {test_task}")

    initial_output = await agent.execute_task(test_task)
    initial_feedback = await simulator.evaluate(test_task, initial_output)

    print(f"Result: {'SUCCESS' if initial_feedback.success else 'FAILURE'}")
    print(f"Critique: {initial_feedback.critique}")

    print("\n=== STAGE 2: INITIATING EVOLUTIONARY DIFFERENTIATION ===")
    evolved_agent = await manager.evolve_to_maturity(
        agent,
        task_suite=task_suite.copy(),
        max_generations=20
    )

    print("\n=== STAGE 3: FINAL EVALUATION (Specialized Phenotype) ===")
    # re-test the test task to see if the evolved agent now passes it
    final_output = await evolved_agent.execute_task(test_task)
    final_feedback = await simulator.evaluate(test_task, final_output)

    print(f"Result: {'SUCCESS' if final_feedback.success else 'FAILURE'}")
    print("\n" + "=" * 50)
    print("EXPERIMENT SUMMARY")
    print(f"TEST TASK: {test_task}")
    print("=" * 50)
    print(f"Baseline (Gen 1) Success: {initial_feedback.success}")
    print(f"Evolved (Gen {evolved_agent.genome.version}) Success: {final_feedback.success}")

    caps = [c.name for c in evolved_agent.genome.capabilities]
    print(f"Final Capabilities: {caps if caps else 'None (General Reasoning)'}")
    print(f"Final Protocol: {evolved_agent.genome.reasoning_protocol}")
    print(f"Detailed logs saved to: {manager.log_dir}")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(run_experiment())
