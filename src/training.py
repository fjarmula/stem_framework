import asyncio
from datetime import datetime
from dotenv import load_dotenv
from src.core.agent import StemAgent
from src.evolution.engine import EvolutionEngine
from src.evolution.manager import DifferentiationManager
from src.regulatory.validator import RegulatoryValidator
from src.evaluation.simulator import EnvironmentSimulator
from src.evaluation.metrics import ExperimentMetrics
from src.services.llm import LLMService
from src.services.prompts import PromptManager
from src.services.task_loader import TaskLoader
from src.utils.config import config

load_dotenv()


async def run_experiment():
    try:
        llm = LLMService.from_config()
    except ValueError as exc:
        print(f"[!] Error: {exc}")
        return

    prompt_manager = PromptManager()
    loader = TaskLoader()
    evolution_tasks = loader.evolution_tasks
    validation_tasks = loader.validation_tasks

    if not validation_tasks:
        print("[!] Error: No validation tasks found in tasks.yaml")
        return

    agent = StemAgent(llm=llm)
    engine = EvolutionEngine(llm=llm, prompt_manager=prompt_manager)
    validator = RegulatoryValidator(llm=llm, prompt_manager=prompt_manager)
    simulator = EnvironmentSimulator(llm=llm, prompt_manager=prompt_manager)
    metrics = ExperimentMetrics()

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
    for task in validation_tasks:
        print(f"[*] Task: {task}")
        output, _ = await agent.execute_task(task)
        feedback = await simulator.evaluate(task, output)
        metrics.record(feedback.success, is_stem=True)
        print(f"    Result: {'SUCCESS' if feedback.success else 'FAILURE'}")
        print(f"    Critique: {feedback.critique}")

    print("\n=== STAGE 2: INITIATING EVOLUTIONARY DIFFERENTIATION ===")
    print(f"[*] Evolving on {len(evolution_tasks)} tasks...")

    evolved_agent = await manager.evolve_to_maturity(
        agent,
        task_suite=evolution_tasks,
        max_epochs=config["evolution"]["max_generations"]
    )

    print("\n=== STAGE 3: FINAL EVALUATION (Specialized Phenotype) ===")
    for task in validation_tasks:
        print(f"[*] Task: {task}")
        output, _ = await evolved_agent.execute_task(task)
        feedback = await simulator.evaluate(task, output)
        metrics.record(feedback.success, is_stem=False)
        print(f"    Result: {'SUCCESS' if feedback.success else 'FAILURE'}")
        print(f"    Critique: {feedback.critique}")

    # TODO - save the model only if it passes final evaluation with a certain threshold of success
    dna_filename = f"mature_cell.json"
    evolved_agent.save_genome(dna_filename)

    print("\n" + "=" * 50)
    print("EXPERIMENT SUMMARY")
    print("=" * 50)
    metrics.print_summary()
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
