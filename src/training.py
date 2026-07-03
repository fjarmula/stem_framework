import argparse
import asyncio
import traceback
from datetime import datetime
from dotenv import load_dotenv
from src.core.agent import StemAgent
from src.evolution.engine import EvolutionEngine
from src.evolution.manager import DifferentiationManager
from src.evaluation.feedback import EnvironmentFeedback
from src.regulatory.validator import RegulatoryValidator
from src.evaluation.simulator import EnvironmentSimulator
from src.evaluation.metrics import ExperimentMetrics
from src.evaluation.stateful_benchmark import format_stateful_output, parse_episode_prompt
from src.services.llm import LLMRateLimitError, LLMService
from src.services.prompts import PromptManager
from src.services.task_loader import TaskLoader
from src.utils.config import config

load_dotenv()


DOMAIN_ALIASES = {
    "all": None,
    "trading": "trading_floor",
    "trade": "trading_floor",
    "trading_floor": "trading_floor",
    "security": "security_sandbox",
    "security_sandbox": "security_sandbox",
    "matrix": "matrix_database",
    "matrix_database": "matrix_database",
}


def task_label(task: str) -> str:
    """Return a compact label for large benchmark episode prompts."""
    payload = parse_episode_prompt(task)
    if payload:
        return str(payload.get("episode_id", "unknown_episode"))
    return task[:80]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stem Cell evolutionary training.")
    parser.add_argument(
        "--domain",
        choices=sorted(DOMAIN_ALIASES),
        default="all",
        help=(
            "Restrict training/evaluation to one benchmark domain. "
            "Example: --domain trading trains on trade_001 and validates on trade_002."
        ),
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=config["evolution"]["max_generations"],
        help="Maximum evolution epochs for this run.",
    )
    return parser.parse_args()


def filter_tasks_by_domain(tasks: list[str], domain: str | None) -> list[str]:
    if domain is None:
        return tasks
    filtered = []
    for task in tasks:
        payload = parse_episode_prompt(task)
        if payload and payload.get("domain_id") == domain:
            filtered.append(task)
    return filtered


def compact_exception_details(exc: Exception) -> str:
    return "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__, limit=3)
    )


async def evaluate_for_cli(
    simulator: EnvironmentSimulator,
    agent: StemAgent,
    task: str,
) -> tuple[str, int, EnvironmentFeedback]:
    try:
        return await simulator.evaluate_agent(agent, task)
    except Exception as exc:
        return (
            compact_exception_details(exc),
            0,
            EnvironmentFeedback(
                success=False,
                critique=(
                    "Evaluation raised a runtime exception from the active "
                    f"phenotype: {type(exc).__name__}."
                ),
                identified_gaps=["runtime_exception", "generated_organ_crash"],
            ),
        )


async def run_experiment(
    domain_filter: str | None = None,
    max_epochs: int = config["evolution"]["max_generations"],
):
    try:
        llm = LLMService.from_config()
    except ValueError as exc:
        print(f"[!] Error: {exc}")
        return

    prompt_manager = PromptManager()
    loader = TaskLoader()
    evolution_tasks = filter_tasks_by_domain(loader.evolution_tasks, domain_filter)
    validation_tasks = filter_tasks_by_domain(loader.validation_tasks, domain_filter)

    if not evolution_tasks or not validation_tasks:
        domain_label = domain_filter or "all domains"
        print(
            f"[!] Error: Missing train/validation tasks for {domain_label} "
            f"in {config['experiments']['dir']}"
        )
        return

    print(f"[*] Loaded benchmark: {loader.benchmark_name}")
    print(f"[*] Domain filter: {domain_filter or 'all'}")
    print(f"[*] Evolution episodes: {len(evolution_tasks)}")
    print(f"[*] Validation episodes: {len(validation_tasks)}")
    print(f"[*] Max evolution epochs: {max_epochs}")
    print("[*] Evolution mode: LLM-generated runtime organs with deterministic verifier pressure.")

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
        print(f"[*] Task: {task_label(task)}")
        output, turns, feedback = await evaluate_for_cli(simulator, agent, task)
        metrics.record(feedback.success, is_stem=True)
        print(f"    Turns: {turns}")
        print(f"    Result: {'SUCCESS' if feedback.success else 'FAILURE'}")
        print(f"    Critique: {feedback.critique}")
        if feedback.success and parse_episode_prompt(task) is not None:
            print("    Accepted artifact:")
            print(format_stateful_output(output))

    print("\n=== STAGE 2: INITIATING EVOLUTIONARY DIFFERENTIATION ===")
    print(f"[*] Evolving on {len(evolution_tasks)} tasks...")

    evolved_agent = await manager.evolve_to_maturity(
        agent,
        task_suite=evolution_tasks,
        max_epochs=max_epochs,
    )

    print("\n=== STAGE 3: FINAL EVALUATION (Specialized Phenotype) ===")
    final_passes = 0
    for task in validation_tasks:
        print(f"[*] Task: {task_label(task)}")
        output, turns, feedback = await evaluate_for_cli(simulator, evolved_agent, task)
        metrics.record(feedback.success, is_stem=False)
        final_passes += int(feedback.success)
        print(f"    Turns: {turns}")
        print(f"    Result: {'SUCCESS' if feedback.success else 'FAILURE'}")
        print(f"    Critique: {feedback.critique}")
        if feedback.success and parse_episode_prompt(task) is not None:
            print("    Accepted artifact:")
            print(format_stateful_output(output))

    dna_filename = f"mature_cell.json"
    if final_passes == len(validation_tasks):
        evolved_agent.save_genome(dna_filename)
    else:
        print(
            f"[!] Mature genome not saved: final validation passed "
            f"{final_passes}/{len(validation_tasks)} episodes."
        )

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
        args = parse_args()
        asyncio.run(run_experiment(
            domain_filter=DOMAIN_ALIASES[args.domain],
            max_epochs=args.max_epochs,
        ))
    except LLMRateLimitError as exc:
        print("\n[!] LLM provider quota/rate limit reached.")
        print(f"    {exc}")
        print("    Wait for quota reset, reduce the benchmark size, or switch to a provider/model with higher limits.")
    except KeyboardInterrupt:
        print("\n[!] Experiment halted by user.")
