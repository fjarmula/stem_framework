import asyncio
from pathlib import Path
from typing import Iterable, Optional

from dotenv import load_dotenv

from src.cli import parse_inference_args, task_label
from src.core.agent import StemAgent
from src.evaluation.feedback import EnvironmentFeedback
from src.evaluation.simulator import EnvironmentSimulator
from src.evaluation.stateful_benchmark import (
    format_stateful_output,
    verify_stateful_episode,
)
from src.services.llm import LLMRateLimitError, LLMService
from src.services.prompts import PromptManager
from src.services.task_loader import TaskLoader

load_dotenv()


def _load_optional_llm() -> Optional[LLMService]:
    try:
        return LLMService.from_config()
    except ValueError as exc:
        print(f"[!] LLM unavailable: {exc}")
        print("[*] Continuing; acquired deterministic organs can still run benchmark episodes.")
        return None


def _benchmark_tasks(split: str) -> Iterable[str]:
    loader = TaskLoader()
    if split == "evolution":
        return loader.evolution_tasks
    if split == "validation":
        return loader.validation_tasks
    return [*loader.evolution_tasks, *loader.validation_tasks]


def _print_verified_output(
    task: str,
    output: str,
    verify: bool,
    turns_taken: int,
    feedback: Optional[EnvironmentFeedback] = None,
) -> None:
    print("OUTPUT:")
    print(format_stateful_output(output))
    if not verify:
        return

    feedback = feedback or verify_stateful_episode(task, output, turns_taken=turns_taken)
    if feedback is None:
        print("VERIFIER: unavailable for this task")
        return

    status = "PASS" if feedback.success else "FAIL"
    print(f"VERIFIER: {status}")
    print(f"CRITIQUE: {feedback.critique}")
    if feedback.identified_gaps:
        print(f"GAPS: {feedback.identified_gaps}")


async def run_single_task(genome_path: str, task: str, verify: bool) -> None:
    llm = _load_optional_llm()
    agent = StemAgent.load_genome(genome_path, llm=llm)
    simulator = EnvironmentSimulator(llm=llm, prompt_manager=PromptManager())

    print(f"[*] Identity: {agent.genome.persona_name}")
    print(f"[*] Capabilities: {[cap.name for cap in agent.genome.capabilities]}")
    print("-" * 50)
    print(f"TASK: {task_label(task)}")
    output, turns, feedback = await simulator.evaluate_agent(agent, task)
    print(f"TURNS: {turns}")
    _print_verified_output(task, output, verify, turns, feedback)


async def run_benchmark(genome_path: str, split: str, verify: bool) -> None:
    llm = _load_optional_llm()
    agent = StemAgent.load_genome(genome_path, llm=llm)
    simulator = EnvironmentSimulator(llm=llm, prompt_manager=PromptManager())
    tasks = list(_benchmark_tasks(split))

    print(f"[*] Identity: {agent.genome.persona_name}")
    print(f"[*] Capabilities: {[cap.name for cap in agent.genome.capabilities]}")
    print(f"[*] Benchmark split: {split}")
    print(f"[*] Episodes: {len(tasks)}")

    passed = 0
    for index, task in enumerate(tasks, start=1):
        print("\n" + "=" * 50)
        print(f"EPISODE {index}/{len(tasks)}: {task_label(task)}")
        output, turns, feedback = await simulator.evaluate_agent(agent, task)
        print(f"TURNS: {turns}")
        _print_verified_output(task, output, verify, turns, feedback)

        if feedback and feedback.success:
            passed += 1

    if verify:
        print("\n" + "=" * 50)
        print(f"VERIFIED PASS RATE: {passed}/{len(tasks)}")


async def main() -> None:
    args = parse_inference_args()
    genome_path = Path(args.genome)
    if not genome_path.exists():
        print(f"[!] Error: Genome file '{genome_path}' not found.")
        return

    verify = not args.no_verify
    if args.benchmark:
        await run_benchmark(str(genome_path), args.benchmark, verify)
        return

    task = args.task or ""
    if args.task_file:
        task = Path(args.task_file).read_text(encoding="utf-8")
    if not task:
        task = input("[*] Enter task: ").strip()
    if not task:
        print("[!] No task provided.")
        return

    await run_single_task(str(genome_path), task, verify)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except LLMRateLimitError as exc:
        print("\n[!] LLM provider quota/rate limit reached.")
        print(f"    {exc}")
