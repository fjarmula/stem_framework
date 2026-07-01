import asyncio
import os
from dotenv import load_dotenv
from src.core.agent import StemAgent
from src.services.llm import LLMService

load_dotenv()


async def run_inference(genome_path: str, task: str):
    """
    Loads a specialized agent from a genome file and executes a task.
    No EvolutionEngine or DifferentiationManager is needed here.
    """
    try:
        llm = LLMService.from_config()
    except ValueError as exc:
        print(f"[!] Error: {exc}")
        return

    if not os.path.exists(genome_path):
        print(f"[!] Error: Genome file '{genome_path}' not found.")
        return
    print(f"[*] Awakening agent from {genome_path}...")
    agent = StemAgent.load_genome(genome_path, llm=llm)

    print(f"[*] Identity: {agent.genome.persona_name}")
    print(f"[*] Protocol: {agent.genome.reasoning_protocol}")
    print("-" * 30)
    print(f"USER: {task}")
    response, turns = await agent.execute_task(task)

    print("-" * 30)
    print(f"AGENT ({turns} turns):")
    print(response)


if __name__ == "__main__":
    mature = "mature_cell.json"
    stem = "stem_cell.json"
    default_task = """What the average of [10, 20, -20, 43, 21, 15],
    but only including numbers greater than 10."""

    user_task = str(input("[*] Enter task (press enter to continue with the default one): "))
    task = user_task if len(user_task.strip()) > 0 else default_task

    # asyncio.run(run_inference(stem, task))
    asyncio.run(run_inference(mature, task))
