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
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[!] Error: OPENAI_API_KEY not found.")
        return

    if not os.path.exists(genome_path):
        print(f"[!] Error: Genome file '{genome_path}' not found.")
        return
    llm = LLMService(api_key=api_key)
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
    target_genome = "mature_agent.json"
    default_task = """Write a Python function that takes a list of numbers and returns the average,
    but only include numbers greater than 10.
    Input: [10, 20, -20, 43, 21, 15]"""

    user_task = str(input("[*] Enter task: "))
    task = user_task if len(user_task.strip()) > 0 else default_task

    asyncio.run(run_inference(target_genome, task))
