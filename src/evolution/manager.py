import json
import os
from datetime import datetime
from typing import List
from src.core.agent import StemAgent
from src.evaluation.feedback import EnvironmentFeedback


class DifferentiationManager:
    """
    Orchestrates the lifecycle of a StemAgent, using environmental pressure
    to force emergent specialization.
    """

    def __init__(self, engine, auditor, environment_simulator, log_dir="logs"):
        self.engine = engine
        self.auditor = auditor
        self.env = environment_simulator  # A mock or real evaluation function
        self.log_dir = log_dir

    def _log_step(self, generation: int, task: str, output: str, feedback: EnvironmentFeedback, genome):
        """Saves a record of the current generation for the report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gen_path = os.path.join(self.log_dir, f"gen_{generation}_{timestamp}")
        os.makedirs(gen_path, exist_ok=True)

        with open(os.path.join(gen_path, "genome.json"), "w") as f:
            f.write(genome.model_dump_json(indent=2))

        trace = {
            "task": task,
            "agent_output": output,
            "success": feedback.success,
            "critique": feedback.critique,
            "identified_gaps": feedback.identified_gaps,
        }

        with open(os.path.join(gen_path, "trace.json"), "w") as f:
            json.dump(trace, f, indent=2)

        print(f"[*] Logs saved to {gen_path}")

    async def evolve_to_maturity(self, agent: StemAgent, task_suite: List[str], max_generations: int = 20) -> StemAgent:
        print(f"--- Initiating Emergent Evolution Sequence ---")

        generation = 1

        # work on a copy to avoid modifying the caller's list unexpectedly
        remaining_tasks = task_suite.copy()

        while generation <= max_generations and remaining_tasks:
            print(f"\n[Epoch {generation}] Current Phenotype: {agent.genome.persona_name}")

            current_task = remaining_tasks[0]
            print(f"[*] Attempting task: {current_task[:50]}...")
            attempt_output, turns = await agent.execute_task(current_task)
            feedback = await self.env.evaluate(current_task, attempt_output)

            self._log_step(generation, current_task, attempt_output, feedback, agent.genome)

            if feedback.success:
                print("[✓] Task successful in current state.")
                remaining_tasks.pop(0)
            else:
                print(f"[!] Task failed. Pressure applied: {feedback.identified_gaps}")
                plan = await self.engine.propose_differentiation(
                    task_context=current_task,
                    failure_feedback=feedback,
                    current_genome=agent.genome
                )

                report = await self.auditor.validate_transformation(agent.genome, plan)

                if report.verdict == "APPROVE":
                    new_genome = self.engine.apply_mutation(agent.genome, plan)
                    agent.update_genome(new_genome)
                    print(f"[+] Evolved new traits to survive environment.")
                else:
                    print(f"[-] Mutation rejected by immune system: {report.critique}")

            generation += 1

        return agent
