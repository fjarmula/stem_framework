import asyncio
from core.genome import AgentGenome
from core.stem import StemAgent
from models.validation import EnvironmentFeedback
from typing import List


class DifferentiationManager:
    """
    Orchestrates the lifecycle of a StemAgent, using environmental pressure
    to force emergent specialization.
    """

    def __init__(self, engine, auditor, environment_simulator):
        self.engine = engine
        self.auditor = auditor
        self.env = environment_simulator  # A mock or real evaluation function

    async def evolve_to_maturity(self, agent: StemAgent, task_suite: List[str], max_generations: int = 5) -> StemAgent:
        print(f"--- Initiating Emergent Evolution Sequence ---")

        generation = 1
        consecutive_successes = 0

        while generation <= max_generations:
            print(f"\n[Generation {generation}] Current Phenotype: {agent.genome.persona_name}")

            # 1. Environmental Pressure: Agent attempts the current task
            current_task = task_suite[0]  # Pick the first unsolved task
            print(f"[*] Attempting task: {current_task[:50]}...")
            attempt_output = await agent.execute_task(current_task)

            # 2. Environmental Feedback: Did it survive/succeed?
            feedback: EnvironmentFeedback = await self.env.evaluate(current_task, attempt_output)

            if feedback.success:
                print("[✓] Task successful in current state.")
                consecutive_successes += 1
                task_suite.pop(0)  # Move to next task

                # Convergence/Maturity Check
                if not task_suite or consecutive_successes >= 3:
                    print("\n[★★★] Agent has reached Maturity. Environment conquered.")
                    break
                continue

            # 3. Adaptation phase (If task failed)
            print(f"[!] Task failed. Pressure applied: {feedback.identified_gaps}")
            consecutive_successes = 0

            # Engine drafts a mutation specifically to solve the identified gaps
            plan = await self.engine.propose_differentiation(
                task_context=current_task,
                failure_feedback=feedback,
                current_genome=agent.genome
            )

            # 4. Regulatory Check
            report = await self.auditor.validate_transformation(agent.genome, plan)

            if report.verdict == "APPROVE":
                # 5. Apply Mutation (Growth)
                new_genome = self.engine.apply_mutation(agent.genome, plan)
                agent.update_genome(new_genome)
                print(f"[+] Evolved new traits to survive environment.")
            else:
                print(f"[-] Mutation rejected by immune system: {report.critique}")

            generation += 1

        return agent
