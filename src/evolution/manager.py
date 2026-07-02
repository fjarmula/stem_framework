import json
import os
from datetime import datetime
from pathlib import Path
from typing import List
from src.evolution.engine import EvolutionEngine
from src.regulatory.validator import RegulatoryValidator
from src.core.genome import AgentGenome, TransformationPlan
from src.core.agent import StemAgent
from src.evaluation.feedback import EnvironmentFeedback
from src.evaluation.simulator import EnvironmentSimulator
from src.evaluation.metrics import ExperimentMetrics
from src.execution.tools import register_compiled_skill


class DifferentiationManager:
    """
    Orchestrates the lifecycle of a StemAgent, using environmental pressure
    to force emergent specialization.
    """

    def __init__(self, engine: "EvolutionEngine", auditor: "RegulatoryValidator", environment_simulator: EnvironmentSimulator, log_dir: str = "logs"):
        self.engine = engine
        self.auditor = auditor
        self.env = environment_simulator  # A mock or real evaluation function
        self.log_dir = log_dir
        self.metrics = ExperimentMetrics()
        self.compiled_skills_dir = Path(__file__).resolve().parents[1] / "compiled_skills"

    def _log_step(self, generation: int, task: str, output: str, feedback: EnvironmentFeedback, genome: "AgentGenome"):
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

    def _compile_generated_tool(self, plan: TransformationPlan) -> bool:
        """Persist and register a generated runtime organ for the active session."""
        if not plan.new_tool_implementation:
            return True

        report = self.auditor.validate_generated_tool(plan)
        if report.verdict != "APPROVE":
            print(f"[-] Generated organ rejected by immune system: {report.critique}")
            return False

        tool_name = self.auditor.generated_tool_name(plan)
        if tool_name is None:
            print("[-] Generated organ rejected: unable to resolve generated tool name.")
            return False

        self.compiled_skills_dir.mkdir(parents=True, exist_ok=True)
        init_path = self.compiled_skills_dir / "__init__.py"
        init_path.touch(exist_ok=True)

        skill_path = self.compiled_skills_dir / f"{tool_name}.py"
        skill_path.write_text(plan.new_tool_implementation.rstrip() + "\n", encoding="utf-8")

        try:
            register_compiled_skill(tool_name, skill_path)
        except Exception as exc:
            print(f"[-] Generated organ failed to register: {exc}")
            return False

        print(f"[+] Generated organ compiled and registered: {tool_name}")
        return True

    async def evolve_to_maturity(self, agent: StemAgent, task_suite: List[str], max_epochs: int = 20, rollback=False) -> StemAgent:
        print(f"--- Initiating Emergent Evolution Sequence ---")

        generation = 1
        epoch = 1

        # work on a copy to avoid modifying the caller's list unexpectedly
        remaining_tasks = task_suite.copy()

        while epoch <= max_epochs and remaining_tasks:
            print(f"\n[Epoch {epoch}] Current Phenotype: {agent.genome.persona_name}")

            current_task = remaining_tasks[0]
            print(f"[*] Attempting task: {current_task[:50]}...")
            attempt_output, turns = await agent.execute_task(current_task)
            feedback = await self.env.evaluate(current_task, attempt_output)

            self._log_step(generation, current_task, attempt_output, feedback, agent.genome)
            self.metrics.record(feedback.success, is_stem=agent.genome.version == 1)

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
                    if not self._compile_generated_tool(plan):
                        generation += 1
                        epoch += 1
                        continue

                    new_genome = self.engine.apply_mutation(agent.genome, plan)
                    agent.update_genome(new_genome)
                    print(f"[+] Evolved new traits to survive environment.")
                    post_mutation, _ = await agent.execute_task(current_task)
                    post_feedback = await self.env.evaluate(current_task, post_mutation)

                    # log and record the post-mutation attempt
                    generation += 1
                    self._log_step(generation, current_task, post_mutation, post_feedback, agent.genome)
                    self.metrics.record(post_feedback.success, is_stem=agent.genome.version == 1)

                    if post_feedback.success:
                        print(f"[+] Transformation verified. Phenotype stabilized at version {agent.genome.version}")
                        remaining_tasks.pop(0)
                    elif rollback:
                        print(f"[!] Mutation failed to solve the problem. Initiating rollback.")
                        agent.rollback()
                else:
                    print(f"[-] Mutation rejected by immune system: {report.critique}")

            generation += 1
            epoch += 1

        if agent.genome.version > 1:
            print(f"\n[✓] Evolution complete. Specializing phenotype name...")
            if agent.genome.persona_name == "StemCell":
                agent.genome.rename("Specialized Cell")
            print(f"[*] Final Identity: {agent.genome.persona_name}")

        return agent
