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
from src.evaluation.stateful_benchmark import format_stateful_output, parse_episode_prompt
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

    @staticmethod
    def _task_label(task: str) -> str:
        """Return a compact label for large benchmark episode prompts."""
        payload = parse_episode_prompt(task)
        if payload:
            return str(payload.get("episode_id", "unknown_episode"))
        return task[:80]

    def _log_step(
        self,
        generation: int,
        task: str,
        output: str,
        feedback: EnvironmentFeedback,
        genome: "AgentGenome",
        turns_taken: int,
    ):
        """Saves a record of the current generation for the report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gen_path = os.path.join(self.log_dir, f"gen_{generation}_{timestamp}")
        os.makedirs(gen_path, exist_ok=True)

        with open(os.path.join(gen_path, "genome.json"), "w") as f:
            f.write(genome.model_dump_json(indent=2))

        trace = {
            "task": task,
            "agent_output": output,
            "turns_taken": turns_taken,
            "success": feedback.success,
            "critique": feedback.critique,
            "identified_gaps": feedback.identified_gaps,
        }

        with open(os.path.join(gen_path, "trace.json"), "w") as f:
            json.dump(trace, f, indent=2)

        print(f"[*] Logs saved to {gen_path}")

    @staticmethod
    def _print_success_output(task: str, output: str) -> None:
        if parse_episode_prompt(task) is None:
            return
        print("[+] Accepted artifact:")
        print(format_stateful_output(output))

    @staticmethod
    def _print_mutation_plan(plan: TransformationPlan) -> None:
        added = [cap.name for cap in plan.added_capabilities]
        removed = plan.removed_capabilities

        print("[*] Proposed structural mutation:")
        print(f"    Causal diagnosis: {plan.reasoning}")
        if plan.new_persona_name:
            print(f"    New phenotype: {plan.new_persona_name}")
        if added:
            print(f"    Added capability: {', '.join(added)}")
        if removed:
            print(f"    Removed capability: {', '.join(removed)}")
        if plan.added_constraints:
            print("    New survival constraints:")
            for constraint in plan.added_constraints:
                print(f"      - {constraint}")
        print(f"    Risk assessment: {plan.risk_assessment}")

    @staticmethod
    def _print_state_trace(output: str) -> None:
        try:
            payload = json.loads(output)
        except json.JSONDecodeError:
            return

        state_trace = payload.get("state_trace")
        if not isinstance(state_trace, list) or not state_trace:
            return

        print("[*] Structured state process:")
        for step in state_trace:
            print(f"    {step}")

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
        mutation_rejection_feedback = ""

        # work on a copy to avoid modifying the caller's list unexpectedly
        remaining_tasks = task_suite.copy()

        while epoch <= max_epochs and remaining_tasks:
            print(f"\n[Epoch {epoch}] Current Phenotype: {agent.genome.persona_name}")

            current_task = remaining_tasks[0]
            print(f"[*] Attempting task: {self._task_label(current_task)}...")
            attempt_output, turns = await agent.execute_task(current_task)
            feedback = await self.env.evaluate(current_task, attempt_output, turns_taken=turns)

            self._log_step(generation, current_task, attempt_output, feedback, agent.genome, turns)
            self.metrics.record(feedback.success, is_stem=agent.genome.version == 1)
            print(f"[*] Structured turns observed: {turns}")

            if feedback.success:
                print("[✓] Task successful in current state.")
                self._print_success_output(current_task, attempt_output)
                remaining_tasks.pop(0)
            else:
                print(f"[!] Task failed. Pressure applied: {feedback.identified_gaps}")
                plan = await self.engine.propose_differentiation(
                    task_context=current_task,
                    failure_feedback=feedback,
                    current_genome=agent.genome,
                    failed_output=attempt_output,
                    mutation_rejection_feedback=mutation_rejection_feedback
                )
                self._print_mutation_plan(plan)
                self._preserve_other_domain_organs(agent, current_task, plan)

                if parse_episode_prompt(current_task) is not None and not plan.new_tool_implementation:
                    mutation_rejection_feedback = (
                        "Stateful benchmark pressure requires a generated runtime organ in "
                        "new_tool_implementation."
                    )
                    print(
                        "[-] Mutation rejected: stateful benchmark pressure requires "
                        "a generated runtime organ in new_tool_implementation."
                    )
                    generation += 1
                    epoch += 1
                    continue

                report = await self.auditor.validate_transformation(agent.genome, plan)
                print(
                    "[*] Immune-system verdict: "
                    f"{report.verdict} ({report.consistency_score}/100) - {report.critique}"
                )

                if report.verdict == "APPROVE":
                    mutation_rejection_feedback = ""
                    if not self._compile_generated_tool(plan):
                        mutation_rejection_feedback = "Generated organ failed to compile or register."
                        generation += 1
                        epoch += 1
                        continue

                    new_genome = self.engine.apply_mutation(agent.genome, plan)
                    agent.update_genome(new_genome)
                    print(f"[+] Evolved new traits to survive environment.")
                    post_mutation, post_turns = await agent.execute_task(current_task)
                    post_feedback = await self.env.evaluate(current_task, post_mutation, turns_taken=post_turns)

                    # log and record the post-mutation attempt
                    generation += 1
                    self._log_step(generation, current_task, post_mutation, post_feedback, agent.genome, post_turns)
                    self.metrics.record(post_feedback.success, is_stem=agent.genome.version == 1)
                    print(f"[*] Structured turns observed: {post_turns}")
                    self._print_state_trace(post_mutation)

                    if post_feedback.success:
                        print(f"[+] Transformation verified. Phenotype stabilized at version {agent.genome.version}")
                        self._print_success_output(current_task, post_mutation)
                        remaining_tasks.pop(0)
                    elif rollback:
                        print(f"[!] Mutation failed to solve the problem. Initiating rollback.")
                        agent.rollback()
                else:
                    mutation_rejection_feedback = report.critique
                    print(f"[-] Mutation rejected by immune system: {report.critique}")

            generation += 1
            epoch += 1

        if remaining_tasks:
            remaining = [self._task_label(task) for task in remaining_tasks]
            print(f"\n[!] Evolution stopped before maturity. Remaining tasks: {remaining}")
        elif agent.genome.version > 1:
            print(f"\n[✓] Evolution complete. Specializing phenotype name...")
            if agent.genome.persona_name == "StemCell":
                agent.genome.rename("Specialized Cell")
            print(f"[*] Final Identity: {agent.genome.persona_name}")

        return agent

    @staticmethod
    def _preserve_other_domain_organs(agent: StemAgent, task: str, plan: TransformationPlan) -> None:
        payload = parse_episode_prompt(task)
        if payload is None or not plan.removed_capabilities:
            return

        current_marker = f"domain_id:{payload.get('domain_id')}"
        protected = set()
        for capability in agent.genome.capabilities:
            domain_contexts = [
                context
                for context in capability.required_context
                if context.startswith("domain_id:")
            ]
            if domain_contexts and current_marker not in domain_contexts:
                protected.add(capability.name)

        if not protected:
            return

        before = list(plan.removed_capabilities)
        plan.removed_capabilities = [
            name for name in plan.removed_capabilities if name not in protected
        ]
        restored = [name for name in before if name not in plan.removed_capabilities]
        if restored:
            print(f"[*] Preserving previously acquired organs: {restored}")
