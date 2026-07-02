from pathlib import Path

from src.core.genome import AgentGenome, TransformationPlan, CapabilityModel
from src.evaluation.feedback import EnvironmentFeedback
from src.evaluation.stateful_benchmark import parse_episode_prompt
from src.execution.tools import TOOL_MAPPING
from src.services.llm import LLMService
from src.services.prompts import PromptManager


DISABLED_STATIC_BENCHMARK_TOOLS = {
    "trading_floor_solver",
    "security_sandbox_solver",
    "matrix_database_solver",
}


class EvolutionEngine:
    """
    The 'Environmental Signal' that pushes agents to evolve.
    """

    def __init__(self, llm: LLMService, prompt_manager: PromptManager):
        self.llm = llm
        self.prompt_manager = prompt_manager

    async def propose_differentiation(
            self,
            task_context: str,
            failure_feedback: EnvironmentFeedback,
            current_genome: AgentGenome,
            failed_output: str = "",
            mutation_rejection_feedback: str = ""
    ) -> TransformationPlan:
        """Analyzes a failure and proposes a mutation."""
        payload = parse_episode_prompt(task_context)

        # Expose support tools, but hide prebuilt benchmark solvers so evolution
        # must synthesize and register a new runtime organ for benchmark pressure.
        available_tools = "\n".join(
            f"- {tool_name}: registered runtime tool"
            for tool_name in sorted(TOOL_MAPPING)
            if tool_name not in DISABLED_STATIC_BENCHMARK_TOOLS
        )
        if not available_tools:
            available_tools = "(none)"

        prompt = self.prompt_manager.get_prompt(
            "evolution_engine.txt",
            current_genome_json=current_genome.model_dump_json(indent=2),
            task_context=task_context,
            public_artifact_observations=self._public_artifact_observations(payload),
            current_generated_organs=self._current_generated_organs(current_genome),
            failed_output_excerpt=self._excerpt(failed_output),
            mutation_rejection_feedback=mutation_rejection_feedback or "(no previous mutation rejection)",
            success=failure_feedback.success,
            critique=failure_feedback.critique,
            identified_gaps=', '.join(failure_feedback.identified_gaps),
            available_tools=available_tools
        )
        plan = await self.llm.get_structured_completion(
            "You are a Master AI Systems Architect.",
            prompt,
            TransformationPlan
        )
        self._annotate_generated_capability_context(plan, payload)
        return plan

    @staticmethod
    def _annotate_generated_capability_context(
            plan: TransformationPlan,
            payload: dict | None
    ) -> None:
        """Attach benchmark routing metadata that generated code cannot infer later."""
        if payload is None or not plan.new_tool_implementation:
            return

        domain_id = payload.get("domain_id")
        if not domain_id:
            return

        domain_marker = f"domain_id:{domain_id}"
        for capability in plan.added_capabilities:
            if capability.name in TOOL_MAPPING:
                continue
            if domain_marker not in capability.required_context:
                capability.required_context.append(domain_marker)
            if "full rendered benchmark task prompt" not in capability.required_context:
                capability.required_context.append("full rendered benchmark task prompt")

    @staticmethod
    def _public_artifact_observations(payload: dict | None) -> str:
        """Expose public benchmark artifacts to the mutation designer."""
        if payload is None:
            return "(not a stateful benchmark episode)"

        artifacts = payload.get("public_artifacts", {})
        if not isinstance(artifacts, dict) or not artifacts:
            return "(no public artifacts listed)"

        observations = []
        for label, raw_path in artifacts.items():
            path = Path(str(raw_path))
            if path.is_dir():
                observations.append(f"## {label}: {path}")
                for child in sorted(path.rglob("*")):
                    if child.is_dir():
                        continue
                    observations.append(f"### {child}")
                    observations.append(EvolutionEngine._read_public_text(child))
            else:
                observations.append(f"## {label}: {path}")
                observations.append(EvolutionEngine._read_public_text(path))
        return "\n".join(observations)

    @staticmethod
    def _current_generated_organs(current_genome: AgentGenome) -> str:
        compiled_dir = Path(__file__).resolve().parents[1] / "compiled_skills"
        sections = []
        for capability in current_genome.capabilities:
            skill_path = compiled_dir / f"{capability.name}.py"
            if not skill_path.exists():
                continue
            sections.append(f"## {capability.name}: {skill_path}")
            sections.append(EvolutionEngine._read_public_text(skill_path, limit=12000))
        return "\n".join(sections) if sections else "(no generated organs in current genome)"

    @staticmethod
    def _read_public_text(path: Path, limit: int = 8000) -> str:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"(unable to read public artifact: {exc})"

        if len(text) > limit:
            return text[:limit] + "\n... [truncated]"
        return text

    @staticmethod
    def _excerpt(text: str, limit: int = 8000) -> str:
        if not text:
            return "(no failed output captured)"
        if len(text) > limit:
            return text[:limit] + "\n... [truncated]"
        return text

    @staticmethod
    def apply_mutation(current_genome: AgentGenome, plan: TransformationPlan) -> AgentGenome:
        capability_map: Dict[str, CapabilityModel] = {cap.name: cap for cap in current_genome.capabilities}

        for name in plan.removed_capabilities:
            capability_map.pop(name, None)

        for cap in plan.added_capabilities:
            capability_map[cap.name] = cap

        new_capabilities = list(capability_map.values())
        new_constraints = [
            constraint
            for constraint in current_genome.constraints
            if constraint not in plan.removed_constraints
        ]
        for constraint in plan.added_constraints:
            if constraint not in new_constraints:
                new_constraints.append(constraint)

        return AgentGenome(
            version=current_genome.version + 1,
            persona_name=plan.new_persona_name or current_genome.persona_name,
            role_description=plan.new_role_description or current_genome.role_description,
            reasoning_protocol=plan.modified_protocol or current_genome.reasoning_protocol,
            capabilities=new_capabilities,
            constraints=new_constraints
        )
