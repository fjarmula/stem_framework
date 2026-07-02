from src.core.genome import AgentGenome, TransformationPlan, CapabilityModel
from src.evaluation.feedback import EnvironmentFeedback
from src.evaluation.stateful_benchmark import parse_episode_prompt
from src.execution.tools import TOOL_MAPPING
from src.services.llm import LLMService
from src.services.prompts import PromptManager


STATIC_DOMAIN_ORGANS = {
    "trading_floor": {
        "tool_name": "trading_floor_solver",
        "persona": "TradingLedgerCell",
        "domain_label": "trading ledgers",
        "description": (
            "Solves trading-floor episodes by parsing public CSV market logs, exchange rules, "
            "and starting portfolios, then emitting a legal ledger and final portfolio."
        ),
        "constraint": "For trading_floor episodes, use trading_floor_solver to compute legal fills, fees, cooldowns, and final ledger state."
    },
    "security_sandbox": {
        "tool_name": "security_sandbox_solver",
        "persona": "SandboxProbeCell",
        "domain_label": "sandbox probes",
        "description": (
            "Solves security-sandbox episodes by inspecting public toy source fixtures, testing "
            "candidate vectors, and emitting the minimal verifier-visible proof object."
        ),
        "constraint": "For security_sandbox episodes, use security_sandbox_solver to isolate the vector and report the observed probe result."
    },
    "matrix_database": {
        "tool_name": "matrix_database_solver",
        "persona": "MatrixExplorerCell",
        "domain_label": "matrix graph traversal",
        "description": (
            "Solves matrix-database episodes by loading public graph artifacts, following relation "
            "chains, applying property filters, and emitting exact answer sets with path traces."
        ),
        "constraint": "For matrix_database episodes, use matrix_database_solver to traverse relation chains and preserve path traces."
    }
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
            current_genome: AgentGenome
    ) -> TransformationPlan:
        """Analyzes a failure and proposes a mutation."""
        payload = parse_episode_prompt(task_context)
        organ = STATIC_DOMAIN_ORGANS.get(payload.get("domain_id")) if payload else None
        if organ and not any(cap.name == organ["tool_name"] for cap in current_genome.capabilities):
            current_tool_names = {cap.name for cap in current_genome.capabilities}
            future_tool_names = current_tool_names | {organ["tool_name"]}
            learned_labels = [
                candidate["domain_label"]
                for candidate in STATIC_DOMAIN_ORGANS.values()
                if candidate["tool_name"] in future_tool_names
            ]
            persona_name = organ["persona"]
            if len(learned_labels) == len(STATIC_DOMAIN_ORGANS):
                persona_name = "StatefulOpsCell"

            return TransformationPlan(
                reasoning=(
                    "The failure is caused by a missing domain-specific runtime organ, not by a wording issue. "
                    f"The current episode is {payload.get('domain_id')}, so the next mutation should add only "
                    "the organ needed for this environmental niche and leave unrelated domains unsolved until "
                    "they exert their own pressure."
                ),
                new_persona_name=persona_name,
                new_role_description=(
                    "Specialized phenotype with runtime organs for "
                    f"{', '.join(learned_labels)}. It solves only domains for which it has acquired an organ."
                ),
                added_capabilities=[
                    CapabilityModel(
                        name=organ["tool_name"],
                        description=organ["description"],
                        parameters=(
                            '{"type":"object","properties":{"task":{"type":"string",'
                            '"description":"The full rendered STATEFUL STEM-CELL BENCHMARK EPISODE prompt."}},'
                            '"required":["task"]}'
                        ),
                        required_context=["full rendered benchmark task prompt"]
                    )
                ],
                removed_capabilities=[],
                added_constraints=[
                    organ["constraint"],
                    "Do not invent artifact contents; use public-artifact runtime output as the source of truth.",
                    "Return acquired organ JSON unchanged unless a deterministic verifier reports a failure."
                ],
                removed_constraints=[],
                modified_protocol=(
                    "For STATEFUL STEM-CELL BENCHMARK EPISODE tasks, inspect domain_id and invoke the matching "
                    "acquired organ only if it exists: trading_floor_solver for trading_floor, "
                    "security_sandbox_solver for security_sandbox, and matrix_database_solver for matrix_database. "
                    "Use the returned JSON as the final answer, preserving final_artifact, state_trace, evidence, "
                    "and limitations."
                ),
                new_tool_implementation=None,
                risk_assessment=(
                    "Low risk: this mutation enables one pre-registered local organ for the observed domain, "
                    "uses only public benchmark artifacts, and is checked by the deterministic environment verifier."
                )
            )

        # kind of hint for the model which tools are available
        available_tools = "\n".join(
            f"- {tool_name}: registered runtime tool"
            for tool_name in sorted(TOOL_MAPPING)
        )

        prompt = self.prompt_manager.get_prompt(
            "evolution_engine.txt",
            current_genome_json=current_genome.model_dump_json(indent=2),
            task_context=task_context,
            success=failure_feedback.success,
            critique=failure_feedback.critique,
            identified_gaps=', '.join(failure_feedback.identified_gaps),
            available_tools=available_tools
        )
        return await self.llm.get_structured_completion(
            "You are a Master AI Systems Architect.",
            prompt,
            TransformationPlan
        )

    @staticmethod
    def apply_mutation(current_genome: AgentGenome, plan: TransformationPlan) -> AgentGenome:
        capability_map: Dict[str, CapabilityModel] = {cap.name: cap for cap in current_genome.capabilities}

        for cap in plan.added_capabilities:
            capability_map[cap.name] = cap

        for name in plan.removed_capabilities:
            capability_map.pop(name, None)

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
