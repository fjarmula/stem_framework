from typing import Type, Dict, List
from pydantic import BaseModel
from src.core.genome import AgentGenome, TransformationPlan, CapabilityModel
from src.evaluation.feedback import EnvironmentFeedback
from src.services.llm import LLMService
from src.services.prompts import PromptManager


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

        # kind of hint for the model which tools are available
        available_tools = """
        -python_interpreter: Allows the agent to execute Python code to solve math, 
         logic, or data processing tasks deterministically.
         Parameters: {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}
        """

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

        return AgentGenome(
            version=current_genome.version + 1,
            persona_name=plan.new_persona_name or current_genome.persona_name,
            role_description=plan.new_role_description or current_genome.role_description,
            reasoning_protocol=plan.modified_protocol or current_genome.reasoning_protocol,
            capabilities=new_capabilities,
            constraints=current_genome.constraints
        )
