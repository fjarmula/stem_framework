from pydantic import BaseModel, Field
from typing import List, Literal
from src.core.genome import AgentGenome, TransformationPlan
from src.execution.tools import TOOL_MAPPING
from src.services.llm import LLMService
from src.services.prompts import PromptManager


class ValidationReport(BaseModel):
    """The result of a safety and consistency check."""
    is_safe: bool
    consistency_score: int = Field(ge=0, le=100)
    identified_risks: List[str]
    verdict: Literal["APPROVE", "REJECT", "REQUIRE_FIXES"]
    critique: str


class RegulatoryValidator:
    """
    Acts as the 'Immune System'. Validates mutations before they are applied.
    """

    def __init__(self, llm: LLMService, prompt_manager: PromptManager):
        self.llm = llm
        self.prompt_manager = prompt_manager

    async def validate_transformation(
            self,
            current_genome: AgentGenome,
            plan: TransformationPlan
    ) -> ValidationReport:
        prompt = self.prompt_manager.get_prompt(
            "safety_validator.txt",
            current_genome=current_genome.model_dump_json(),
            plan=plan.model_dump_json(),
            available_tools=list(TOOL_MAPPING.keys())
        )

        return await self.llm.get_structured_completion(
            "You are a Senior AI Safety & Systems Auditor.",
            prompt,
            ValidationReport
        )
