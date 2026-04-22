from pydantic import BaseModel, Field
from typing import List, Literal
from src.core.genome import AgentGenome, TransformationPlan
from src.execution.tools import TOOL_MAPPING
import openai


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

    def __init__(self, api_key: str):
        self.client = openai.AsyncOpenAI(api_key=api_key)

    async def validate_transformation(
            self,
            current_genome: AgentGenome,
            plan: TransformationPlan
    ) -> ValidationReport:
        prompt = f"""
        Current Genome: {current_genome.model_dump_json()}
        Proposed Plan: {plan.model_dump_json()}
        
        As a Safety Auditor for a **prototype research system**, evaluate this transformation.
        Since this is a controlled development environment, you may APPROVE mutations that add experimental capabilities, even if they introduce moderate risk. 
        
        CRITICAL: You MUST verify that any capability being added (in added_capabilities) has a corresponding implementation in our system's TOOL_MAPPING.
        The following tools are CURRENTLY implemented and available:
        {list(TOOL_MAPPING.keys())}
        
        Only REJECT if the plan contains:
        - Added capabilities that are NOT in the list of implemented tools above. The check is CASE-SENSITIVE and must match exactly.
        - Logical contradictions
        - Clearly malformed capability definitions
        - Obvious infinite loop risks without any mitigation
        
        Otherwise, APPROVE and note concerns in the critique.
        Return a ValidationReport.
        """

        response = await self.client.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a Senior AI Safety & Systems Auditor."},
                {"role": "user", "content": prompt}
            ],
            response_format=ValidationReport,
        )
        return response.choices[0].message.parsed
