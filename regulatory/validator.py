from pydantic import BaseModel, Field
from typing import List, Literal
from core.genome import AgentGenome, TransformationPlan
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
        self.client = openai.OpenAI(api_key=api_key)

    async def validate_transformation(
            self,
            current_genome: AgentGenome,
            plan: TransformationPlan
    ) -> ValidationReport:
        """
        Critiques a proposed evolution plan for logical consistency and safety.
        """
        prompt = f"""
        Current Genome: {current_genome.model_dump_json()}
        Proposed Plan: {plan.model_dump_json()}

        As a Safety Auditor, evaluate this transformation.
        - Does the 'Modified Protocol' contradict the 'Constraints'?
        - Are the 'Added Capabilities' actually useful for the task?
        - Is there a risk of infinite loops or hallucinatory tool usage?

        Return a ValidationReport.
        """

        response = self.client.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a Senior AI Safety & Systems Auditor."},
                {"role": "user", "content": prompt}
            ],
            response_format=ValidationReport,
        )
        return response.choices[0].message.parsed
