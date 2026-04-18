from typing import Type
from pydantic import BaseModel
from core.genome import AgentGenome, TransformationPlan
from models.validation import EnvironmentFeedback
import openai


class EvolutionEngine:
    """
    The 'Environmental Signal' that pushes agents to evolve.
    """

    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)

    def _generate_structured_completion(self, prompt: str, response_model: Type[BaseModel]) -> str:
        """Helper method to get structured responses (Pydantic regulatory) from the OpenAI API."""
        response = self.client.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a Master AI Systems Architect."},
                {"role": "user", "content": prompt}
            ],
            response_format=response_model,
        )
        return response.choices[0].message.parsed

    async def propose_differentiation(self,
                                      task_context: str,
                                      failure_feedback: EnvironmentFeedback,
                                      current_genome: AgentGenome
                                      ) -> TransformationPlan:
        """
        Analyzes the target task and proposes a mutation plan.
        """
        prompt = f"""
        Current Agent Genome:
        {current_genome.model_dump_json(indent=2)}

        Task Attempted:
        {task_context}

        Environmental Feedback:
        - Success: {failure_feedback.success}
        - Critique: {failure_feedback.critique}
        - Identified Capability Gaps: {', '.join(failure_feedback.identified_gaps)}

        As a Master AI Systems Architect, analyze why the agent failed at this task. Then propose a specific TransformationPlan that addresses the identified gaps.
        The plan must include:
        1. **Added Capabilities**: 2-3 new concrete tools or skills that directly remedy the gaps. For each, define a `name`, `description`, `parameters` (if any), and `required_context`.
        2. **Removed Capabilities**: List any existing capability names that are redundant or counterproductive.
        3. **Modified Reasoning Protocol**: A refined instruction that improves the agent's decision-making (e.g., "Step-by-step verification with self-critique").
        4. **Risk Assessment**: Describe potential downsides or new failure modes introduced by these changes.

        Return a TransformationPlan JSON object.
        """
        return self._generate_structured_completion(prompt, TransformationPlan)

    def apply_mutation(self, current_genome: AgentGenome, plan: TransformationPlan) -> AgentGenome:
        """
        Pure function to generate a new Genome based on the current one and the proposed transformation plan.
        This represents the actual differentiation step.
        """
        new_capabilities = current_genome.capabilities.copy()
        new_capabilities.extend(plan.added_capabilities)
        new_capabilities = [c for c in new_capabilities if c.name not in plan.removed_capabilities]
        return AgentGenome(
            version=current_genome.version + 1,
            persona_name=current_genome.persona_name,
            role_description=current_genome.role_description,
            reasoning_protocol=plan.modified_protocol or current_genome.reasoning_protocol,
            capabilities=new_capabilities,
            constraints=current_genome.constraints
        )
