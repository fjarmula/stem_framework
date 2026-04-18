from typing import Type, Optional
from pydantic import BaseModel
from models.genome import AgentGenome, TransformationPlan
import openai
import asyncio

class EvolutionEngine:
    """
    The 'Environmental Signal' that pushes agents to evolve.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key

    def _generate_structured_completion(self, prompt: str, response_model: Type[BaseModel]) -> str:
        """Helper method to get structured responses (Pydantic models) from the OpenAI API."""
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
                                    task_class: str,
                                      current_genome: AgentGenome
                                      )->TransformationPlan:
        """
        Analyzes the target task and proposes a mutation plan.
        """
        prompt = f"""
                Current Agent State: {current_genome.model_dump_json()}
                Target Task Class: {task_class}

                Analyze the workflows required for '{task_class}'. 
                1. Identify 2-3 specific capabilities (tools/skills) this agent needs.
                2. Propose a refined reasoning protocol (e.g., 'Step-by-step verification').
                3. Identify potential risks in this transformation.

                Provide a TransformationPlan to move the agent toward this specialized state.
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


