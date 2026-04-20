from typing import Type
from pydantic import BaseModel
from core.genome import AgentGenome, TransformationPlan
from evaluation.validation import EnvironmentFeedback
import openai


class EvolutionEngine:
    """
    The 'Environmental Signal' that pushes agents to evolve.
    """

    def __init__(self, api_key: str):
        self.client = openai.AsyncOpenAI(api_key=api_key)

    async def _generate_structured_completion(
            self, prompt: str, response_model: Type[BaseModel]
    ) -> BaseModel:
        """Helper method to get structured responses (Pydantic regulatory) from the OpenAI API."""
        response = await self.client.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a Master AI Systems Architect."},
                {"role": "user", "content": prompt}
            ],
            response_format=response_model,
        )
        return response.choices[0].message.parsed

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

        prompt = f"""
        Current Agent Genome:
        {current_genome.model_dump_json(indent=2)}

        Task Attempted:
        {task_context}

        Environmental Feedback:
        - Success: {failure_feedback.success}
        - Critique: {failure_feedback.critique}
        - Identified Capability Gaps: {', '.join(failure_feedback.identified_gaps)}
        
        Available Physical Tools:
        {available_tools}

       INSTRUCTIONS:
        1. Analyze the failure. If the critique suggests a lack of deterministic logic or calculation errors, propose adding the 'python_interpreter'.
        2. Evolution should be MINIMAL. Do not add tools unless they are strictly necessary to solve the identified gaps.
        3. Propose a 'Modified Reasoning Protocol' that MANDATES the agent to:
           a) Use the python_interpreter.
           b) ALWAYS include the exact code used inside a ```python ``` block in the final response so the environment can verify it.
        
        Return a TransformationPlan.
        """
        return await self._generate_structured_completion(prompt, TransformationPlan)

    def apply_mutation(self, current_genome: AgentGenome, plan: TransformationPlan) -> AgentGenome:
        capability_map = {cap.name: cap for cap in current_genome.capabilities}

        for cap in plan.added_capabilities:
            capability_map[cap.name] = cap

        for name in plan.removed_capabilities:
            capability_map.pop(name, None)

        new_capabilities = list(capability_map.values())

        return AgentGenome(
            version=current_genome.version + 1,
            persona_name=current_genome.persona_name,
            role_description=current_genome.role_description,
            reasoning_protocol=plan.modified_protocol or current_genome.reasoning_protocol,
            capabilities=new_capabilities,
            constraints=current_genome.constraints
        )
