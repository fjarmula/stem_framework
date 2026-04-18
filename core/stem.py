from typing import List, Optional
import openai
from models.genome import AgentGenome, TransformationPlan

class StemAgent:
    """
    The vessel for the evolving AI. It is entirely defined by its genome.
    It manages the current state of the agent, its history of transformations, and interactions with the environment (via OpenAI API calls).
    The StemAgent can propose transformations to its genome based on its experiences and feedback, allowing it to adapt and improve over time.
    """
    def __init__(self, genome: Optional[AgentGenome]=None,  api_key: str=None):
        self.genome = genome or AgentGenome() # if no genome is provided, start with a default one
        self.history: List[AgentGenome] = [self.genome]
        self.client: openai.Client = openai.Client(api_key=api_key)

    def _compile_system_message(self) -> str:
        """Compile the current genome into a system message for the OpenAI API."""
        capabilities_text = "\n".join([f"- {cap.name}: {cap.description} Requires: {', '.join(cap.required_context)}"for cap in
                                    self.genome.capabilities])
        constraints_text = "\n".join([f"- {constraint}" for constraint in self.genome.constraints])
        return f"""
        Identity: {self.genome.persona_name}
        Role: {self.genome.role_description}
        
        Reasoning Protocol: {self.genome.reasoning_protocol}
        
        Available Capabilities:
        {capabilities_text if self.genome.capabilities else "General LLM reasoning."}
        
        Constraints:
        {constraints_text if self.genome.constraints else "Standard AI safety guidelines."}
        """.strip()

    def update_genome(self, new_genome: AgentGenome):
        """Appliers a mutation and tracks history. Gives opportunity for potential rollback."""
        self.history.append(self.genome)
        self.genome = new_genome
        print(f"[*] Evolution successful. Transitioned to version {self.genome.version}")

    def rollback(self):
        """Rolls back the current genome."""
        if len(self.history) > 1:
            self.genome = self.history.pop()
            print(f"[!] Rollback initiated. Reverted to version {self.genome.version}")

    async def execute_task(self, user_input: str):
        """Executes a task based on the current genome and user input."""
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": self._compile_system_message()},
                {"role": "user", "content": user_input}
            ]
        )
        return response.choices[0].message.content