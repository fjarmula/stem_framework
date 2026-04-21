import json
from typing import List, Optional
import openai
from core.genome import AgentGenome
from execution.registry import TOOL_MAPPING


class StemAgent:
    """
    The vessel for the evolving AI. It is entirely defined by its genome.
    It manages the current state of the agent, its history of transformations, and interactions with the environment (via OpenAI API calls).
    The StemAgent can propose transformations to its genome based on its experiences and feedback, allowing it to adapt and improve over time.
    """

    def __init__(self, genome: Optional[AgentGenome] = None, api_key: str = None):
        self.genome = genome or AgentGenome()  # if no genome is provided, start with a default one
        self.history: List[AgentGenome] = [self.genome]
        self.client: openai.AsyncOpenAI = openai.AsyncOpenAI(api_key=api_key)

    def _compile_system_message(self) -> str:
        """Compile the current genome into a system message for the OpenAI API."""
        capabilities_text = "\n".join([f"- {cap.name}: {cap.description} Requires: {', '.join(cap.required_context)}" for cap in
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

    async def execute_task(self, user_input: str, max_turns: int = 5):
        """Executes a task based on the current genome and user input."""
        messages = [
            {"role": "system", "content": self._compile_system_message()},
            {"role": "user", "content": user_input}
        ]

        for _ in range(max_turns):
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=self._get_openai_tools()
            )
            response_message = response.choices[0].message
            messages.append(response_message)

            if not response_message.tool_calls:
                return response_message.content

            tool_calls = response_message.tool_calls
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                print(f"[*] Agent executing: {function_name}...")

                if function_name in TOOL_MAPPING:
                    function_response = TOOL_MAPPING[function_name](**function_args)
                else:
                    function_response = f"Error: Tool {function_name} not found in registry. Available tools: {list(TOOL_MAPPING.keys())}"

                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                })
        return messages[-1].get("content", "Error: Maximum reasoning turns reached.")

    def _get_openai_tools(self):
        """Converts genome capabilities into OpenAI tool specification."""
        if not self.genome.capabilities:
            return None

        tools = []
        for cap in self.genome.capabilities:
            tools.append({
                "type": "function",
                "function": {
                    "name": cap.name,
                    "description": cap.description,
                    "parameters": json.loads(cap.parameters) if cap.parameters else {
                        "type": "object",
                        "properties": {"code": {"type": "string"}},
                        "required": ["code"]
                    }
                }
            })
        return tools
