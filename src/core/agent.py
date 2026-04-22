import json
from typing import List, Optional
from src.core.genome import AgentGenome
from src.execution.tools import TOOL_MAPPING
from src.config import config
from src.services.llm import LLMService


class StemAgent:
    """
    The vessel for the evolving AI. It is entirely defined by its genome.
    It manages the current state of the agent, its history of transformations, and interactions with the environment (via OpenAI API calls).
    The StemAgent can propose transformations to its genome based on its experiences and feedback, allowing it to adapt and improve over time.
    """

    def __init__(self, genome: Optional[AgentGenome] = None, llm: LLMService = None):
        self.genome = genome or AgentGenome()  # if no genome is provided, start with a default one
        self.history: List[AgentGenome] = [self.genome]
        self.llm = llm

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

    async def execute_task(self, user_input: str, max_turns: int = config["agent"]["max_turns"]):
        """
        Executes a task based on the current genome.
        Returns a tuple of (final_content, turns_taken).
        """
        messages = [
            {"role": "system", "content": self._compile_system_message()},
            {"role": "user", "content": user_input}
        ]

        turns_taken = 0
        for turn in range(max_turns):
            turns_taken += 1

            # Use the centralized LLM Service
            response_message = await self.llm.get_chat_completion(
                messages=messages,
                tools=self._get_openai_tools()
            )

            # Store the assistant's message (including tool calls if any)
            assistant_msg = {
                "role": "assistant",
                "content": response_message.content,
            }
            if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in response_message.tool_calls
                ]

            messages.append(assistant_msg)

            # If no tool calls, the agent is finished
            if not getattr(response_message, 'tool_calls', None):
                return response_message.content, turns_taken

            # Process tool calls
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                print(f"[*] Agent executing: {function_name}...")

                if function_name in TOOL_MAPPING:
                    try:
                        function_response = TOOL_MAPPING[function_name](**function_args)
                    except Exception as e:
                        function_response = f"Execution Error: {str(e)}"
                else:
                    function_response = f"Error: Tool {function_name} not found."

                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": str(function_response),
                })

        # If loop finishes without returning, we hit max turns
        final_content = messages[-1].get("content") or "Error: Maximum reasoning turns reached."
        return final_content, turns_taken

    def _get_openai_tools(self):
        if not self.genome.capabilities:
            return None

        tools = []
        for cap in self.genome.capabilities:
            params = None
            if cap.parameters:
                try:
                    params = json.loads(cap.parameters)
                except json.JSONDecodeError:
                    params = None
            if not isinstance(params, dict):
                params = {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"]
                }
            tools.append({
                "type": "function",
                "function": {
                    "name": cap.name,
                    "description": cap.description,
                    "parameters": params
                }
            })
        return tools
