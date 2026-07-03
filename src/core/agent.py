import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from src.core.genome import AgentGenome
from src.evaluation.stateful_benchmark import parse_episode_prompt
from src.execution.tools import TOOL_MAPPING, register_compiled_skill
from src.utils.config import config
from src.services.llm import LLMService


class StemAgent:
    """
    The vessel for the evolving AI. It is entirely defined by its genome.
    It manages the current state of the agent, its history of transformations, and interactions with the environment.
    The StemAgent can propose transformations to its genome based on its experiences and feedback, allowing it to adapt and improve over time.
    """

    def __init__(self, genome: Optional[AgentGenome] = None, llm: LLMService = None):
        self.genome = genome or AgentGenome()  # if no genome is provided, start with a default one
        self.history: List[AgentGenome] = [self.genome]
        self.llm = llm
        self._ensure_capability_tools()

    def save_genome(self, file_path: str) -> None:
        """Saves the current genome to a JSON file."""
        with open(file_path, "w") as f:
            f.write(self.genome.model_dump_json(indent=2))
        print(f"[*] Genome saved to {file_path}")

    @classmethod
    def load_genome(cls, file_path: str, llm: LLMService) -> "StemAgent":
        """Creates a new StemAgent instance from a saved genome file."""
        with open(file_path, "r") as f:
            genome_data = json.load(f)

            genome = AgentGenome(**genome_data)
            print(f"[*] Genome {genome.version} loaded from {file_path}")
            return cls(genome=genome, llm=llm)

    def _compile_system_message(self) -> str:
        """Compile the current genome into a system message for the configured chat model."""
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

    def update_genome(self, new_genome: AgentGenome) -> None:
        """Appliers a mutation and tracks history. Gives opportunity for potential rollback."""
        self.history.append(self.genome)
        self.genome = new_genome
        self._ensure_capability_tools()
        print(f"[*] Evolution successful. Transitioned to version {self.genome.version}")

    def rollback(self) -> None:
        """Rolls back the current genome."""
        if len(self.history) > 1:
            self.genome = self.history.pop()
            print(f"[!] Rollback initiated. Reverted to version {self.genome.version}")

    async def execute_task(self, user_input: str, max_turns: int = config["agent"]["max_turns"]) -> Tuple[str, int]:
        """
        Executes a task based on the current genome.
        Returns a tuple of (final_content, turns_taken).
        """
        payload = parse_episode_prompt(user_input)
        if payload is not None:
            return (
                "Error: Stateful benchmark tasks must be executed by the environment "
                "episode runtime, not by a one-shot agent answer.",
                0
            )

        if self.llm is None:
            return "Error: No LLM service configured and no acquired organ can handle this task.", 0

        messages: List[Dict[str, Any]] = [
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
                raw_args = tool_call.function.arguments
                if isinstance(raw_args, str):
                    function_args = json.loads(raw_args or "{}")
                elif isinstance(raw_args, dict):
                    function_args = raw_args
                else:
                    function_args = {}

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

    async def execute_episode_turn(self, observation: str) -> Tuple[str, bool, Optional[str]]:
        """
        Execute one physical benchmark turn.

        Returns (tool_output, tool_invoked, tool_name). The environment owns the
        loop and trace files; the agent only chooses and runs an acquired organ.
        """
        self._ensure_capability_tools()
        try:
            payload = json.loads(observation)
        except json.JSONDecodeError:
            return "Error: observation is not valid JSON.", False, None

        tool_name = self._select_stateful_tool(payload)
        if tool_name is None:
            return (
                "Error: No acquired organ can handle this stateful benchmark "
                f"domain: {payload.get('domain_id')}",
                False,
                None,
            )

        print(f"[*] Agent executing episode turn with: {tool_name}...")
        try:
            output = TOOL_MAPPING[tool_name](observation=observation)
        except Exception as exc:
            output = f"Execution Error: {type(exc).__name__}: {exc}"
        return str(output), True, tool_name

    def _ensure_capability_tools(self) -> None:
        """Load compiled generated organs referenced by this genome."""
        compiled_dir = Path(__file__).resolve().parents[1] / "compiled_skills"
        for capability in self.genome.capabilities:
            if capability.name in TOOL_MAPPING:
                continue

            skill_path = compiled_dir / f"{capability.name}.py"
            if not skill_path.exists():
                continue

            try:
                register_compiled_skill(capability.name, skill_path)
            except Exception as exc:
                print(f"[!] Warning: failed to load compiled organ {capability.name}: {exc}")

    def _select_stateful_tool(self, payload: Dict[str, Any]) -> Optional[str]:
        """Select the newest generated organ tagged for this benchmark domain."""
        domain_id = payload.get("domain_id")
        if not domain_id:
            return None

        domain_marker = f"domain_id:{domain_id}"
        for capability in reversed(self.genome.capabilities):
            if capability.name not in TOOL_MAPPING:
                continue
            if domain_marker in capability.required_context:
                return capability.name
        return None

    @staticmethod
    def _count_stateful_turns(agent_output: str) -> int:
        """Count internal stateful work steps emitted by a deterministic organ."""
        try:
            output = json.loads(agent_output)
        except json.JSONDecodeError:
            return 1

        state_trace = output.get("state_trace")
        if isinstance(state_trace, list):
            return len(state_trace)
        return 1

    def _get_openai_tools(self) -> Optional[List[Dict[str, Any]]]:
        if not self.genome.capabilities:
            return None

        tools = []
        for cap in self.genome.capabilities:
            if not cap.name or not all(c.isalnum() or c in "-_" for c in cap.name):
                print(f"[!] Warning: Capability name '{cap.name}' contains invalid characters. Cleaning up...")
                cap.name = "".join(c for c in cap.name if c.isalnum() or c in "-_")

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
