import json
import importlib.util
import inspect
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from src.core.genome import AgentGenome, CapabilityModel
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
        self._episode_memory: Dict[str, Dict[str, Any]] = {}
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
            self._episode_memory.clear()
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

        capability = self._select_stateful_capability(payload)
        if capability is None:
            output = await self._attempt_episode_turn_without_organ(payload)
            return output, False, None

        memory_key = self._episode_memory_key(payload)
        if int(payload.get("turn") or 0) <= 1:
            self._episode_memory.pop(memory_key, None)
        if memory_key in self._episode_memory:
            payload["memory"] = self._episode_memory[memory_key]

        tool_name = capability.name
        print(f"[*] Agent executing episode turn with: {tool_name}...")
        output = self._execute_compiled_capability(capability, payload)
        self._preserve_episode_memory(memory_key, output)
        return self._stringify_tool_output(output), True, tool_name

    async def _attempt_episode_turn_without_organ(self, payload: Dict[str, Any]) -> str:
        """Let the unspecialized stem cell attempt a turn without runtime organs."""
        if self.llm is None:
            return (
                "Error: No acquired organ can handle this stateful benchmark "
                f"domain: {payload.get('domain_id')}"
            )

        response = await self.llm.get_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an unspecialized stem-cell baseline. You have no "
                        "runtime organs or tools. Attempt the observation honestly, "
                        "but do not claim physical execution or file modification."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(payload, indent=2, sort_keys=True),
                },
            ],
            tools=None,
        )
        return getattr(response, "content", "") or ""

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
        capability = self._select_stateful_capability(payload)
        return capability.name if capability else None

    def _select_stateful_capability(self, payload: Dict[str, Any]) -> Optional[CapabilityModel]:
        """Select the newest generated capability tagged for this benchmark domain."""
        domain_id = payload.get("domain_id")
        if not domain_id:
            return None

        domain_marker = f"domain_id:{domain_id}"
        for capability in reversed(self.genome.capabilities):
            if domain_marker in capability.required_context:
                if capability.name in TOOL_MAPPING or self._compiled_skill_path(capability.name).exists():
                    return capability
        return None

    def _execute_compiled_capability(self, capability: CapabilityModel, payload: Dict[str, Any]) -> Any:
        """
        Execute a compiled stateful organ without giving the LLM an opportunity
        to skip the tool route.
        """
        if self._compiled_skill_path(capability.name).exists():
            module = self._load_compiled_module(capability.name)
            organ_class = getattr(module, capability.name, None)
            if inspect.isclass(organ_class):
                instance = organ_class()
                execute = getattr(instance, "execute", None)
                if callable(execute):
                    return execute(payload)
                for method_name in ("run", "__call__"):
                    method = getattr(instance, method_name, None)
                    if callable(method):
                        return method(json.dumps(payload, sort_keys=True))
                raise AttributeError(
                    f"Compiled organ class {capability.name} must expose execute(), run(), or __call__()."
                )

            module_run = getattr(module, "run", None)
            if callable(module_run):
                return module_run(observation=json.dumps(payload, sort_keys=True))

        if capability.name in TOOL_MAPPING:
            return TOOL_MAPPING[capability.name](observation=json.dumps(payload, sort_keys=True))

        raise ValueError(f"Compiled organ {capability.name} is not registered and has no runnable entrypoint.")

    @staticmethod
    def _episode_memory_key(payload: Dict[str, Any]) -> str:
        return f"{payload.get('domain_id', 'unknown')}::{payload.get('episode_id', 'unknown')}"

    def _preserve_episode_memory(self, memory_key: str, output: Any) -> None:
        output_object = self._coerce_output_object(output)
        if output_object is None:
            return

        retained: Dict[str, Any] = {}
        memory = output_object.get("memory")
        if isinstance(memory, dict):
            retained.update(memory)
        for key in ("state_trace", "internal_state"):
            if key in output_object:
                retained[key] = output_object[key]
        if retained:
            self._episode_memory[memory_key] = retained

    @staticmethod
    def _coerce_output_object(output: Any) -> Optional[Dict[str, Any]]:
        if isinstance(output, dict):
            return output
        if not isinstance(output, str):
            return None
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _stringify_tool_output(output: Any) -> str:
        if isinstance(output, str):
            return output
        return json.dumps(output, indent=2, sort_keys=True)

    @staticmethod
    def _load_compiled_module(name: str) -> Any:
        skill_path = StemAgent._compiled_skill_path(name)
        if not skill_path.exists():
            raise FileNotFoundError(f"Compiled organ source not found: {skill_path}")

        spec = importlib.util.spec_from_file_location(f"src.compiled_skills.{name}", skill_path.resolve())
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load compiled organ module: {skill_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _compiled_skill_path(name: str) -> Path:
        return Path(__file__).resolve().parents[1] / "compiled_skills" / f"{name}.py"

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
