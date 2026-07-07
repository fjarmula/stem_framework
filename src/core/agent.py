import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from src.core.genome import AgentGenome, CapabilityModel
from src.execution.tools import TOOL_MAPPING, register_compiled_skill
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

        tool_name = capability.name
        print(f"[*] Agent executing episode turn with: {tool_name}...")
        output = self._execute_registered_capability(capability, observation)
        return self._stringify_tool_output(output), True, tool_name

    async def _attempt_episode_turn_without_organ(self, payload: Dict[str, Any]) -> str:
        """Let the unspecialized stem cell attempt a turn without runtime organs."""
        if self.llm is None:
            return (
                "Error: No acquired organ can handle this stateful benchmark "
                f"domain: {payload.get('domain_id')}"
            )

        try:
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
        except Exception as exc:
            return (
                "Error: Unspecialized baseline could not complete this turn "
                f"without an acquired runtime organ ({type(exc).__name__})."
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

    def _select_stateful_capability(self, payload: Dict[str, Any]) -> Optional[CapabilityModel]:
        """Select the newest generated capability tagged for this benchmark domain."""
        domain_id = payload.get("domain_id")
        if not domain_id:
            return None

        domain_marker = f"domain_id:{domain_id}"
        for capability in reversed(self.genome.capabilities):
            if domain_marker in capability.required_context:
                if capability.name in TOOL_MAPPING:
                    return capability
        return None

    @staticmethod
    def _execute_registered_capability(capability: CapabilityModel, observation_text: str) -> Any:
        """Execute a domain-matched runtime organ through the active tool belt."""
        try:
            tool = TOOL_MAPPING[capability.name]
        except KeyError as exc:
            raise ValueError(f"Runtime organ {capability.name} is not registered.") from exc
        return tool(observation_text)

    @staticmethod
    def _stringify_tool_output(output: Any) -> str:
        if isinstance(output, str):
            return output
        return json.dumps(output, indent=2, sort_keys=True)

    @staticmethod
    def _compiled_skill_path(name: str) -> Path:
        return Path(__file__).resolve().parents[1] / "compiled_skills" / f"{name}.py"
