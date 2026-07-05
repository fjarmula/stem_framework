import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from src.evaluation.feedback import EnvironmentFeedback


SUPPORTED_STATEFUL_DOMAINS = {"trading_floor", "security_sandbox", "matrix_database"}


@dataclass
class EpisodeRunResult:
    """Result returned by the physical multi-turn benchmark runtime."""
    output: str
    turns_taken: int
    feedback: EnvironmentFeedback
    workspace: str


TurnExecutor = Callable[[str], Awaitable[Tuple[str, bool, Optional[str]]]]


def parse_episode_prompt(task: str) -> Optional[Dict[str, Any]]:
    """Extract the public benchmark payload from a rendered v2 episode prompt."""
    marker = "STATEFUL STEM-CELL BENCHMARK EPISODE"
    if marker not in task:
        return None

    start = task.find("{")
    if start < 0:
        return None

    try:
        payload, _ = json.JSONDecoder().raw_decode(task[start:])
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict) or payload.get("benchmark_version") != "2.0":
        return None
    return payload


def minimum_turns(payload: Dict[str, Any]) -> int:
    contract = payload.get("episode_contract", {})
    if not isinstance(contract, dict):
        return 0
    try:
        return int(contract.get("minimum_turns") or 0)
    except (TypeError, ValueError):
        return 0


def extract_output_object(agent_output: str) -> Optional[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    text = agent_output.strip()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def final_artifact(output: Dict[str, Any]) -> Dict[str, Any]:
    artifact = output.get("final_artifact", output)
    return artifact if isinstance(artifact, dict) else {}


def difference_location(observed: Any, expected: Any) -> str:
    if not isinstance(observed, list) or not isinstance(expected, list):
        return "container shape differs"
    if len(observed) != len(expected):
        return f"length differs; observed length {len(observed)}"
    for index, (observed_row, expected_row) in enumerate(zip(observed, expected)):
        if observed_row != expected_row:
            return f"row {index}"
    return "no row-level difference found"
