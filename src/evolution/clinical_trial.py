import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.evaluation.feedback import EnvironmentFeedback
from src.evaluation.simulator import EnvironmentSimulator
from src.evaluation.stateful_contract import extract_output_object


@dataclass
class ClinicalTrialReport:
    """Structured result of exercising a phenotype or generated organ."""
    organ_name: Optional[str]
    output: str
    turns_taken: int
    feedback: EnvironmentFeedback
    exception_type: Optional[str] = None
    turn_transcript: str = ""

    @property
    def success(self) -> bool:
        return self.feedback.success

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "organ_name": self.organ_name,
            "output": self.output,
            "turns_taken": self.turns_taken,
            "exception_type": self.exception_type,
            "turn_transcript": self.turn_transcript,
            "feedback": {
                "success": self.feedback.success,
                "critique": self.feedback.critique,
                "identified_gaps": self.feedback.identified_gaps,
            },
        }


class OrganClinicalTrialHarness:
    """
    Executes phenotype trials and converts runtime crashes into structured data.

    The harness knows nothing about task domains. It delegates all environment
    behavior and deterministic scoring to EnvironmentSimulator.
    """

    def __init__(self, environment_simulator: EnvironmentSimulator):
        self.environment_simulator = environment_simulator

    async def run(self, agent: Any, task: str, organ_name: Optional[str]) -> ClinicalTrialReport:
        try:
            output, turns_taken, feedback = await self.environment_simulator.evaluate_agent(agent, task)
            return ClinicalTrialReport(
                organ_name=organ_name,
                output=output,
                turns_taken=turns_taken,
                feedback=feedback,
                turn_transcript=self._turn_transcript(output),
            )
        except Exception as exc:
            details = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__, limit=8)
            )
            return ClinicalTrialReport(
                organ_name=organ_name,
                output=details,
                turns_taken=0,
                feedback=EnvironmentFeedback(
                    success=False,
                    critique=(
                        "Clinical trial raised a runtime exception from the active "
                        f"phenotype: {type(exc).__name__}."
                    ),
                    identified_gaps=["runtime_exception", "generated_organ_crash"],
                ),
                exception_type=type(exc).__name__,
            )

    @classmethod
    def _turn_transcript(cls, output: str, limit: int = 5) -> str:
        parsed = extract_output_object(output)
        if not isinstance(parsed, dict):
            return ""

        trace = parsed.get("state_trace")
        if not isinstance(trace, list):
            return ""

        rows: List[str] = []
        for step in trace:
            if not isinstance(step, dict):
                continue
            turn = step.get("turn", "?")
            observation = cls._read_json(step.get("observation_trace"))
            action = cls._read_json(step.get("action_trace"))
            result = cls._read_json(step.get("result_trace"))
            if not isinstance(observation, dict) or not isinstance(action, dict):
                continue

            raw_action = action.get("raw_action", "")
            parsed_action = extract_output_object(raw_action) if isinstance(raw_action, str) else None
            observation_delta = observation.get("observation_delta", {})
            row = {
                "turn": turn,
                "event": observation.get("event"),
                "delta_keys": sorted(observation_delta.keys()) if isinstance(observation_delta, dict) else [],
                "tool": action.get("tool_name"),
                "action_summary": cls._summarize_value(parsed_action),
                "candidate_final_submitted": (
                    result.get("candidate_final_submitted")
                    if isinstance(result, dict)
                    else None
                ),
            }
            rows.append(json.dumps(row, sort_keys=True))
            if len(rows) >= limit:
                break

        if not rows:
            return ""
        return "\n".join(rows)

    @classmethod
    def _summarize_value(cls, value: Any, depth: int = 0) -> Any:
        if depth >= 3:
            return cls._type_label(value)
        if isinstance(value, dict):
            if cls._looks_like_data_map(value):
                first_value = next(iter(value.values())) if value else None
                return {
                    "type": "dict",
                    "len": len(value),
                    "value": cls._summarize_value(first_value, depth + 1) if value else "empty",
                }
            summary: Dict[str, Any] = {}
            for key, child in sorted(value.items()):
                if key in {"query_text", "raw_action"}:
                    summary[key] = cls._type_label(child)
                else:
                    summary[str(key)] = cls._summarize_value(child, depth + 1)
            return summary
        if isinstance(value, list):
            return {
                "type": "list",
                "len": len(value),
                "item": cls._summarize_value(value[0], depth + 1) if value else "empty",
            }
        return cls._type_label(value)

    @staticmethod
    def _type_label(value: Any) -> str:
        if isinstance(value, str):
            return f"str[{len(value)}]"
        if isinstance(value, dict):
            return f"dict[{len(value)}]"
        if isinstance(value, list):
            return f"list[{len(value)}]"
        return type(value).__name__

    @staticmethod
    def _looks_like_data_map(value: Dict[Any, Any]) -> bool:
        if len(value) > 5:
            return True
        for key in value:
            text = str(key)
            if "." in text or "/" in text:
                return True
        return False

    @staticmethod
    def _read_json(raw_path: Any) -> Any:
        if not raw_path:
            return None
        try:
            return json.loads(Path(str(raw_path)).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
