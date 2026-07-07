import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.evaluation.stateful_contract import extract_output_object, final_artifact


@dataclass
class ProbeFailure:
    name: str
    critique: str
    tags: List[str]


def evaluate_clinical_probes(payload: Dict[str, Any], output: str) -> List[ProbeFailure]:
    """Evaluate task-provided public clinical probes against traces and output."""
    probes = payload.get("clinical_probes") or []
    if not isinstance(probes, list) or not probes:
        return []

    parsed = extract_output_object(output)
    if not isinstance(parsed, dict):
        return [
            ProbeFailure(
                name="parse_final_output",
                critique="Clinical probes could not parse the final output as a JSON object.",
                tags=["clinical_probe_failure", "incomplete_final_artifact"],
            )
        ]

    context = {
        "final": parsed,
        "final_artifact": final_artifact(parsed),
        "memory": _latest_memory(parsed),
    }

    failures: List[ProbeFailure] = []
    for index, probe in enumerate(probes, start=1):
        if not isinstance(probe, dict):
            continue
        failure = _evaluate_probe(index, probe, context)
        if failure:
            failures.append(failure)
    return failures


def _evaluate_probe(index: int, probe: Dict[str, Any], context: Dict[str, Any]) -> Optional[ProbeFailure]:
    name = str(probe.get("name") or f"probe_{index}")
    assertion = str(probe.get("assert") or "exists")
    source = str(probe.get("source") or "final")
    value = _resolve_source(source, context)

    passed = False
    if assertion == "exists":
        passed = value is not None
    elif assertion == "non_empty":
        passed = _is_non_empty(value)
    elif assertion == "type":
        passed = _type_name(value) == str(probe.get("expected_type"))
    elif assertion == "min_length":
        passed = _length(value) >= int(probe.get("min_length", 1))
    elif assertion == "contains_keys":
        keys = probe.get("keys") or []
        passed = isinstance(value, dict) and all(str(key) in value for key in keys)
    elif assertion == "equals":
        passed = value == probe.get("expected")
    else:
        return ProbeFailure(
            name=name,
            critique=f"Clinical probe {name!r} uses unsupported assertion {assertion!r}.",
            tags=["clinical_probe_failure", "unsupported_clinical_probe"],
        )

    if passed:
        return None

    tags = probe.get("failure_tags") or ["clinical_probe_failure"]
    critique = str(
        probe.get("critique")
        or (
            f"Clinical probe {name!r} failed: expected {source} to satisfy "
            f"{assertion}, observed {_shape(value)}."
        )
    )
    return ProbeFailure(name=name, critique=critique, tags=[str(tag) for tag in tags])


def _resolve_source(source: str, context: Dict[str, Any]) -> Any:
    parts = [part for part in source.split(".") if part]
    if not parts:
        return None

    value = context.get(parts[0])
    for part in parts[1:]:
        if isinstance(value, dict):
            value = value.get(part)
        elif isinstance(value, list):
            try:
                value = value[int(part)]
            except (TypeError, ValueError, IndexError):
                return None
        else:
            return None
    return value


def _latest_memory(parsed_output: Dict[str, Any]) -> Dict[str, Any]:
    memory = parsed_output.get("memory")
    if isinstance(memory, dict):
        return memory

    latest: Dict[str, Any] = {}
    trace = parsed_output.get("state_trace")
    if not isinstance(trace, list):
        return latest

    for step in trace:
        if not isinstance(step, dict) or not step.get("action_trace"):
            continue
        action = _read_json(step.get("action_trace"))
        if not isinstance(action, dict):
            continue
        raw_action = action.get("raw_action")
        parsed_action = extract_output_object(raw_action) if isinstance(raw_action, str) else None
        if isinstance(parsed_action, dict) and isinstance(parsed_action.get("memory"), dict):
            latest = parsed_action["memory"]
    return latest


def _read_json(raw_path: Any) -> Any:
    if not raw_path:
        return None
    try:
        return json.loads(Path(str(raw_path)).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _is_non_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (str, list, dict, tuple, set)):
        return len(value) > 0
    return True


def _length(value: Any) -> int:
    if isinstance(value, (str, list, dict, tuple, set)):
        return len(value)
    return 0


def _type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if value is None:
        return "null"
    return type(value).__name__


def _shape(value: Any) -> str:
    if isinstance(value, dict):
        return f"object keys={sorted(value.keys())[:8]}"
    if isinstance(value, list):
        return f"array length={len(value)}"
    if isinstance(value, str):
        return f"string length={len(value)}"
    return _type_name(value)
