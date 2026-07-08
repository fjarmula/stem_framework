import importlib.util
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.evaluation.feedback import EnvironmentFeedback
from src.evaluation.stateful_contract import (
    extract_output_object,
    final_artifact,
    minimum_turns,
    parse_episode_prompt,
)
from src.services.task_repository import find_task_episode


def unverifiable_inference_feedback(reason: str = "No runtime tool produced a physical trace.") -> EnvironmentFeedback:
    return EnvironmentFeedback(
        success=False,
        critique=(
            f"{reason} Stateful benchmark answers are schema-only until produced "
            "inside the physical multi-turn episode runtime."
        ),
        identified_gaps=["unverifiable_inference", "missing_physical_trace"],
    )


def verify_physical_episode(
    payload: Dict[str, Any],
    agent_output: str,
    workspace: Path,
    turns_taken: int,
    tool_invocations: int,
) -> EnvironmentFeedback:
    """Verify final output only after the runner has produced physical traces."""
    if tool_invocations <= 0:
        return unverifiable_inference_feedback("The agent did not invoke an acquired runtime organ.")

    trace_feedback = _verify_physical_trace(payload, workspace)
    if trace_feedback is not None:
        return trace_feedback

    if not agent_output.strip():
        return EnvironmentFeedback(
            success=False,
            critique="The physical episode completed without a parseable final submission.",
            identified_gaps=["incomplete_final_artifact", "missing_execution_trace"],
        )

    return _verify_stateful_payload_output(
        payload,
        agent_output,
        turns_taken=turns_taken,
        workspace=workspace,
    )


def verify_stateful_episode(task: str, agent_output: str, turns_taken: Optional[int] = None) -> Optional[EnvironmentFeedback]:
    """Reject standalone benchmark answers that did not pass through the runtime."""
    payload = parse_episode_prompt(task)
    if payload is None:
        return None

    return unverifiable_inference_feedback(
        "A standalone final answer was submitted without the physical episode runtime."
    )


def _verify_physical_trace(payload: Dict[str, Any], workspace: Path) -> Optional[EnvironmentFeedback]:
    trace_dir = workspace / "trace"
    sandbox_dir = workspace / "sandbox"
    required_turns = max(minimum_turns(payload), len(payload.get("turns") or []), 1)
    missing_trace_files = []
    for turn_number in range(1, required_turns + 1):
        for suffix in ("observation", "action", "result"):
            path = trace_dir / f"turn_{turn_number:03d}_{suffix}.json"
            if not path.exists():
                missing_trace_files.append(str(path))
        sandbox_state_path = sandbox_dir / f"turn_{turn_number:03d}_state.json"
        if not sandbox_state_path.exists():
            missing_trace_files.append(str(sandbox_state_path))
    if not missing_trace_files:
        return None
    return EnvironmentFeedback(
        success=False,
        critique=(
            "The episode did not leave a complete physical turn trace. "
            f"Missing files: {missing_trace_files[:3]}"
        ),
        identified_gaps=["missing_physical_trace", "incomplete_episode_trace"],
    )


def _verify_stateful_payload_output(
    payload: Dict[str, Any],
    agent_output: str,
    turns_taken: Optional[int] = None,
    workspace: Optional[Path] = None,
) -> EnvironmentFeedback:
    """Verify final output through the task-owned contract and verifier artifacts."""
    output = extract_output_object(agent_output)
    if output is None:
        return EnvironmentFeedback(
            success=False,
            critique="The output does not contain a parseable final artifact object.",
            identified_gaps=["missing_physical_trace", "incomplete_final_artifact"],
        )

    trace_feedback = _verify_state_trace_contract(payload, output, turns_taken)
    if trace_feedback is not None:
        return trace_feedback

    artifact = final_artifact(output)
    schema_feedback = _verify_output_contract(payload.get("output_contract", {}), artifact)
    if schema_feedback is not None:
        return schema_feedback

    task_definition = find_task_episode(
        str(payload.get("domain_id", "")),
        str(payload.get("episode_id", "")),
    )
    private_artifacts = (task_definition or {}).get("private_verifier_artifacts", {})
    if not isinstance(private_artifacts, dict):
        private_artifacts = {}

    expected_path = _artifact_path(private_artifacts.get("expected"))
    verifier_path = _artifact_path(private_artifacts.get("verifier"))
    if verifier_path is not None and verifier_path.exists():
        return _verify_with_task_module(
            verifier_path=verifier_path,
            submission=_select_submission(output, private_artifacts),
            expected_path=expected_path,
            workspace=workspace,
        )

    if expected_path is not None and expected_path.exists():
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        return _verify_expected_json(_select_submission(output, private_artifacts), expected)

    return EnvironmentFeedback(
        success=False,
        critique="The benchmark episode has no task-owned verifier artifact available.",
        identified_gaps=["unverifiable_output"],
    )


def _verify_state_trace_contract(
    payload: Dict[str, Any],
    output: Dict[str, Any],
    turns_taken: Optional[int],
) -> Optional[EnvironmentFeedback]:
    required_turns = minimum_turns(payload)
    trace = output.get("state_trace")
    if required_turns <= 0:
        return None
    if turns_taken is not None and turns_taken < required_turns:
        return EnvironmentFeedback(
            success=False,
            critique=(
                f"The episode collapsed into {turns_taken} actual turn(s), "
                f"but the contract requires at least {required_turns} stateful turns."
            ),
            identified_gaps=["multi_turn_collapse", "missing_physical_trace"],
        )
    if isinstance(trace, list) and len(trace) >= required_turns:
        return None
    trace_count = len(trace) if isinstance(trace, list) else 0
    return EnvironmentFeedback(
        success=False,
        critique=(
            f"The state_trace contains {trace_count} turn(s), "
            f"but the contract requires at least {required_turns} reconstructable turns."
        ),
        identified_gaps=["multi_turn_collapse", "missing_execution_trace"],
    )


def _verify_output_contract(
    output_contract: Dict[str, Any],
    artifact: Dict[str, Any],
) -> Optional[EnvironmentFeedback]:
    final_contract = output_contract.get("final_artifact", {})
    if not isinstance(final_contract, dict):
        return None

    required_keys = final_contract.get("required_keys", [])
    if isinstance(required_keys, list):
        missing = sorted(str(key) for key in required_keys if key not in artifact)
        if missing:
            return EnvironmentFeedback(
                success=False,
                critique=(
                    "The final_artifact is missing required key(s) from the task output_contract: "
                    f"{missing}. Observed keys: {sorted(artifact.keys())}."
                ),
                identified_gaps=["incomplete_final_artifact"],
            )

    for contract_key, required in final_contract.items():
        if not isinstance(required, list):
            continue
        if contract_key.endswith("_row_required_keys"):
            target_key = contract_key.removesuffix("_row_required_keys")
        elif contract_key.endswith("_required_keys"):
            target_key = contract_key.removesuffix("_required_keys")
        else:
            continue
        target = artifact.get(target_key)
        feedback = _verify_nested_required_keys(target_key, target, required)
        if feedback is not None:
            return feedback

    return None


def _verify_nested_required_keys(
    target_key: str,
    target: Any,
    required_keys: List[Any],
) -> Optional[EnvironmentFeedback]:
    required = {str(key) for key in required_keys}
    if isinstance(target, dict):
        missing = sorted(required - set(target))
        if missing:
            return EnvironmentFeedback(
                success=False,
                critique=f"final_artifact.{target_key} is missing required key(s): {missing}.",
                identified_gaps=["incomplete_final_artifact"],
            )
    elif isinstance(target, list):
        for index, row in enumerate(target):
            if not isinstance(row, dict):
                return EnvironmentFeedback(
                    success=False,
                    critique=f"final_artifact.{target_key}[{index}] must be an object.",
                    identified_gaps=["incorrect_output"],
                )
            missing = sorted(required - set(row))
            if missing:
                return EnvironmentFeedback(
                    success=False,
                    critique=(
                        f"final_artifact.{target_key}[{index}] is missing required key(s): "
                        f"{missing}. Every row must include {sorted(required)}."
                    ),
                    identified_gaps=["incorrect_output"],
                )
    return None


def _verify_with_task_module(
    verifier_path: Path,
    submission: Any,
    expected_path: Optional[Path],
    workspace: Optional[Path],
) -> EnvironmentFeedback:
    submission_path = (workspace or Path.cwd()) / "verifier_submission.json"
    submission_path.write_text(json.dumps(submission, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    try:
        module = _load_verifier_module(verifier_path)
        verify = getattr(module, "verify")
        kwargs: Dict[str, Any] = {"repo_root": str(Path.cwd())}
        signature = inspect.signature(verify)
        if "expected_path" in signature.parameters and expected_path is not None:
            kwargs["expected_path"] = str(expected_path)
        result = verify(str(submission_path), **kwargs)
    except Exception as exc:
        return EnvironmentFeedback(
            success=False,
            critique=f"Task verifier raised {type(exc).__name__}: {exc}",
            identified_gaps=["verifier_exception", "incorrect_output"],
        )

    return _normalize_verifier_result(result)


def _load_verifier_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(f"stem_task_verifier_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import verifier module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_verifier_result(result: Any) -> EnvironmentFeedback:
    if isinstance(result, EnvironmentFeedback):
        return result
    if isinstance(result, dict):
        success = bool(result.get("success"))
        critique = str(result.get("critique") or ("Task verifier accepted output." if success else "Task verifier rejected output."))
        tags = result.get("identified_gaps", [])
        return EnvironmentFeedback(
            success=success,
            critique=critique,
            identified_gaps=[str(tag) for tag in tags] if isinstance(tags, list) else ["incorrect_output"],
        )
    if isinstance(result, tuple) and len(result) == 3:
        success, critique, tags = result
        return EnvironmentFeedback(
            success=bool(success),
            critique=str(critique),
            identified_gaps=[str(tag) for tag in tags] if isinstance(tags, list) else ["incorrect_output"],
        )
    if isinstance(result, bool):
        return EnvironmentFeedback(
            success=result,
            critique="Task verifier accepted output." if result else "Task verifier rejected output.",
            identified_gaps=[] if result else ["incorrect_output"],
        )
    return EnvironmentFeedback(
        success=False,
        critique=f"Task verifier returned unsupported result type: {type(result).__name__}.",
        identified_gaps=["verifier_exception"],
    )


def _verify_expected_json(submission: Any, expected: Any) -> EnvironmentFeedback:
    success = submission == expected
    return EnvironmentFeedback(
        success=success,
        critique="The output exactly matches the task expected artifact." if success else "The output does not match the task expected artifact.",
        identified_gaps=[] if success else ["incorrect_output"],
    )


def _select_submission(output: Dict[str, Any], private_artifacts: Dict[str, Any]) -> Any:
    source = str(private_artifacts.get("submission_source", "final_artifact"))
    selected: Any = output
    for part in source.split("."):
        if part == "":
            continue
        if part == "final_artifact":
            selected = final_artifact(output)
            continue
        if isinstance(selected, dict):
            selected = selected.get(part)
        else:
            return None
    return selected


def _artifact_path(raw_path: Any) -> Optional[Path]:
    if not raw_path:
        return None
    return Path(str(raw_path))
