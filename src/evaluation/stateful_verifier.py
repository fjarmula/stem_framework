import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.evaluation.feedback import EnvironmentFeedback
from src.evaluation.stateful_contract import (
    SUPPORTED_STATEFUL_DOMAINS,
    difference_location,
    extract_output_object,
    final_artifact,
    minimum_turns,
    parse_episode_prompt,
)


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
    if missing_trace_files:
        return EnvironmentFeedback(
            success=False,
            critique=(
                "The episode did not leave a complete physical turn trace. "
                f"Missing files: {missing_trace_files[:3]}"
            ),
            identified_gaps=["missing_physical_trace", "incomplete_episode_trace"],
        )

    if not agent_output.strip():
        return EnvironmentFeedback(
            success=False,
            critique="The physical episode completed without a parseable final submission.",
            identified_gaps=["incomplete_final_artifact", "missing_execution_trace"],
        )

    return _verify_stateful_payload_output(payload, agent_output, turns_taken=turns_taken)


def verify_stateful_episode(task: str, agent_output: str, turns_taken: Optional[int] = None) -> Optional[EnvironmentFeedback]:
    """Reject standalone benchmark answers that did not pass through the runtime."""
    payload = parse_episode_prompt(task)
    if payload is None:
        return None

    return unverifiable_inference_feedback(
        "A standalone final answer was submitted without the physical episode runtime."
    )


def _verify_stateful_payload_output(
    payload: Dict[str, Any],
    agent_output: str,
    turns_taken: Optional[int] = None,
) -> EnvironmentFeedback:
    """Deterministically verify final output after physical runtime evidence exists."""
    if payload.get("domain_id") not in SUPPORTED_STATEFUL_DOMAINS:
        return EnvironmentFeedback(
            success=False,
            critique=f"Unsupported benchmark domain: {payload.get('domain_id')}",
            identified_gaps=["unsupported_domain", "missing_acquired_organ"]
        )

    expected_path = _expected_path(payload)
    if expected_path is None or not expected_path.exists():
        return EnvironmentFeedback(
            success=False,
            critique="The benchmark episode has no deterministic verifier artifact available.",
            identified_gaps=["unverifiable_output"]
        )

    output = extract_output_object(agent_output)
    if output is None:
        return EnvironmentFeedback(
            success=False,
            critique="The output does not contain a parseable final artifact object.",
            identified_gaps=["missing_physical_trace", "incomplete_final_artifact"]
        )

    required_turns = minimum_turns(payload)
    trace = output.get("state_trace")
    if required_turns > 0:
        if turns_taken is not None and turns_taken < required_turns:
            return EnvironmentFeedback(
                success=False,
                critique=(
                    f"The episode collapsed into {turns_taken} actual turn(s), "
                    f"but the contract requires at least {required_turns} stateful turns."
                ),
                identified_gaps=["multi_turn_collapse", "missing_physical_trace"]
            )
        if not isinstance(trace, list) or len(trace) < required_turns:
            trace_count = len(trace) if isinstance(trace, list) else 0
            return EnvironmentFeedback(
                success=False,
                critique=(
                    f"The state_trace contains {trace_count} turn(s), "
                    f"but the contract requires at least {required_turns} reconstructable turns."
                ),
                identified_gaps=["multi_turn_collapse", "missing_execution_trace"]
            )

    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    domain_id = payload.get("domain_id")

    if domain_id == "trading_floor":
        success, critique, tags = _verify_trading_output(output, expected)
    elif domain_id == "security_sandbox":
        success, critique, tags = _verify_security_output(output, expected)
    elif domain_id == "matrix_database":
        success, critique, tags = _verify_matrix_output(output, expected)
    else:
        success, critique, tags = False, "The episode domain is unsupported by the deterministic verifier.", ["unverifiable_output"]

    return EnvironmentFeedback(success=success, critique=critique, identified_gaps=tags)


def _verify_trading_output(output: Dict[str, Any], expected: Dict[str, Any]) -> Tuple[bool, str, List[str]]:
    artifact = final_artifact(output)
    if "final_portfolio" not in artifact:
        return (
            False,
            "Trading final_artifact must contain key 'final_portfolio'; "
            f"observed keys: {sorted(artifact.keys())}.",
            ["incomplete_final_artifact"]
        )

    portfolio = artifact.get("final_portfolio", {})
    ledger = artifact.get("ledger", [])
    if not isinstance(portfolio, dict):
        return (
            False,
            "Trading final_artifact.final_portfolio must be an object containing cash and nested positions.",
            ["incomplete_final_artifact"]
        )
    if "positions" not in portfolio:
        flat_position_keys = sorted(key for key in portfolio if key != "cash")
        return (
            False,
            "Trading final_portfolio must contain nested key 'positions' plus key 'cash'. "
            f"Observed flat position keys: {flat_position_keys}.",
            ["incomplete_final_artifact", "incorrect_output"]
        )
    if not isinstance(ledger, list):
        return (
            False,
            "Trading final_artifact.ledger must be a list of transaction row objects.",
            ["incomplete_final_artifact"]
        )
    if "ledger" not in artifact:
        observed_keys = sorted(artifact.keys())
        alias_hint = ""
        if "ledger_summary" in artifact:
            alias_hint = " Use key 'ledger', not 'ledger_summary'."
        return (
            False,
            "Trading final_artifact must contain key 'ledger' with transaction row objects; "
            f"observed keys: {observed_keys}.{alias_hint}",
            ["ledger_mismatch", "incomplete_final_artifact"]
        )

    required_ledger_keys = {"tick", "asset", "side", "quantity", "price", "fee", "cash_after"}
    for index, row in enumerate(ledger):
        if not isinstance(row, dict):
            return (
                False,
                f"Trading ledger row {index} must be an object; observed {type(row).__name__}.",
                ["ledger_mismatch", "state_tracking_failure"]
            )
        missing_keys = sorted(required_ledger_keys - set(row))
        if missing_keys:
            alias_hint = " Use key 'quantity', not 'qty'." if "qty" in row and "quantity" in missing_keys else ""
            return (
                False,
                f"Trading ledger row {index} is missing required keys {missing_keys}.{alias_hint} "
                "Every row must include tick, asset, side, quantity, price, fee, and cash_after.",
                ["ledger_mismatch", "state_tracking_failure"]
            )

    positions = portfolio.get("positions", {})
    cash = portfolio.get("cash")
    if positions != expected.get("final_positions") or cash != expected.get("final_cash"):
        expected_positions = expected.get("final_positions", {})
        position_deltas = {
            asset: positions.get(asset, 0) - target
            for asset, target in expected_positions.items()
        }
        extra_assets = sorted(
            asset for asset, quantity in positions.items()
            if asset not in expected_positions and quantity
        )
        return (
            False,
            "The final portfolio does not match the deterministic ledger verifier. "
            f"Public rules imply target positions={expected_positions}; "
            f"observed positions={positions}, cash={cash}, position_deltas={position_deltas}, "
            f"extra_non_target_assets={extra_assets}. "
            "Parse target holdings from rules_text, trade only the remaining deficit for each target asset, "
            "skip distractor assets, stop buying once a target is reached, and include flat fees in cash_after.",
            ["ledger_mismatch", "incorrect_output"]
        )
    normalized_ledger = _normalize_trading_ledger(ledger)
    expected_ledger = _normalize_trading_ledger(expected.get("ledger", []))
    if normalized_ledger != expected_ledger:
        first_diff = difference_location(ledger, expected_ledger)
        return (
            False,
            "The transaction ledger rows do not match the deterministic verifier trace. "
            "Rows must use key 'quantity' rather than aliases such as 'qty'. "
            f"First differing location: {first_diff}. "
            f"Observed first row={ledger[0] if ledger else None}; expected first row={expected.get('ledger', [None])[0] if expected.get('ledger') else None}.",
            ["ledger_mismatch", "state_tracking_failure"]
        )

    note = expected.get("note")
    if note and "impossible" not in str(output.get("limitations", "")).lower():
        return False, "The output reaches the best legal ledger but does not emit the required impossibility limitation.", ["incomplete_final_artifact"]

    return True, "The output contains the verified final portfolio, ledger, and state trace.", []


def _normalize_trading_ledger(ledger: Any) -> Any:
    if not isinstance(ledger, list):
        return ledger
    numeric_keys = {"tick", "quantity", "price", "fee", "cash_after"}
    normalized = []
    for row in ledger:
        if not isinstance(row, dict):
            normalized.append(row)
            continue
        normalized.append({
            key: _normalize_trading_number(value) if key in numeric_keys else value
            for key, value in row.items()
        })
    return normalized


def _normalize_trading_number(value: Any) -> Any:
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _verify_security_output(output: Dict[str, Any], expected: Dict[str, Any]) -> Tuple[bool, str, List[str]]:
    artifact = final_artifact(output)
    expected_proof = expected.get("proof_object")
    if "proof_object" not in artifact and artifact != expected_proof:
        observed_keys = sorted(artifact.keys())
        expected_keys = sorted(expected_proof.keys()) if isinstance(expected_proof, dict) else []
        return (
            False,
            "Security final_artifact must contain key 'proof_object' or be the direct proof object; "
            f"observed artifact keys: {observed_keys}; expected proof keys: {expected_keys}.",
            ["vector_isolation_failure", "incomplete_final_artifact"]
        )
    proof = artifact.get("proof_object", artifact)
    if proof != expected_proof:
        observed_keys = sorted(proof.keys()) if isinstance(proof, dict) else []
        expected_keys = sorted(expected_proof.keys()) if isinstance(expected_proof, dict) else []
        alias_hint = ""
        if isinstance(proof, dict) and "result" in proof and "observed_result" in expected_keys:
            alias_hint = " Use key 'observed_result', not 'result'."
        return (
            False,
            "The proof object does not match the sandbox verifier result. "
            f"Observed proof keys: {observed_keys}; expected proof keys: {expected_keys}."
            f"{alias_hint}",
            ["vector_isolation_failure", "incorrect_output"]
        )
    if not output.get("state_trace"):
        return False, "The proof object is present but the output lacks the required execution trace.", ["missing_execution_trace"]
    return True, "The output contains the verified proof object and sandbox trace.", []


def _verify_matrix_output(output: Dict[str, Any], expected: Dict[str, Any]) -> Tuple[bool, str, List[str]]:
    artifact = final_artifact(output)
    if "answer_set" not in artifact:
        observed_keys = sorted(artifact.keys())
        alias_hint = ""
        if "answers" in artifact:
            alias_hint = " Use key 'answer_set', not 'answers'."
        return (
            False,
            "Matrix final_artifact must contain key 'answer_set' with final node id strings; "
            f"observed keys: {observed_keys}.{alias_hint}",
            ["answer_set_mismatch", "incomplete_final_artifact"]
        )
    raw_answer_set = artifact.get("answer_set", [])
    if not isinstance(raw_answer_set, list) or not all(isinstance(item, str) for item in raw_answer_set):
        return (
            False,
            "Matrix final_artifact.answer_set must be a list of node id strings; "
            f"observed={raw_answer_set!r}.",
            ["answer_set_mismatch", "incorrect_output"]
        )

    answer_set = sorted(raw_answer_set)
    if "paths" not in artifact:
        observed_keys = sorted(artifact.keys())
        alias_hint = ""
        if "path_traces" in artifact:
            alias_hint = " Use key 'paths', not 'path_traces'."
        return (
            False,
            "Matrix final_artifact must contain key 'paths' with full node-id path traces; "
            f"observed keys: {observed_keys}.{alias_hint}",
            ["path_trace_missing", "incorrect_output"]
        )

    paths = artifact.get("paths", [])
    if not isinstance(paths, list) or not all(isinstance(path, list) for path in paths):
        return (
            False,
            "Matrix final_artifact.paths must be a list of path lists; "
            f"observed={paths!r}.",
            ["path_trace_missing", "incorrect_output"]
        )
    if not all(all(isinstance(node_id, str) for node_id in path) for path in paths):
        return (
            False,
            "Matrix final_artifact.paths must contain only node-id strings inside each path list; "
            f"observed first path={paths[0] if paths else None!r}.",
            ["path_trace_missing", "incorrect_output"]
        )

    path_terminal_nodes = sorted({path[-1] for path in paths if path})
    path_start_nodes = sorted({path[0] for path in paths if path})
    path_lengths = sorted({len(path) for path in paths})
    if path_terminal_nodes and answer_set != path_terminal_nodes:
        return (
            False,
            "Matrix final_artifact.answer_set must contain the terminal node ids from "
            "final_artifact.paths. "
            f"Observed answer_set={answer_set}; path_terminal_nodes={path_terminal_nodes}; "
            f"path_start_nodes={path_start_nodes}; path_lengths={path_lengths}; "
            f"observed first path={paths[0] if paths else None!r}. "
            "Return terminal result nodes, not intermediate traversal nodes, and make each "
            "path end at its corresponding answer node.",
            ["answer_set_mismatch", "path_trace_missing"]
        )

    expected_answer_set = sorted(expected.get("answer_set", []))
    expected_paths = expected.get("paths", [])
    if answer_set != expected_answer_set:
        loaded_memory = _memory_shape_summary(output.get("memory")) or _trace_memory_shape_summary(output)
        if not answer_set and not paths and loaded_memory:
            return (
                False,
                "The answer set is empty even though the organ retained non-empty observed memory "
                f"({loaded_memory}). This usually means the runtime parser or traversal/filter logic "
                "discarded all candidates after loading the artifacts. Bind each parsed constraint to "
                "the specific entity, field, or path step named by the task text; do not apply one "
                "property filter globally unless the observed instructions require it.",
                ["answer_set_mismatch", "graph_traversal_failure"]
            )
        return (
            False,
            "The answer set does not match the graph verifier. "
            f"Observed answer_set={answer_set}; "
            f"path_terminal_nodes={path_terminal_nodes}; "
            f"path_start_nodes={path_start_nodes}; path_lengths={path_lengths}. "
            "Recompute terminal result nodes from public graph artifacts and the query contract.",
            ["answer_set_mismatch"]
        )
    if sorted(paths) != sorted(expected_paths):
        first_diff = difference_location(sorted(paths), sorted(expected_paths))
        expected_path_lengths = sorted({len(path) for path in expected_paths if isinstance(path, list)})
        expected_start_nodes = sorted({path[0] for path in expected_paths if isinstance(path, list) and path})
        return (
            False,
            "The path traces do not match the graph verifier. "
            "Emit them under final_artifact.paths, not aliases such as path_traces. "
            f"First differing location: {first_diff}. "
            f"Observed path_start_nodes={path_start_nodes}; expected_start_nodes={expected_start_nodes}; "
            f"observed_path_lengths={path_lengths}; expected_path_lengths={expected_path_lengths}; "
            f"observed first path={paths[0] if paths else None!r}.",
            ["path_trace_missing", "graph_traversal_failure"]
        )
    return True, "The output contains the verified answer set and path traces.", []


def _trace_memory_shape_summary(output: Dict[str, Any]) -> str:
    trace = output.get("state_trace")
    if not isinstance(trace, list):
        return ""

    latest_summary = ""
    for step in trace:
        if not isinstance(step, dict):
            continue
        raw_path = step.get("action_trace")
        if not raw_path:
            continue
        path = Path(str(raw_path))
        if not path.name.endswith("_action.json"):
            continue
        try:
            action = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        raw_action = action.get("raw_action")
        parsed_action = extract_output_object(raw_action) if isinstance(raw_action, str) else None
        if not isinstance(parsed_action, dict):
            continue
        summary = _memory_shape_summary(parsed_action.get("memory"))
        if summary:
            latest_summary = summary
    return latest_summary


def _memory_shape_summary(memory: Any) -> str:
    if not isinstance(memory, dict):
        return ""
    parts = []
    for key, value in sorted(memory.items()):
        if isinstance(value, list) and value:
            parts.append(f"{key}=list[{len(value)}]")
        elif isinstance(value, dict) and value:
            parts.append(f"{key}=object[{len(value)}]")
        elif isinstance(value, str) and value:
            parts.append(f"{key}=text[{len(value)}]")
    return ", ".join(parts[:6])


def _expected_path(payload: Dict[str, Any]) -> Optional[Path]:
    episode_id = payload.get("episode_id")
    domain_id = payload.get("domain_id")
    if not episode_id or not domain_id:
        return None

    domain_dir = {
        "trading_floor": "trading",
        "security_sandbox": "security",
        "matrix_database": "matrix",
    }.get(domain_id)
    if domain_dir is None:
        return None
    return Path("benchmarks") / "private" / domain_dir / episode_id / "expected.json"
