import csv
import importlib.util
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.evaluation.feedback import EnvironmentFeedback


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


def solve_stateful_episode(task: str) -> str:
    """Solve a v2 stateful benchmark episode using only public artifact paths."""
    payload = parse_episode_prompt(task)
    if payload is None:
        return json.dumps({
            "success": False,
            "error": "Input is not a stateful benchmark episode."
        }, indent=2)

    domain_id = payload.get("domain_id")
    if domain_id == "trading_floor":
        result = _solve_trading(payload)
    elif domain_id == "security_sandbox":
        result = _solve_security(payload)
    elif domain_id == "matrix_database":
        result = _solve_matrix(payload)
    else:
        result = {
            "success": False,
            "error": f"Unsupported domain_id: {domain_id}"
        }
    return json.dumps(result, indent=2, sort_keys=True)


def solve_trading_floor_episode(task: str) -> str:
    """Solve only a trading-floor benchmark episode."""
    return _solve_specific_domain(task, "trading_floor", _solve_trading)


def solve_security_sandbox_episode(task: str) -> str:
    """Solve only a security-sandbox benchmark episode."""
    return _solve_specific_domain(task, "security_sandbox", _solve_security)


def solve_matrix_database_episode(task: str) -> str:
    """Solve only a matrix-database benchmark episode."""
    return _solve_specific_domain(task, "matrix_database", _solve_matrix)


def verify_stateful_episode(task: str, agent_output: str, turns_taken: Optional[int] = None) -> Optional[EnvironmentFeedback]:
    """Deterministically verify v2 episode output when possible."""
    payload = parse_episode_prompt(task)
    if payload is None:
        return None

    if payload.get("domain_id") not in {"trading_floor", "security_sandbox", "matrix_database"}:
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

    output = _extract_output_object(agent_output)
    if output is None:
        return EnvironmentFeedback(
            success=False,
            critique="The output does not contain a parseable final artifact object.",
            identified_gaps=["missing_physical_trace", "incomplete_final_artifact"]
        )

    minimum_turns = _minimum_turns(payload)
    trace = output.get("state_trace")
    if minimum_turns > 0:
        if turns_taken is not None and turns_taken < minimum_turns:
            return EnvironmentFeedback(
                success=False,
                critique=(
                    f"The episode collapsed into {turns_taken} actual turn(s), "
                    f"but the contract requires at least {minimum_turns} stateful turns."
                ),
                identified_gaps=["multi_turn_collapse", "missing_physical_trace"]
            )
        if not isinstance(trace, list) or len(trace) < minimum_turns:
            trace_count = len(trace) if isinstance(trace, list) else 0
            return EnvironmentFeedback(
                success=False,
                critique=(
                    f"The state_trace contains {trace_count} turn(s), "
                    f"but the contract requires at least {minimum_turns} reconstructable turns."
                ),
                identified_gaps=["multi_turn_collapse", "missing_execution_trace"]
            )

    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    domain_id = payload.get("domain_id")

    if domain_id == "trading_floor":
        success, critique, tags = _verify_trading_output(payload, output, expected)
    elif domain_id == "security_sandbox":
        success, critique, tags = _verify_security_output(payload, output, expected)
    elif domain_id == "matrix_database":
        success, critique, tags = _verify_matrix_output(payload, output, expected)
    else:
        success, critique, tags = False, "The episode domain is unsupported by the deterministic verifier.", ["unverifiable_output"]

    return EnvironmentFeedback(success=success, critique=critique, identified_gaps=tags)


def format_stateful_output(agent_output: str) -> str:
    """Return a readable console view of a benchmark answer."""
    output = _extract_output_object(agent_output)
    if output is None:
        return agent_output

    compact = {
        "final_artifact": output.get("final_artifact", output),
        "state_trace": output.get("state_trace", []),
        "evidence": output.get("evidence", []),
        "limitations": output.get("limitations", "none"),
    }
    return json.dumps(compact, indent=2, sort_keys=True)


def _solve_specific_domain(task: str, expected_domain: str, solver) -> str:
    payload = parse_episode_prompt(task)
    if payload is None:
        result = {
            "success": False,
            "error": "Input is not a stateful benchmark episode."
        }
    elif payload.get("domain_id") != expected_domain:
        result = {
            "success": False,
            "error": f"This organ handles {expected_domain}, not {payload.get('domain_id')}."
        }
    else:
        result = solver(payload)
    return json.dumps(result, indent=2, sort_keys=True)


def _solve_trading(payload: Dict[str, Any]) -> Dict[str, Any]:
    artifacts = payload["public_artifacts"]
    market_rows = _read_market_rows(artifacts["market_log"])
    rules_text = Path(artifacts["rules"]).read_text(encoding="utf-8")
    portfolio = json.loads(Path(artifacts["starting_portfolio"]).read_text(encoding="utf-8"))

    targets = _parse_position_targets(rules_text, portfolio["positions"])
    cash_threshold = _parse_cash_threshold(rules_text)
    fee = _parse_fee(rules_text)

    cash = int(portfolio["cash"])
    positions = {asset: int(amount) for asset, amount in portfolio["positions"].items()}
    ledger: List[Dict[str, Any]] = []
    last_trade_tick_by_asset: Dict[str, int] = {}

    for row in market_rows:
        tick = int(row["tick"])
        asset = row["asset"]
        target = targets.get(asset, positions.get(asset, 0))
        current = positions.get(asset, 0)
        needed = target - current
        if needed <= 0:
            continue
        if last_trade_tick_by_asset.get(asset) == tick - 1:
            continue

        quantity = min(needed, int(row["max_buy_qty"]))
        if quantity <= 0:
            continue

        price = int(row["ask"])
        total_cost = quantity * price + fee
        if cash < total_cost:
            continue

        cash -= total_cost
        positions[asset] = current + quantity
        last_trade_tick_by_asset[asset] = tick
        ledger.append({
            "tick": tick,
            "asset": asset,
            "side": "BUY",
            "quantity": quantity,
            "price": price,
            "fee": fee,
            "cash_after": cash
        })

    final_positions = {asset: positions.get(asset, 0) for asset in targets}
    goals_satisfied = final_positions == targets
    cash_threshold_satisfied = cash_threshold is None or cash >= cash_threshold
    impossible = bool(goals_satisfied and cash_threshold is not None and cash < cash_threshold)

    limitations = "none"
    if impossible:
        limitations = (
            f"Position targets are reachable, but the best legal ledger ends with cash {cash}, "
            f"below the threshold {cash_threshold}; the cash threshold is impossible under visible rules."
        )

    return {
        "domain_id": payload["domain_id"],
        "episode_id": payload["episode_id"],
        "final_artifact": {
            "final_portfolio": {
                "cash": cash,
                "positions": final_positions
            },
            "ledger": ledger,
            "goals_satisfied": goals_satisfied,
            "cash_threshold_satisfied": cash_threshold_satisfied,
            "impossible_cash_threshold": impossible
        },
        "state_trace": [
            f"turn_1: loaded starting cash {portfolio['cash']} and positions {portfolio['positions']}",
            f"turn_2: parsed {len(market_rows)} market rows and fee {fee}",
            f"turn_3: selected {len(ledger)} legal BUY actions toward target positions {targets}",
            f"turn_4: applied fills, flat fees, max quantities, and cooldown checks",
            f"turn_5: final cash {cash}, final positions {final_positions}"
        ],
        "evidence": list(artifacts.values()),
        "limitations": limitations
    }


def _solve_security(payload: Dict[str, Any]) -> Dict[str, Any]:
    artifacts = payload["public_artifacts"]
    source_dir = Path(artifacts["source_dir"])
    candidates_path = Path(artifacts["candidate_vectors"])
    candidates = [line.strip() for line in candidates_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    proof: Dict[str, str] = {}
    observations: List[str] = []

    toy_service = source_dir / "toy_service.py"
    parser = source_dir / "parser.py"
    if toy_service.exists():
        module = _load_module(toy_service, f"{payload['episode_id']}_toy_service")
        for vector in candidates:
            observed = str(module.fetch_document(vector))
            observations.append(f"{vector} -> {observed}")
            if observed.startswith("sandbox-token"):
                proof = {"vector": vector, "observed_result": observed}
                break
    elif parser.exists():
        module = _load_module(parser, f"{payload['episode_id']}_parser")
        for packet in candidates:
            observed = str(module.authorize(packet))
            observations.append(f"{packet} -> {observed}")
            if observed.startswith("GRANTED:"):
                proof = {"packet": packet, "observed_result": observed}
                break

    return {
        "domain_id": payload["domain_id"],
        "episode_id": payload["episode_id"],
        "final_artifact": {
            "proof_object": proof
        },
        "state_trace": [
            f"turn_1: enumerated {source_dir}",
            f"turn_2: inspected source files {[path.name for path in source_dir.glob('*.py')]}",
            f"turn_3: tested {len(candidates)} candidate vectors",
            f"turn_4: observations {observations}",
            f"turn_5: selected proof object {proof}"
        ],
        "evidence": [str(source_dir), str(candidates_path)],
        "limitations": "none"
    }


def _solve_matrix(payload: Dict[str, Any]) -> Dict[str, Any]:
    artifacts = payload["public_artifacts"]
    nodes = json.loads(Path(artifacts["nodes"]).read_text(encoding="utf-8"))
    edges = json.loads(Path(artifacts["edges"]).read_text(encoding="utf-8"))
    query = Path(artifacts["query"]).read_text(encoding="utf-8")

    node_by_id = {node["id"]: node for node in nodes}
    seed = _parse_seed(query)
    relation_chain = _parse_relation_chain(query)
    raw_paths = _traverse_relation_chain(seed, relation_chain, edges)
    filtered_paths = [path for path in raw_paths if _path_matches_query_filters(path, node_by_id, query)]
    answer_set = sorted({path[-1] for path in filtered_paths})

    return {
        "domain_id": payload["domain_id"],
        "episode_id": payload["episode_id"],
        "final_artifact": {
            "answer_set": answer_set,
            "paths": filtered_paths
        },
        "state_trace": [
            f"turn_1: loaded {len(nodes)} nodes and {len(edges)} edges from public artifacts",
            f"turn_2: expanded seed {seed} through {relation_chain[0] if relation_chain else 'none'}",
            f"turn_3: applied relation chain {' -> '.join(relation_chain)}",
            f"turn_4: retained {len(filtered_paths)} paths after property filters",
            f"turn_5: answer set {answer_set}"
        ],
        "evidence": list(artifacts.values()),
        "limitations": "none"
    }


def _verify_trading_output(payload: Dict[str, Any], output: Dict[str, Any], expected: Dict[str, Any]) -> Tuple[bool, str, List[str]]:
    artifact = _final_artifact(output)
    if "final_portfolio" not in artifact:
        return (
            False,
            "Trading final_artifact must contain key 'final_portfolio'; "
            f"observed keys: {sorted(artifact.keys())}.",
            ["incomplete_final_artifact"]
        )

    portfolio = artifact.get("final_portfolio", {})
    positions = portfolio.get("positions", {})
    cash = portfolio.get("cash")
    ledger = artifact.get("ledger", [])

    if positions != expected.get("final_positions") or cash != expected.get("final_cash"):
        return (
            False,
            "The final portfolio does not match the deterministic ledger verifier. "
            "Recompute the artifact from public inputs and keep the ledger and final portfolio internally consistent.",
            ["ledger_mismatch", "incorrect_output"]
        )
    if ledger != expected.get("ledger"):
        expected_ledger = expected.get("ledger")
        first_diff = _difference_location(ledger, expected_ledger)
        return (
            False,
            "The transaction ledger rows do not match the deterministic verifier trace. "
            "Rows must use key 'quantity' rather than aliases such as 'qty'. "
            f"First differing location: {first_diff}",
            ["ledger_mismatch", "state_tracking_failure"]
        )

    note = expected.get("note")
    if note and "impossible" not in str(output.get("limitations", "")).lower():
        return False, "The output reaches the best legal ledger but does not emit the required impossibility limitation.", ["incomplete_final_artifact"]

    return True, "The output contains the verified final portfolio, ledger, and state trace.", []


def _verify_security_output(payload: Dict[str, Any], output: Dict[str, Any], expected: Dict[str, Any]) -> Tuple[bool, str, List[str]]:
    artifact = _final_artifact(output)
    proof = artifact.get("proof_object", artifact)
    expected_proof = expected.get("proof_object")
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


def _verify_matrix_output(payload: Dict[str, Any], output: Dict[str, Any], expected: Dict[str, Any]) -> Tuple[bool, str, List[str]]:
    artifact = _final_artifact(output)
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

    expected_answer_set = sorted(expected.get("answer_set", []))
    expected_paths = expected.get("paths", [])
    if answer_set != expected_answer_set:
        return (
            False,
            "The answer set does not match the graph verifier. "
            "Recompute the result from public graph artifacts and the query contract.",
            ["answer_set_mismatch"]
        )
    if sorted(paths) != sorted(expected_paths):
        first_diff = _difference_location(sorted(paths), sorted(expected_paths))
        return (
            False,
            "The path traces do not match the graph verifier. "
            "Emit them under final_artifact.paths, not aliases such as path_traces. "
            f"First differing location: {first_diff}",
            ["path_trace_missing", "graph_traversal_failure"]
        )
    return True, "The output contains the verified answer set and path traces.", []


def _read_market_rows(path: str) -> List[Dict[str, str]]:
    text = Path(path).read_text(encoding="utf-8")
    return list(csv.DictReader(text.splitlines()))


def _parse_position_targets(rules_text: str, starting_positions: Dict[str, Any]) -> Dict[str, int]:
    targets = {asset: int(amount) for asset, amount in starting_positions.items()}
    for amount, asset in re.findall(r"Hold exactly (\d+) ([A-Z]+)", rules_text):
        targets[asset] = int(amount)
    for asset in re.findall(r"Hold 0 ([A-Z]+)", rules_text):
        targets[asset] = 0
    return targets


def _parse_cash_threshold(rules_text: str) -> Optional[int]:
    match = re.search(r"cash\s*>=\s*(\d+)", rules_text)
    return int(match.group(1)) if match else None


def _parse_fee(rules_text: str) -> int:
    match = re.search(r"fee of (\d+)", rules_text)
    return int(match.group(1)) if match else 0


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_seed(query: str) -> str:
    match = re.search(r"Seed node:\s*(\S+)", query)
    if not match:
        raise ValueError("Query does not contain a seed node")
    return match.group(1)


def _parse_relation_chain(query: str) -> List[str]:
    lines = [line.strip() for line in query.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        if "relation chain" in line and index + 1 < len(lines):
            return [part.strip() for part in lines[index + 1].split("->")]
    raise ValueError("Query does not contain a relation chain")


def _traverse_relation_chain(seed: str, relation_chain: List[str], edges: List[Dict[str, str]]) -> List[List[str]]:
    paths = [[seed]]
    for relation in relation_chain:
        next_paths: List[List[str]] = []
        for path in paths:
            tail = path[-1]
            for edge in edges:
                if edge["from"] == tail and edge["relation"] == relation:
                    next_paths.append(path + [edge["to"]])
        paths = next_paths
    return paths


def _path_matches_query_filters(path: List[str], node_by_id: Dict[str, Dict[str, Any]], query: str) -> bool:
    final_node = node_by_id[path[-1]]
    if "active" in query and final_node.get("status") != "active":
        return False
    if "tier-2" in query and final_node.get("tier") != 2:
        return False
    if "risk must be >= 8" in query and int(final_node.get("risk", -1)) < 8:
        return False

    for node_id in path[1:-1]:
        node = node_by_id[node_id]
        if "color=green" in query and node.get("type") == "archive" and node.get("color") != "green":
            return False
        if "class=transactional" in query and node.get("type") == "table" and node.get("class") != "transactional":
            return False
    return True


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


def _minimum_turns(payload: Dict[str, Any]) -> int:
    contract = payload.get("episode_contract", {})
    if not isinstance(contract, dict):
        return 0
    try:
        return int(contract.get("minimum_turns") or 0)
    except (TypeError, ValueError):
        return 0


def _extract_output_object(agent_output: str) -> Optional[Dict[str, Any]]:
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


def _final_artifact(output: Dict[str, Any]) -> Dict[str, Any]:
    artifact = output.get("final_artifact", output)
    return artifact if isinstance(artifact, dict) else {}


def _difference_location(observed: Any, expected: Any) -> str:
    if not isinstance(observed, list) or not isinstance(expected, list):
        return "container shape differs"
    if len(observed) != len(expected):
        return f"length differs; observed length {len(observed)}"
    for index, (observed_row, expected_row) in enumerate(zip(observed, expected)):
        if observed_row != expected_row:
            return f"row {index}"
    return "no row-level difference found"
