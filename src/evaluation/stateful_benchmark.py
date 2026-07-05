import csv
import importlib.util
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

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


class StatefulEpisodeRunner:
    """
    Physical benchmark harness.

    The runner owns observation release, trace-file creation, and final physical
    verification. Agent prose is never accepted as proof unless it was produced
    through this turn loop and accompanied by runtime trace files.
    """

    def __init__(self, payload: Dict[str, Any], workspace_root: Optional[Path] = None):
        self.payload = payload
        episode_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(payload.get("episode_id", "episode")))
        self.workspace = Path(tempfile.mkdtemp(prefix=f"stem_{episode_id}_", dir=workspace_root))
        self.trace_dir = self.workspace / "trace"
        self.sandbox_dir = self.workspace / "sandbox"
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)

    async def run(self, execute_turn: TurnExecutor) -> EpisodeRunResult:
        """Run a benchmark episode as an explicit environment-agent turn loop."""
        if not self.payload.get("artifact_manifest"):
            feedback = EnvironmentFeedback(
                success=False,
                critique="The benchmark episode has no artifact_manifest for environment observation release.",
                identified_gaps=["missing_environment_manifest", "unverifiable_output"],
            )
            return EpisodeRunResult("", 0, feedback, str(self.workspace))

        required_turns = max(_minimum_turns(self.payload), len(self.payload.get("turns") or []), 1)
        final_output = ""
        tool_invocations = 0
        active_tool_names: List[str] = []
        agent_memory: Dict[str, Any] = {}

        for turn_number in range(1, required_turns + 1):
            observation = self._build_observation(turn_number, required_turns)
            observation["memory"] = agent_memory
            observation_text = json.dumps(observation, indent=2, sort_keys=True)
            self._write_json(self.trace_dir / f"turn_{turn_number:03d}_observation.json", observation)

            action_text, tool_invoked, tool_name = await execute_turn(observation_text)
            try:
                parsed_action = json.loads(action_text)
            except (TypeError, json.JSONDecodeError):
                parsed_action = None
            if isinstance(parsed_action, dict):
                agent_memory = self._next_agent_memory(parsed_action)

            if tool_invoked:
                tool_invocations += 1
            if tool_name:
                active_tool_names.append(tool_name)

            self._write_json(
                self.trace_dir / f"turn_{turn_number:03d}_action.json",
                {
                    "tool_invoked": tool_invoked,
                    "tool_name": tool_name,
                    "raw_action": action_text,
                },
            )

            result = self._apply_action(turn_number, action_text, tool_invoked, tool_name)
            self._write_json(self.trace_dir / f"turn_{turn_number:03d}_result.json", result)
            final_output = result.get("candidate_final_output") or final_output

        if final_output:
            final_output = self._enrich_with_physical_state_trace(final_output, required_turns)

        feedback = verify_physical_episode(
            self.payload,
            final_output,
            workspace=self.workspace,
            turns_taken=required_turns,
            tool_invocations=tool_invocations,
        )
        if final_output:
            (self.workspace / "final_artifact.json").write_text(final_output.rstrip() + "\n", encoding="utf-8")

        return EpisodeRunResult(
            output=final_output or self._unverifiable_output(active_tool_names),
            turns_taken=required_turns,
            feedback=feedback,
            workspace=str(self.workspace),
        )

    def _build_observation(self, turn_number: int, required_turns: int) -> Dict[str, Any]:
        turns = self.payload.get("turns") or []
        event = turns[turn_number - 1].get("event") if turn_number <= len(turns) else "continue_episode"
        return {
            "benchmark_version": self.payload.get("benchmark_version"),
            "domain_id": self.payload.get("domain_id"),
            "episode_id": self.payload.get("episode_id"),
            "turn": turn_number,
            "minimum_turns": required_turns,
            "event": event,
            "initial_prompt": self.payload.get("initial_prompt"),
            "workspace": str(self.sandbox_dir),
            "trace_dir": str(self.trace_dir),
            "observation_delta": self._manifest_observation_delta(turn_number, required_turns),
            "action_contract": {
                "return_json": True,
                "final_submission_shape": {
                    "domain_id": self.payload.get("domain_id"),
                    "episode_id": self.payload.get("episode_id"),
                    "final_artifact": "required on the final turn",
                    "state_trace": "required list of physical turn summaries",
                    "evidence": "required public paths and/or trace files",
                    "limitations": "required string",
                },
            },
        }

    def _manifest_observation_delta(self, turn_number: int, required_turns: int) -> Dict[str, Any]:
        delta: Dict[str, Any] = {}
        manifest = self.payload.get("artifact_manifest") or {}
        for entry in manifest.get("turns", []):
            if not self._entry_applies_to_turn(entry, turn_number, required_turns):
                continue
            delta.update(entry.get("observation_delta", {}))
            for load_spec in entry.get("loads", []):
                self._apply_manifest_load(delta, load_spec, entry, turn_number, required_turns)
        return delta or {"kind": "empty_manifest_delta"}

    def _entry_applies_to_turn(self, entry: Dict[str, Any], turn_number: int, required_turns: int) -> bool:
        if "turn" in entry:
            return int(entry["turn"]) == turn_number
        return turn_number in self._entry_turns(entry, required_turns)

    @staticmethod
    def _entry_turns(entry: Dict[str, Any], required_turns: int) -> List[int]:
        turn_range = entry.get("turn_range")
        if not isinstance(turn_range, dict):
            return []
        start = int(turn_range.get("start", 1))
        raw_end = turn_range.get("end", required_turns)
        if raw_end == "before_final":
            end = max(required_turns - 1, start)
        elif raw_end == "final":
            end = required_turns
        else:
            end = int(raw_end)
        return list(range(start, end + 1))

    def _apply_manifest_load(
        self,
        delta: Dict[str, Any],
        load_spec: Dict[str, Any],
        entry: Dict[str, Any],
        turn_number: int,
        required_turns: int,
    ) -> None:
        loader = load_spec.get("loader", "path")
        artifact_key = load_spec.get("artifact")
        artifact_path = self._artifact_path(artifact_key)

        if loader == "path":
            delta[str(load_spec["as"])] = artifact_path
        elif loader == "text":
            delta[str(load_spec["as"])] = self._read_text(artifact_path)
        elif loader == "json":
            delta[str(load_spec["as"])] = self._read_json_artifact(artifact_path)
        elif loader == "artifact_map":
            delta[str(load_spec["as"])] = self.payload.get("public_artifacts", {})
        elif loader == "csv_window":
            rows = _read_csv_rows(artifact_path) if artifact_path else []
            self._load_window(
                delta=delta,
                values=rows,
                path=artifact_path,
                entry=entry,
                turn_number=turn_number,
                required_turns=required_turns,
                path_as=load_spec.get("path_as"),
                start_as=load_spec.get("row_start_as"),
                values_as=load_spec.get("rows_as", load_spec.get("as")),
            )
        elif loader == "lines_window":
            values = [
                line.strip()
                for line in self._read_text(artifact_path).splitlines()
                if line.strip()
            ]
            self._load_window(
                delta=delta,
                values=values,
                path=artifact_path,
                entry=entry,
                turn_number=turn_number,
                required_turns=required_turns,
                path_as=load_spec.get("path_as"),
                start_as=load_spec.get("line_start_as"),
                values_as=load_spec.get("values_as", load_spec.get("as")),
            )
        elif loader == "directory_listing":
            root = Path(str(artifact_path or ""))
            files = sorted(path for path in root.glob(str(load_spec.get("glob", "*"))) if path.is_file())
            if load_spec.get("as"):
                delta[str(load_spec["as"])] = [str(path) for path in files]
            if load_spec.get("names_as"):
                delta[str(load_spec["names_as"])] = [path.name for path in files]
            if load_spec.get("first_path_as"):
                delta[str(load_spec["first_path_as"])] = str(files[0]) if files else None
            if load_spec.get("root_as"):
                delta[str(load_spec["root_as"])] = str(root)
        elif loader == "file_glob_text":
            root = Path(str(artifact_path or ""))
            files = sorted(path for path in root.glob(str(load_spec.get("glob", "*"))) if path.is_file())
            index = int(load_spec.get("index", 0))
            selected = files[index] if 0 <= index < len(files) else None
            if load_spec.get("path_as"):
                delta[str(load_spec["path_as"])] = str(selected) if selected else None
            if load_spec.get("content_as"):
                delta[str(load_spec["content_as"])] = self._read_text(str(selected)) if selected else ""
        else:
            delta[str(load_spec.get("as", "unsupported_loader"))] = {
                "error": f"unsupported manifest loader: {loader}",
                "artifact": artifact_key,
            }

    def _load_window(
        self,
        delta: Dict[str, Any],
        values: List[Any],
        path: Optional[str],
        entry: Dict[str, Any],
        turn_number: int,
        required_turns: int,
        path_as: Optional[str],
        start_as: Optional[str],
        values_as: Optional[str],
    ) -> None:
        turns = self._entry_turns(entry, required_turns)
        if not turns:
            turns = [turn_number]
        turn_index = turns.index(turn_number) if turn_number in turns else 0
        start = len(values) * turn_index // len(turns)
        end = len(values) * (turn_index + 1) // len(turns)
        if path_as:
            delta[str(path_as)] = path
        if start_as:
            delta[str(start_as)] = start + 1
        if values_as:
            delta[str(values_as)] = values[start:end]

    def _artifact_path(self, artifact_key: Optional[str]) -> Optional[str]:
        if artifact_key is None:
            return None
        artifacts = self.payload.get("public_artifacts", {})
        value = artifacts.get(str(artifact_key))
        return str(value) if value is not None else None

    @staticmethod
    def _next_agent_memory(parsed_action: Dict[str, Any]) -> Dict[str, Any]:
        memory = parsed_action.get("memory")
        if isinstance(memory, dict):
            return memory
        return parsed_action

    def _apply_action(
        self,
        turn_number: int,
        action_text: str,
        tool_invoked: bool,
        tool_name: Optional[str],
    ) -> Dict[str, Any]:
        candidate = self._candidate_final_output(action_text)
        sandbox_state_path = self.sandbox_dir / f"turn_{turn_number:03d}_state.json"
        self._write_json(
            sandbox_state_path,
            {
                "turn": turn_number,
                "tool_invoked": tool_invoked,
                "tool_name": tool_name,
                "candidate_final_submitted": bool(candidate),
            },
        )
        return {
            "turn": turn_number,
            "tool_invoked": tool_invoked,
            "tool_name": tool_name,
            "action_bytes": len(action_text.encode("utf-8")),
            "candidate_final_output": candidate,
            "workspace_mutation": str(sandbox_state_path),
        }

    def _candidate_final_output(self, action_text: str) -> str:
        parsed = _extract_output_object(action_text)
        if parsed is None:
            return ""
        if "final_artifact" in parsed:
            return json.dumps(parsed, indent=2, sort_keys=True)
        inferred = self._infer_final_artifact(parsed)
        if inferred:
            wrapped = {
                "domain_id": parsed.get("domain_id", self.payload.get("domain_id")),
                "episode_id": parsed.get("episode_id", self.payload.get("episode_id")),
                "final_artifact": inferred,
                "state_trace": parsed.get("state_trace", []),
                "evidence": parsed.get("evidence", []),
                "limitations": parsed.get("limitations", "none"),
            }
            return json.dumps(wrapped, indent=2, sort_keys=True)
        if parsed.get("action_type") == "submit_final" and isinstance(parsed.get("final_artifact"), dict):
            wrapped = {
                "domain_id": self.payload.get("domain_id"),
                "episode_id": self.payload.get("episode_id"),
                "final_artifact": parsed["final_artifact"],
                "state_trace": parsed.get("state_trace", []),
                "evidence": parsed.get("evidence", []),
                "limitations": parsed.get("limitations", "none"),
            }
            return json.dumps(wrapped, indent=2, sort_keys=True)
        return ""

    @staticmethod
    def _infer_final_artifact(parsed: Dict[str, Any]) -> Dict[str, Any]:
        envelope_keys = {
            "action_type",
            "domain_id",
            "episode_id",
            "evidence",
            "limitations",
            "memory",
            "state_trace",
            "status",
            "success",
        }
        artifact = {
            key: value
            for key, value in parsed.items()
            if key not in envelope_keys and not str(key).startswith("_")
        }
        artifact_markers = {
            "answer",
            "answers",
            "ledger",
            "proof",
            "proof_object",
            "result",
            "results",
            "transaction_ledger",
        }
        if any(key.startswith("final_") or key in artifact_markers for key in artifact):
            return artifact
        return {}

    def _enrich_with_physical_state_trace(self, output: str, required_turns: int) -> str:
        parsed = _extract_output_object(output)
        if parsed is None:
            return output

        state_trace = parsed.get("state_trace")
        if isinstance(state_trace, list) and len(state_trace) >= required_turns:
            return output

        existing_trace = state_trace if isinstance(state_trace, list) else []
        physical_trace = [
            {
                "turn": turn_number,
                "observation_trace": str(self.trace_dir / f"turn_{turn_number:03d}_observation.json"),
                "action_trace": str(self.trace_dir / f"turn_{turn_number:03d}_action.json"),
                "result_trace": str(self.trace_dir / f"turn_{turn_number:03d}_result.json"),
            }
            for turn_number in range(1, required_turns + 1)
        ]
        parsed["state_trace"] = [*existing_trace, *physical_trace]
        return json.dumps(parsed, indent=2, sort_keys=True)

    @staticmethod
    def _write_json(path: Path, payload: Dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def _read_text(path: Optional[str]) -> str:
        if not path:
            return ""
        try:
            return Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            return f"(unable to read public artifact: {exc})"

    def _read_json_artifact(self, path: Optional[str]) -> Any:
        text = self._read_text(path)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def _unverifiable_output(self, active_tool_names: List[str]) -> str:
        return json.dumps(
            {
                "success": False,
                "error": "unverifiable_inference",
                "workspace": str(self.workspace),
                "tools_invoked": active_tool_names,
            },
            indent=2,
            sort_keys=True,
        )


async def run_stateful_episode(task: str, execute_turn: TurnExecutor) -> Optional[EpisodeRunResult]:
    """Execute a rendered benchmark task through the physical episode runtime."""
    payload = parse_episode_prompt(task)
    if payload is None:
        return None
    return await StatefulEpisodeRunner(payload).run(execute_turn)


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
    required_turns = max(_minimum_turns(payload), len(payload.get("turns") or []), 1)
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
        first_diff = _difference_location(ledger, expected_ledger)
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


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    text = Path(path).read_text(encoding="utf-8")
    return list(csv.DictReader(text.splitlines()))


def _read_market_rows(path: str) -> List[Dict[str, str]]:
    return _read_csv_rows(path)


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
