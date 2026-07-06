import csv
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.evaluation.feedback import EnvironmentFeedback
from src.evaluation.stateful_contract import (
    EpisodeRunResult,
    TurnExecutor,
    extract_output_object,
    minimum_turns,
    parse_episode_prompt,
)
from src.evaluation.stateful_verifier import verify_physical_episode


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

        required_turns = max(minimum_turns(self.payload), len(self.payload.get("turns") or []), 1)
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
        parsed = extract_output_object(action_text)
        if parsed is None:
            return ""
        if "final_artifact" in parsed:
            if isinstance(parsed.get("final_artifact"), dict):
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
        return ""

    @staticmethod
    def _infer_final_artifact(parsed: Dict[str, Any]) -> Dict[str, Any]:
        envelope_keys = {
            "action_type",
            "domain_id",
            "episode_id",
            "evidence",
            "final_artifact",
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
            "answer_set",
            "answers",
            "ledger",
            "observed_result",
            "paths",
            "proof",
            "proof_object",
            "result",
            "results",
            "transaction_ledger",
            "vector",
        }
        if parsed.get("action_type") == "submit_final" and artifact:
            return artifact
        if any(key.startswith("final_") or key in artifact_markers for key in artifact):
            return artifact
        return {}

    def _enrich_with_physical_state_trace(self, output: str, required_turns: int) -> str:
        parsed = extract_output_object(output)
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


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    text = Path(path).read_text(encoding="utf-8")
    return list(csv.DictReader(text.splitlines()))
