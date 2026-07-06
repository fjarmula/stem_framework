import json
import re
from pathlib import Path
from typing import Any, Dict, List

from src.core.genome import AgentGenome, TransformationPlan, CapabilityModel
from src.evaluation.feedback import EnvironmentFeedback
from src.evaluation.stateful_benchmark import parse_episode_prompt
from src.execution.tools import TOOL_MAPPING
from src.services.llm import LLMService
from src.services.prompts import PromptManager


DISABLED_STATIC_BENCHMARK_TOOLS = {
    "trading_floor_solver",
    "security_sandbox_solver",
    "matrix_database_solver",
}


class EvolutionEngine:
    """
    The 'Environmental Signal' that pushes agents to evolve.
    """

    def __init__(self, llm: LLMService, prompt_manager: PromptManager):
        self.llm = llm
        self.prompt_manager = prompt_manager

    async def propose_differentiation(
            self,
            task_context: str,
            failure_feedback: EnvironmentFeedback,
            current_genome: AgentGenome,
            failed_output: str = "",
            mutation_rejection_feedback: str = "",
            failure_history: List[Any] | None = None,
            clinical_trial_history: List[Any] | None = None,
            repair_target_organ: str | None = None,
    ) -> TransformationPlan:
        """Analyzes a failure and proposes a mutation."""
        payload = parse_episode_prompt(task_context)
        mutation_mode = (
            "REPAIR_EXISTING_ORGAN"
            if repair_target_organ
            else "NOVEL_DIFFERENTIATION"
        )

        # Expose support tools, but hide prebuilt benchmark solvers so evolution
        # must synthesize and register a new runtime organ for benchmark pressure.
        available_tools = "\n".join(
            f"- {tool_name}: registered runtime tool"
            for tool_name in sorted(TOOL_MAPPING)
            if tool_name not in DISABLED_STATIC_BENCHMARK_TOOLS
        )
        if not available_tools:
            available_tools = "(none)"

        prompt = self.prompt_manager.get_prompt(
            "evolution_engine.txt",
            current_genome_json=current_genome.model_dump_json(indent=2),
            task_context=self._sanitized_task_context(task_context, payload),
            public_artifact_observations=self._public_artifact_observations(payload),
            current_generated_organs=self._current_generated_organs(current_genome),
            failed_output_excerpt=self._excerpt(failed_output),
            mutation_rejection_feedback=mutation_rejection_feedback or "(no previous mutation rejection)",
            phenotypic_scars=self._format_failure_history(failure_history or []),
            mutation_mode=mutation_mode,
            repair_target_organ=repair_target_organ or "(none)",
            clinical_trial_postmortem=self._format_clinical_trial_history(
                clinical_trial_history or []
            ),
            success=failure_feedback.success,
            critique=failure_feedback.critique,
            identified_gaps=', '.join(failure_feedback.identified_gaps),
            available_tools=available_tools
        )
        plan = await self.llm.get_structured_completion(
            "You are a Master AI Systems Architect.",
            prompt,
            TransformationPlan
        )
        self._annotate_generated_capability_context(plan, payload)
        return plan

    @staticmethod
    def _sanitized_task_context(task_context: str, payload: dict | None) -> str:
        """Hide train-instance constants while preserving the runtime contract."""
        if payload is None:
            return task_context

        sanitized = {
            key: value
            for key, value in payload.items()
            if key not in {"public_artifacts", "private_verifier_artifacts", "artifact_manifest"}
        }
        sanitized["episode_id"] = "(provided at runtime)"
        sanitized["public_artifacts"] = {
            label: "(runtime path withheld; use observation_delta materialized values)"
            for label in (payload.get("public_artifacts") or {})
        }
        sanitized["artifact_manifest_runtime_contract"] = EvolutionEngine._manifest_runtime_contract(
            payload.get("artifact_manifest") or {}
        )
        return (
            "STATEFUL STEM-CELL BENCHMARK EPISODE\n"
            "Training constants and artifact paths are withheld from the mutation prompt. "
            "The compiled organ must derive values from runtime observations.\n\n"
            f"{json.dumps(sanitized, indent=2, sort_keys=True)}"
        )

    @staticmethod
    def _manifest_runtime_contract(manifest: dict) -> List[Dict[str, Any]]:
        entries = []
        for entry in manifest.get("turns", []):
            materialized_keys = {}
            for load_spec in entry.get("loads", []):
                loader = str(load_spec.get("loader", "path"))
                key_types = EvolutionEngine._manifest_load_key_types(loader, load_spec)
                materialized_keys.update(key_types)
            entries.append({
                "turn": entry.get("turn"),
                "turn_range": entry.get("turn_range"),
                "observation_delta": entry.get("observation_delta", {}),
                "runtime_observation_delta_keys": dict(sorted(materialized_keys.items())),
                "runtime_contract_note": (
                    "These keys are inserted directly into observation_delta at runtime; "
                    "there is no nested materialized_observation_delta object."
                ),
            })
        return entries

    @staticmethod
    def _manifest_load_key_types(loader: str, load_spec: Dict[str, Any]) -> Dict[str, str]:
        key_types: Dict[str, str] = {}
        if loader == "path":
            if load_spec.get("as"):
                key_types[str(load_spec["as"])] = "string path"
        elif loader == "text":
            if load_spec.get("as"):
                key_types[str(load_spec["as"])] = "string text"
        elif loader == "json":
            if load_spec.get("as"):
                key_types[str(load_spec["as"])] = "parsed JSON value, not a JSON string"
        elif loader == "artifact_map":
            if load_spec.get("as"):
                key_types[str(load_spec["as"])] = "object map"
        elif loader == "csv_window":
            if load_spec.get("path_as"):
                key_types[str(load_spec["path_as"])] = "string path"
            if load_spec.get("row_start_as"):
                key_types[str(load_spec["row_start_as"])] = "integer row offset"
            key_types[str(load_spec.get("rows_as", load_spec.get("as", "rows")))] = "list of row objects"
        elif loader == "lines_window":
            if load_spec.get("path_as"):
                key_types[str(load_spec["path_as"])] = "string path"
            if load_spec.get("line_start_as"):
                key_types[str(load_spec["line_start_as"])] = "integer line offset"
            key_types[str(load_spec.get("values_as", load_spec.get("as", "values")))] = "list of strings"
        elif loader == "directory_listing":
            for key in ("as", "names_as"):
                if load_spec.get(key):
                    key_types[str(load_spec[key])] = "list of strings"
            for key in ("root_as", "first_path_as"):
                if load_spec.get(key):
                    key_types[str(load_spec[key])] = "string path or null"
        elif loader == "file_glob_text":
            if load_spec.get("path_as"):
                key_types[str(load_spec["path_as"])] = "string path or null"
            if load_spec.get("content_as"):
                key_types[str(load_spec["content_as"])] = "string text"
        elif load_spec.get("as"):
            key_types[str(load_spec["as"])] = "unknown loader result"
        return key_types

    @staticmethod
    def _annotate_generated_capability_context(
            plan: TransformationPlan,
            payload: dict | None
    ) -> None:
        """Attach benchmark routing metadata that generated code cannot infer later."""
        if payload is None or not plan.new_tool_implementation:
            return

        domain_id = payload.get("domain_id")
        if not domain_id:
            return

        domain_marker = f"domain_id:{domain_id}"
        for capability in plan.added_capabilities:
            if domain_marker not in capability.required_context:
                capability.required_context.append(domain_marker)
            if "stateful benchmark observation JSON" not in capability.required_context:
                capability.required_context.append("stateful benchmark observation JSON")

    @staticmethod
    def _public_artifact_observations(payload: dict | None) -> str:
        """Expose artifact schemas without leaking train-instance constants."""
        if payload is None:
            return "(not a stateful benchmark episode)"

        artifacts = payload.get("public_artifacts", {})
        if not isinstance(artifacts, dict) or not artifacts:
            return "(no public artifacts listed)"

        observations = []
        for label, raw_path in artifacts.items():
            path = Path(str(raw_path))
            if path.is_dir():
                files = sorted(child for child in path.rglob("*") if child.is_file())
                observations.append(f"## {label}")
                observations.append(f"- directory with {len(files)} visible file(s)")
                observations.extend(
                    f"- file suffix: {child.suffix or '(none)'}"
                    for child in files[:8]
                )
            else:
                observations.append(f"## {label}")
                observations.append(EvolutionEngine._artifact_schema_summary(path))
        return "\n".join(observations)

    @staticmethod
    def _artifact_schema_summary(path: Path) -> str:
        suffix = path.suffix.lower()
        text = EvolutionEngine._read_public_text(path)
        if text.startswith("(unable to read public artifact:"):
            return text

        if suffix == ".csv":
            header = text.splitlines()[0] if text.splitlines() else ""
            columns = [column.strip() for column in header.split(",") if column.strip()]
            return f"- CSV columns: {columns}"
        if suffix == ".json":
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return "- JSON artifact with unparsable preview withheld"
            return f"- JSON schema: {EvolutionEngine._json_shape(parsed)}"
        return EvolutionEngine._text_schema_summary(text)

    @staticmethod
    def _text_schema_summary(text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        templates = [
            EvolutionEngine._redact_public_text_line(line)
            for line in lines[:12]
        ]
        return (
            f"- text artifact with {len(text.splitlines())} line(s); "
            f"redacted line templates: {templates}"
        )

    @staticmethod
    def _redact_public_text_line(line: str) -> str:
        redacted = line
        redacted = re.sub(r"(['\"])(.*?)(\1)", r"\1<value>\3", redacted)
        redacted = re.sub(r"\b[\w.-]+/[\w./-]+\b", "<path>", redacted)
        redacted = re.sub(r"\b[A-Za-z_]+_\d+\b", "<identifier>", redacted)
        redacted = re.sub(r"\b[A-Za-z][A-Za-z0-9_-]*\.[A-Za-z0-9_.-]+\b", "<identifier>", redacted)
        redacted = re.sub(r"=\s*[^,\s.;]+", "=<value>", redacted)
        redacted = re.sub(r"\b[A-Z][A-Z0-9_]{1,}\b", "<SYMBOL>", redacted)
        redacted = re.sub(r"\b\d+(?:\.\d+)?\b", "<number>", redacted)
        redacted = re.sub(r"\bReturn\s+[a-z][a-z0-9_-]*\b", "Return <value>", redacted)
        redacted = re.sub(r"\b([a-z][a-z0-9_-]*)\s+(node|nodes)\b", r"<value> \2", redacted)
        redacted = re.sub(r"\breturned\s+[a-z][a-z0-9_-]*\b", "returned <value>", redacted)
        return redacted

    @staticmethod
    def _json_shape(value: Any, depth: int = 0) -> Any:
        if depth >= 3:
            return EvolutionEngine._json_type_name(value)
        if isinstance(value, dict):
            return {
                "type": "object",
                "keys": {
                    str(key): EvolutionEngine._json_shape(child, depth + 1)
                    for key, child in sorted(value.items())
                },
            }
        if isinstance(value, list):
            return EvolutionEngine._json_array_shape(value, depth)
        return EvolutionEngine._json_type_name(value)

    @staticmethod
    def _json_array_shape(values: List[Any], depth: int) -> Any:
        if not values:
            return {"type": "array", "length": 0, "items": "unknown"}

        inspected = values[:50]
        if all(isinstance(item, dict) for item in inspected):
            key_types: Dict[str, set[str]] = {}
            for item in inspected:
                for key, child in item.items():
                    key_types.setdefault(str(key), set()).add(
                        EvolutionEngine._json_type_name(child)
                    )
            return {
                "type": "array",
                "length": len(values),
                "items": {
                    "type": "object",
                    "keys": {
                        key: sorted(types) if len(types) > 1 else next(iter(types))
                        for key, types in sorted(key_types.items())
                    },
                },
            }

        item_types = sorted({EvolutionEngine._json_type_name(item) for item in inspected})
        if len(item_types) == 1 and isinstance(inspected[0], (list, dict)):
            item_shape = EvolutionEngine._json_shape(inspected[0], depth + 1)
        else:
            item_shape = item_types[0] if len(item_types) == 1 else item_types
        return {"type": "array", "length": len(values), "items": item_shape}

    @staticmethod
    def _json_type_name(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int) and not isinstance(value, bool):
            return "int"
        if isinstance(value, float):
            return "float"
        if isinstance(value, str):
            return "str"
        if isinstance(value, dict):
            return f"object[{len(value)}]"
        if isinstance(value, list):
            return f"array[{len(value)}]"
        return type(value).__name__

    @staticmethod
    def _current_generated_organs(current_genome: AgentGenome) -> str:
        compiled_dir = Path(__file__).resolve().parents[1] / "compiled_skills"
        sections = []
        for capability in current_genome.capabilities:
            skill_path = compiled_dir / f"{capability.name}.py"
            if not skill_path.exists():
                continue
            sections.append(f"## {capability.name}: {skill_path}")
            sections.append(EvolutionEngine._read_public_text(skill_path, limit=12000))
        return "\n".join(sections) if sections else "(no generated organs in current genome)"

    @staticmethod
    def _read_public_text(path: Path, limit: int = 8000) -> str:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"(unable to read public artifact: {exc})"

        if len(text) > limit:
            return text[:limit] + "\n... [truncated]"
        return text

    @staticmethod
    def _excerpt(text: str, limit: int = 8000) -> str:
        if not text:
            return "(no failed output captured)"
        if len(text) > limit:
            return text[:limit] + "\n... [truncated]"
        return text

    @staticmethod
    def _format_failure_history(failure_history: List[Any], limit: int = 8) -> str:
        """Render compact clinical-trial scars for the mutation prompt."""
        if not failure_history:
            return "(no phenotypic scars recorded yet)"

        recent = failure_history[-limit:]
        rows = [
            "| # | Phase | Organ | Error | Details |",
            "|---|---|---|---|---|",
        ]
        start_index = len(failure_history) - len(recent) + 1
        for offset, scar in enumerate(recent, start=start_index):
            phase = EvolutionEngine._scar_value(scar, "phase", "UNKNOWN")
            organ = EvolutionEngine._scar_value(scar, "proposed_organ_name", "unknown")
            error_type = EvolutionEngine._scar_value(scar, "error_type", "Unknown")
            details = EvolutionEngine._scar_value(scar, "details", "")
            rows.append(
                "| {index} | {phase} | {organ} | {error_type} | {details} |".format(
                    index=offset,
                    phase=EvolutionEngine._markdown_cell(phase),
                    organ=EvolutionEngine._markdown_cell(organ or "unknown"),
                    error_type=EvolutionEngine._markdown_cell(error_type),
                    details=EvolutionEngine._markdown_cell(details, limit=500),
                )
            )

        if len(failure_history) > limit:
            rows.append(
                f"\nShowing the most recent {limit} of {len(failure_history)} recorded scars."
            )
        return "\n".join(rows)

    @staticmethod
    def _format_clinical_trial_history(trial_history: List[Any], limit: int = 5) -> str:
        if not trial_history:
            return "(no clinical trials recorded yet)"

        recent = trial_history[-limit:]
        rows = [
            "| # | Organ | Success | Turns | Failure Tags | Critique | Output Excerpt | Turn Transcript |",
            "|---|---|---|---:|---|---|---|---|",
        ]
        start_index = len(trial_history) - len(recent) + 1
        for offset, trial in enumerate(recent, start=start_index):
            feedback = EvolutionEngine._trial_feedback(trial)
            rows.append(
                "| {index} | {organ} | {success} | {turns} | {tags} | {critique} | {output} | {transcript} |".format(
                    index=offset,
                    organ=EvolutionEngine._markdown_cell(
                        EvolutionEngine._trial_value(trial, "organ_name", "unknown") or "unknown"
                    ),
                    success=EvolutionEngine._markdown_cell(
                        str(EvolutionEngine._feedback_value(feedback, "success", False))
                    ),
                    turns=EvolutionEngine._markdown_cell(
                        EvolutionEngine._trial_value(trial, "turns_taken", "0")
                    ),
                    tags=EvolutionEngine._markdown_cell(
                        ", ".join(EvolutionEngine._feedback_value(feedback, "identified_gaps", []) or [])
                    ),
                    critique=EvolutionEngine._markdown_cell(
                        EvolutionEngine._feedback_value(feedback, "critique", ""),
                        limit=260,
                    ),
                    output=EvolutionEngine._markdown_cell(
                        EvolutionEngine._trial_value(trial, "output", ""),
                        limit=360,
                    ),
                    transcript=EvolutionEngine._markdown_cell(
                        EvolutionEngine._trial_value(trial, "turn_transcript", ""),
                        limit=520,
                    ),
                )
            )

        if len(trial_history) > limit:
            rows.append(
                f"\nShowing the most recent {limit} of {len(trial_history)} clinical trials."
            )
        return "\n".join(rows)

    @staticmethod
    def _trial_feedback(trial: Any) -> Any:
        if isinstance(trial, dict):
            return trial.get("feedback")
        return getattr(trial, "feedback", None)

    @staticmethod
    def _feedback_value(feedback: Any, field: str, default: Any) -> Any:
        if isinstance(feedback, dict):
            return feedback.get(field, default)
        return getattr(feedback, field, default)

    @staticmethod
    def _trial_value(trial: Any, field: str, default: str) -> str:
        if isinstance(trial, dict):
            value = trial.get(field, default)
        else:
            value = getattr(trial, field, default)
        return default if value is None else str(value)

    @staticmethod
    def _scar_value(scar: Any, field: str, default: str) -> str:
        if isinstance(scar, dict):
            value = scar.get(field, default)
        else:
            value = getattr(scar, field, default)
        return default if value is None else str(value)

    @staticmethod
    def _markdown_cell(text: str, limit: int = 160) -> str:
        text = " ".join(str(text).split())
        text = text.replace("|", "\\|")
        if len(text) > limit:
            return text[:limit] + " ... [truncated]"
        return text

    @staticmethod
    def apply_mutation(current_genome: AgentGenome, plan: TransformationPlan) -> AgentGenome:
        capability_map: Dict[str, CapabilityModel] = {cap.name: cap for cap in current_genome.capabilities}

        for name in plan.removed_capabilities:
            capability_map.pop(name, None)

        for cap in plan.added_capabilities:
            capability_map[cap.name] = cap

        new_capabilities = list(capability_map.values())
        new_constraints = [
            constraint
            for constraint in current_genome.constraints
            if constraint not in plan.removed_constraints
        ]
        for constraint in plan.added_constraints:
            if constraint not in new_constraints:
                new_constraints.append(constraint)

        return AgentGenome(
            version=current_genome.version + 1,
            persona_name=plan.new_persona_name or current_genome.persona_name,
            role_description=plan.new_role_description or current_genome.role_description,
            reasoning_protocol=plan.modified_protocol or current_genome.reasoning_protocol,
            capabilities=new_capabilities,
            constraints=new_constraints
        )
