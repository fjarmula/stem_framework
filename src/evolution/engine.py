import json
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
    ) -> TransformationPlan:
        """Analyzes a failure and proposes a mutation."""
        payload = parse_episode_prompt(task_context)

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
                "materialized_observation_delta": dict(sorted(materialized_keys.items())),
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
        return f"- text artifact with {len(text.splitlines())} line(s); content withheld until runtime"

    @staticmethod
    def _json_shape(value: Any) -> Any:
        if isinstance(value, dict):
            shape = {}
            for key, child in value.items():
                if isinstance(child, dict):
                    shape[key] = f"object[{len(child)}]"
                elif isinstance(child, list):
                    shape[key] = f"array[{len(child)}]"
                else:
                    shape[key] = type(child).__name__
            return shape
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
