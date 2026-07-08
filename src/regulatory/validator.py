import ast
import json
import re
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Literal, TYPE_CHECKING, Optional

from src.evaluation.stateful_benchmark import parse_episode_prompt

if TYPE_CHECKING:
    from src.core.genome import AgentGenome, TransformationPlan
else:
    from src.core.genome import AgentGenome, TransformationPlan
from src.execution.tools import TOOL_MAPPING
from src.services.llm import LLMService
from src.services.prompts import PromptManager


DISABLED_STATIC_BENCHMARK_TOOLS = {
    "trading_floor_solver",
    "security_sandbox_solver",
    "matrix_database_solver",
}


class ValidationReport(BaseModel):
    """The result of a safety and consistency check."""
    is_safe: bool
    consistency_score: int = Field(ge=0, le=100)
    identified_risks: List[str]
    verdict: Literal["APPROVE", "REJECT", "REQUIRE_FIXES"]
    critique: str


class RegulatoryValidator:
    """
    Acts as the 'Immune System'. Validates mutations before they are applied.
    """

    def __init__(self, llm: LLMService, prompt_manager: PromptManager):
        self.llm = llm
        self.prompt_manager = prompt_manager

    @staticmethod
    def generated_tool_name(plan: TransformationPlan) -> Optional[str]:
        if not plan.new_tool_implementation:
            return None
        if len(plan.added_capabilities) != 1:
            return None
        return plan.added_capabilities[0].name

    def validate_generated_tool(self, plan: TransformationPlan) -> ValidationReport:
        """Deterministically inspect generated organ source before compilation."""
        if not plan.new_tool_implementation:
            return ValidationReport(
                is_safe=True,
                consistency_score=100,
                identified_risks=[],
                verdict="APPROVE",
                critique="No generated runtime organ was proposed."
            )

        delta_report = self._validate_capability_delta(plan)
        if delta_report is not None:
            return delta_report

        tool_name = self.generated_tool_name(plan)
        if tool_name is None or not tool_name.isidentifier():
            return self._reject_generated_tool(
                "generated organ must have exactly one matching new capability with a valid Python identifier"
            )

        matching_capability = next(
            capability for capability in plan.added_capabilities if capability.name == tool_name
        )
        try:
            params = json.loads(matching_capability.parameters or "{}")
        except json.JSONDecodeError:
            return self._reject_generated_tool("generated organ capability parameters must be valid JSON")
        if not isinstance(params, dict):
            return self._reject_generated_tool("generated organ capability parameters must be a JSON object")
        properties = params.get("properties", {})
        required = params.get("required", [])
        observation_property = properties.get("observation") if isinstance(properties, dict) else None
        if not isinstance(observation_property, dict) or observation_property.get("type") != "string":
            return self._reject_generated_tool(
                "generated benchmark organ parameters must define string property 'observation'"
            )
        if not isinstance(required, list) or "observation" not in required:
            return self._reject_generated_tool("generated benchmark organ parameters must require 'observation'")
        if not any(context.startswith("domain_id:") for context in matching_capability.required_context):
            return self._reject_generated_tool("generated benchmark organ capability must include domain_id required_context")

        source = plan.new_tool_implementation
        try:
            tree = ast.parse(source)
            compile(source, f"<generated_skill:{tool_name}>", "exec")
        except SyntaxError as exc:
            return self._reject_generated_tool(f"generated organ has invalid Python syntax: {exc.msg}")

        for forbidden_literal in (
            "benchmarks/",
            "trade_001",
            "trade_002",
            "trade_003",
            "sec_001",
            "sec_002",
            "matrix_001",
            "matrix_002",
        ):
            if self._source_contains_runtime_literal(tree, forbidden_literal):
                return self._reject_generated_tool(
                    "generated benchmark organ must read public artifact paths from observations and trace files, "
                    f"not hard-code {forbidden_literal}"
                )
        if self._source_contains_placeholder_literal(tree):
            return self._reject_generated_tool(
                "generated organ must be complete executable logic, not placeholder or abbreviated source"
            )

        issues = self._inspect_generated_tool_ast(tree, tool_name)
        if issues:
            return ValidationReport(
                is_safe=False,
                consistency_score=0,
                identified_risks=issues,
                verdict="REJECT",
                critique=f"Generated organ rejected: {issues[0]}"
            )

        return ValidationReport(
            is_safe=True,
            consistency_score=90,
            identified_risks=["generated organ still requires runtime isolation during execution"],
            verdict="APPROVE",
            critique=f"Generated organ {tool_name} passed deterministic source inspection."
        )

    @staticmethod
    def _validate_capability_delta(plan: TransformationPlan) -> Optional[ValidationReport]:
        """Allow one organ addition and one organ deprecation in the same epoch."""
        if len(plan.added_capabilities) != 1:
            return RegulatoryValidator._reject_generated_tool(
                "generated organ mutation must add exactly one capability"
            )
        if len(plan.removed_capabilities) > 1:
            return RegulatoryValidator._reject_generated_tool(
                "generated organ mutation may remove at most one deprecated capability"
            )
        return None

    @staticmethod
    def _reject_generated_tool(reason: str) -> ValidationReport:
        if reason in {"call not allowed: open", "attribute call not allowed: open"}:
            reason = f"{reason}; use pathlib.Path(path).read_text(encoding='utf-8') for read-only artifact access"
        return ValidationReport(
            is_safe=False,
            consistency_score=0,
            identified_risks=[reason],
            verdict="REJECT",
            critique=f"Generated organ rejected: {reason}"
        )

    @staticmethod
    def _runtime_string_literals(tree: ast.AST) -> List[str]:
        """Return executable string literals, excluding comments and docstrings."""
        docstring_value_ids = set()
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if not isinstance(body, list) or not body:
                continue
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                docstring_value_ids.add(id(first.value))

        literals: List[str] = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstring_value_ids
            ):
                literals.append(node.value)
        return literals

    @classmethod
    def _source_contains_runtime_literal(cls, tree: ast.AST, literal: str) -> bool:
        return any(literal in value for value in cls._runtime_string_literals(tree))

    @classmethod
    def _source_contains_placeholder_literal(cls, tree: ast.AST) -> bool:
        placeholder_literals = (
            "todo",
            "not implemented",
            "placeholder",
            "abbreviated",
            "omitted",
            "stub",
        )
        for value in cls._runtime_string_literals(tree):
            lowered = value.lower()
            if any(placeholder in lowered for placeholder in placeholder_literals):
                return True

        placeholder_names = {"NotImplemented", "NotImplementedError"}
        return any(
            isinstance(node, ast.Name) and node.id in placeholder_names
            for node in ast.walk(tree)
        )

    @staticmethod
    def _inspect_generated_tool_ast(tree: ast.Module, tool_name: str) -> List[str]:
        allowed_imports = {
            "collections",
            "copy",
            "csv",
            "decimal",
            "itertools",
            "json",
            "importlib",
            "math",
            "pathlib",
            "re",
            "statistics",
            "typing",
        }
        banned_calls = {
            "__import__",
            "breakpoint",
            "compile",
            "eval",
            "exec",
            "globals",
            "input",
            "locals",
            "open",
            "vars",
        }
        banned_attributes = {
            "chmod",
            "hardlink_to",
            "mkdir",
            "open",
            "rename",
            "rmdir",
            "symlink_to",
            "system",
            "touch",
            "unlink",
            "write_bytes",
            "write_text",
        }
        issues: List[str] = []
        initialized_memory_keys = RegulatoryValidator._assigned_memory_keys(tree)

        public_functions = [
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
        ]
        public_classes = [
            node.name
            for node in tree.body
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_")
        ]

        run_function = next(
            (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run"),
            None
        )
        organ_class = next(
            (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == tool_name),
            None
        )

        if run_function is None and organ_class is None:
            issues.append(
                f"module must define run entrypoint or class {tool_name} with execute/run/__call__"
            )
        if run_function is not None and organ_class is not None:
            issues.append("module must expose only one runtime entrypoint: either run or organ class")
        if run_function is not None and public_functions != ["run"]:
            issues.append("module-level public functions must be exactly ['run']")
        if organ_class is not None:
            extra_classes = [name for name in public_classes if name != tool_name]
            if extra_classes:
                issues.append(f"module has unexpected public classes: {extra_classes}")
            method_names = {
                node.name
                for node in organ_class.body
                if isinstance(node, ast.FunctionDef)
            }
            if not method_names.intersection({"execute", "run", "__call__"}):
                issues.append(f"class {tool_name} must define execute(), run(), or __call__()")

        if run_function is None:
            pass
        else:
            if run_function.args.posonlyargs:
                issues.append("run entrypoint must not use positional-only parameters")
            if run_function.args.vararg is not None:
                issues.append("run entrypoint must not require *args")

        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
                continue
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and not any(
                    isinstance(child, ast.Call) for child in ast.walk(node)
            ):
                continue
            issues.append("module has top-level execution outside imports, constants, and function definitions")
            break

        if RegulatoryValidator._source_hardcodes_dynamic_probe_proof_key(tree):
            issues.append(
                "generated organ reads probe key metadata but hard-codes a proof input key; "
                "construct proof_object keys from observation_delta probe_input_key/probe_result_key "
                "or the runtime output_contract"
            )
        if RegulatoryValidator._source_hardcodes_dynamic_probe_row_read(tree):
            issues.append(
                "generated organ reads probe key metadata but hard-codes a probe row input lookup; "
                "read the selected row value with row.get(probe_input_key), not row.get('vector') "
                "or row.get('packet')"
            )

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    if root not in allowed_imports:
                        issues.append(f"import not allowed: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".", 1)[0]
                if node.level != 0 or root not in allowed_imports:
                    issues.append(f"import not allowed: {node.module}")
            elif isinstance(node, ast.Call):
                function = node.func
                if isinstance(function, ast.Name) and function.id == "next" and len(node.args) < 2:
                    issues.append(
                        "generated organ must use next(iterator, default) when parsing runtime data "
                        "so missing observations produce repairable diagnostics instead of StopIteration"
                    )
                elif isinstance(function, ast.Name) and function.id in banned_calls:
                    issue = f"call not allowed: {function.id}"
                    if function.id == "open":
                        issue += "; use pathlib.Path(path).read_text(encoding='utf-8') for read-only artifact access"
                    issues.append(issue)
                elif isinstance(function, ast.Attribute):
                    if function.attr in banned_attributes or function.attr.startswith("__"):
                        issue = f"attribute call not allowed: {function.attr}"
                        if function.attr == "open":
                            issue += "; use pathlib.Path(path).read_text(encoding='utf-8') for read-only artifact access"
                        issues.append(issue)
            elif isinstance(node, ast.Subscript):
                key = None
                if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                    key = node.slice.value
                if key == "loads":
                    issues.append(
                        "runtime observations do not expose artifact_manifest loads; "
                        "read already-loaded data from observation_delta"
                    )
                if RegulatoryValidator._is_unsafe_nested_memory_subscript(node, initialized_memory_keys):
                    issues.append(
                        "generated organ must not write through nested memory subscripts such as "
                        "memory['state']['field']; initialize nested dictionaries with setdefault/get "
                        "before assignment so empty memory payloads do not raise KeyError"
                    )
            elif isinstance(node, ast.Pass):
                issues.append("generated organ contains a pass statement instead of executable logic")
            elif isinstance(node, ast.Constant) and node.value is Ellipsis:
                issues.append("generated organ contains an ellipsis placeholder instead of executable logic")
            elif isinstance(node, ast.Name) and node.id == "__builtins__":
                issues.append("direct __builtins__ access is not allowed")

        return issues

    @staticmethod
    def _assigned_memory_keys(tree: ast.AST) -> set[str]:
        keys: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Subscript) or not isinstance(node.ctx, ast.Store):
                continue
            if not isinstance(node.value, ast.Name) or node.value.id not in {"mem", "memory"}:
                continue
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                keys.add(node.slice.value)
        return keys

    @staticmethod
    def _is_unsafe_nested_memory_subscript(node: ast.Subscript, initialized_keys: set[str]) -> bool:
        if not isinstance(node.ctx, ast.Store):
            return False
        parent = node.value
        if not isinstance(parent, ast.Subscript):
            return False
        root = parent.value
        if not isinstance(root, ast.Name) or root.id not in {"mem", "memory"}:
            return False
        if isinstance(parent.slice, ast.Constant) and isinstance(parent.slice.value, str):
            return parent.slice.value not in initialized_keys
        return True

    @classmethod
    def _source_hardcodes_dynamic_probe_proof_key(cls, tree: ast.AST) -> bool:
        literals = cls._runtime_string_literals(tree)
        if not {"probe_input_key", "selection_match"}.intersection(literals):
            return False

        hardcoded_input_keys = {"vector", "packet"}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = {
                key.value
                for key in node.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            if "observed_result" in keys and keys.intersection(hardcoded_input_keys):
                return True
        return False

    @classmethod
    def _source_hardcodes_dynamic_probe_row_read(cls, tree: ast.AST) -> bool:
        literals = cls._runtime_string_literals(tree)
        if not {"probe_input_key", "selection_match"}.intersection(literals):
            return False

        hardcoded_input_keys = {"vector", "packet"}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "get" or not node.args:
                continue
            first_arg = node.args[0]
            if (
                isinstance(first_arg, ast.Constant)
                and isinstance(first_arg.value, str)
                and first_arg.value in hardcoded_input_keys
            ):
                return True
        return False

    async def validate_transformation(
            self,
            current_genome: AgentGenome,
            plan: TransformationPlan,
            task_context: str = "",
    ) -> ValidationReport:
        generated_tool_report = self.validate_generated_tool(plan)
        if generated_tool_report.verdict != "APPROVE":
            return generated_tool_report

        task_literal_report = self.validate_against_task_literals(plan, task_context)
        if task_literal_report is not None:
            return task_literal_report

        disabled_static = [
            capability.name
            for capability in plan.added_capabilities
            if capability.name in DISABLED_STATIC_BENCHMARK_TOOLS
        ]
        if disabled_static:
            return ValidationReport(
                is_safe=False,
                consistency_score=0,
                identified_risks=["static_benchmark_shortcut"],
                verdict="REJECT",
                critique=(
                    "Pre-registered benchmark solvers are disabled for evolution; "
                    f"generate a new organ instead of adding {disabled_static}."
                )
            )

        generated_tool_name = self.generated_tool_name(plan)
        if plan.new_tool_implementation and generated_tool_name:
            return ValidationReport(
                is_safe=True,
                consistency_score=90,
                identified_risks=["generated organ behavior still requires environment verification"],
                verdict="APPROVE",
                critique=(
                    f"Approved generated runtime organ {generated_tool_name} after deterministic "
                    "schema, routing, and source inspection."
                )
            )

        added_existing_tools = [
            capability.name
            for capability in plan.added_capabilities
            if capability.name in TOOL_MAPPING
        ]

        available_tools = list(TOOL_MAPPING.keys())
        if generated_tool_name and generated_tool_name not in available_tools:
            available_tools.append(generated_tool_name)

        prompt = self.prompt_manager.get_prompt(
            "safety_validator.txt",
            current_genome=current_genome.model_dump_json(),
            plan=plan.model_dump_json(),
            available_tools=available_tools
        )

        return await self.llm.get_structured_completion(
            "You are a Senior AI Safety & Systems Auditor.",
            prompt,
            ValidationReport
        )

    def validate_against_task_literals(
        self,
        plan: TransformationPlan,
        task_context: str,
    ) -> Optional[ValidationReport]:
        if not plan.new_tool_implementation or not task_context:
            return None
        payload = parse_episode_prompt(task_context)
        if payload is None:
            return None

        tokens = self._public_artifact_tokens(payload)
        try:
            tree = ast.parse(plan.new_tool_implementation)
        except SyntaxError:
            return None

        runtime_literals = self._runtime_string_literals(tree)
        for token in sorted(tokens):
            if any(re.search(rf"\b{re.escape(token)}\b", literal) for literal in runtime_literals):
                return self._reject_generated_tool(
                    f"generated organ hard-codes public artifact token {token!r}; "
                    "parse identifiers and labels from observation_delta or public artifacts"
                )
        flat_fee = self._public_flat_fee(payload)
        if flat_fee is not None and self._source_hardcodes_fee_literal(tree, flat_fee):
            return self._reject_generated_tool(
                f"generated organ hard-codes public artifact fee {flat_fee:g}; "
                "parse flat fees dynamically from observation_delta rules_text"
            )
        return None

    @staticmethod
    def _public_artifact_tokens(payload: dict) -> set[str]:
        ignored = {
            "BUY",
            "SELL",
            "CSV",
            "JSON",
            "TRUE",
            "FALSE",
            "NONE",
            "NULL",
        }
        tokens: set[str] = set()
        artifacts = payload.get("public_artifacts", {})
        if not isinstance(artifacts, dict):
            return tokens

        for raw_path in artifacts.values():
            path = Path(str(raw_path))
            paths = [path]
            if path.is_dir():
                paths = [child for child in path.rglob("*") if child.is_file()]
            for artifact_path in paths:
                try:
                    text = artifact_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                for token in re.findall(r"\b[A-Z][A-Z0-9_]{1,}\b", text):
                    if token not in ignored:
                        tokens.add(token)
        return tokens

    @staticmethod
    def _public_flat_fee(payload: dict) -> Optional[float]:
        artifacts = payload.get("public_artifacts", {})
        if not isinstance(artifacts, dict):
            return None
        for label, raw_path in artifacts.items():
            if "rule" not in str(label).lower():
                continue
            try:
                text = Path(str(raw_path)).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            match = re.search(r"\bflat fee of\s+(\d+(?:\.\d+)?)\b", text, flags=re.IGNORECASE)
            if match:
                return float(match.group(1))
        return None

    @staticmethod
    def _source_hardcodes_fee_literal(tree: ast.AST, flat_fee: float) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(RegulatoryValidator._target_mentions_fee(target) for target in targets):
                    value = node.value
                    if RegulatoryValidator._numeric_literal_equals(value, flat_fee):
                        return True
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "get" and len(node.args) >= 2:
                    key_arg = node.args[0]
                    default_arg = node.args[1]
                    if (
                        isinstance(key_arg, ast.Constant)
                        and str(key_arg.value).lower() == "fee"
                        and RegulatoryValidator._numeric_literal_equals(default_arg, flat_fee)
                    ):
                        return True
        return False

    @staticmethod
    def _target_mentions_fee(target: ast.AST) -> bool:
        if isinstance(target, ast.Name):
            return "fee" in target.id.lower()
        if isinstance(target, ast.Attribute):
            return "fee" in target.attr.lower()
        if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant):
            return "fee" in str(target.slice.value).lower()
        return False

    @staticmethod
    def _numeric_literal_equals(node: ast.AST, expected: float) -> bool:
        return (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and float(node.value) == expected
        )
