import ast
import json
from pydantic import BaseModel, Field
from typing import List, Literal, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.core.genome import AgentGenome, TransformationPlan
else:
    from src.core.genome import AgentGenome, TransformationPlan
from src.execution.tools import TOOL_MAPPING
from src.services.llm import LLMService
from src.services.prompts import PromptManager


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
        generated_capabilities = [
            capability.name
            for capability in plan.added_capabilities
            if capability.name not in TOOL_MAPPING
        ]
        if len(generated_capabilities) != 1:
            return None
        return generated_capabilities[0]

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

        source = plan.new_tool_implementation
        try:
            tree = ast.parse(source)
            compile(source, f"<generated_skill:{tool_name}>", "exec")
        except SyntaxError as exc:
            return self._reject_generated_tool(f"generated organ has invalid Python syntax: {exc.msg}")

        issues = self._inspect_generated_tool_ast(tree)
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
    def _inspect_generated_tool_ast(tree: ast.Module) -> List[str]:
        allowed_imports = {
            "collections",
            "csv",
            "itertools",
            "json",
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
            "replace",
            "rmdir",
            "symlink_to",
            "system",
            "touch",
            "unlink",
            "write_bytes",
            "write_text",
        }
        issues: List[str] = []

        public_functions = [
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
        ]
        if public_functions != ["run"]:
            issues.append("module must expose exactly one public function named run")

        run_function = next(
            (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run"),
            None
        )
        if run_function is None:
            issues.append("module must define run entrypoint")
        else:
            if run_function.args.posonlyargs:
                issues.append("run entrypoint must not use positional-only parameters")
            if run_function.args.vararg is not None:
                issues.append("run entrypoint must not require *args")

        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
                continue
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and not any(
                    isinstance(child, ast.Call) for child in ast.walk(node)
            ):
                continue
            issues.append("module has top-level execution outside imports, constants, and function definitions")
            break

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
                if isinstance(function, ast.Name) and function.id in banned_calls:
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
            elif isinstance(node, ast.Name) and node.id == "__builtins__":
                issues.append("direct __builtins__ access is not allowed")

        return issues

    async def validate_transformation(
            self,
            current_genome: AgentGenome,
            plan: TransformationPlan
    ) -> ValidationReport:
        generated_tool_report = self.validate_generated_tool(plan)
        if generated_tool_report.verdict != "APPROVE":
            return generated_tool_report

        allowed_static_organs = {
            "trading_floor_solver",
            "security_sandbox_solver",
            "matrix_database_solver",
        }
        added_existing_tools = [
            capability.name
            for capability in plan.added_capabilities
            if capability.name in TOOL_MAPPING
        ]
        if (
            not plan.new_tool_implementation
            and added_existing_tools
            and set(added_existing_tools).issubset(allowed_static_organs)
            and len(added_existing_tools) == len(plan.added_capabilities)
            and not plan.removed_capabilities
            and not plan.removed_constraints
        ):
            return ValidationReport(
                is_safe=True,
                consistency_score=95,
                identified_risks=[],
                verdict="APPROVE",
                critique="Approved deterministic mutation that only enables pre-registered runtime tools."
            )

        available_tools = list(TOOL_MAPPING.keys())
        generated_tool_name = self.generated_tool_name(plan)
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
