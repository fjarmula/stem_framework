from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.core.genome import TransformationPlan
from src.execution.tools import register_compiled_skill
from src.regulatory.validator import RegulatoryValidator


@dataclass
class ToolCompilationResult:
    """Outcome of compiling and registering a generated runtime organ."""
    success: bool
    critique: str
    tool_name: Optional[str] = None
    skill_path: Optional[Path] = None


class RuntimeToolCreator:
    """
    Persists generated organ source into src/compiled_skills and registers it
    on the active runtime belt.
    """

    def __init__(
        self,
        auditor: RegulatoryValidator,
        compiled_skills_dir: Optional[Path] = None,
    ):
        self.auditor = auditor
        self.compiled_skills_dir = (
            compiled_skills_dir
            or Path(__file__).resolve().parents[1] / "compiled_skills"
        )

    def compile_and_register(self, plan: TransformationPlan) -> ToolCompilationResult:
        """Compile one generated organ from a transformation plan."""
        if not plan.new_tool_implementation:
            return ToolCompilationResult(
                success=True,
                critique="No generated runtime organ was proposed.",
            )

        report = self.auditor.validate_generated_tool(plan)
        if report.verdict != "APPROVE":
            return ToolCompilationResult(
                success=False,
                critique=f"Generated organ rejected by immune system: {report.critique}",
            )

        tool_name = self.auditor.generated_tool_name(plan)
        if tool_name is None:
            return ToolCompilationResult(
                success=False,
                critique="Generated organ rejected: unable to resolve generated tool name.",
            )

        self.compiled_skills_dir.mkdir(parents=True, exist_ok=True)
        init_path = self.compiled_skills_dir / "__init__.py"
        init_path.touch(exist_ok=True)

        skill_path = self.compiled_skills_dir / f"{tool_name}.py"
        skill_path.write_text(plan.new_tool_implementation.rstrip() + "\n", encoding="utf-8")

        try:
            register_compiled_skill(tool_name, skill_path)
        except Exception as exc:
            return ToolCompilationResult(
                success=False,
                critique=f"Generated organ failed to register: {exc}",
                tool_name=tool_name,
                skill_path=skill_path,
            )

        return ToolCompilationResult(
            success=True,
            critique=f"Generated organ compiled and registered: {tool_name}",
            tool_name=tool_name,
            skill_path=skill_path,
        )
