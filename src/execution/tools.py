from pathlib import Path
from typing import Dict, Callable
import contextlib
import importlib.util
import traceback
import io


def python_interpreter(code: str) -> str:
    """
    Executes Python code in a restricted local environment and returns the output
    :param code:
    :return:
    """

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        try:
            exec(code, {"__builtins__": __import__("builtins")}, {})
        except Exception:
            return traceback.format_exc()
    output = f.getvalue()
    if output.strip():
        return output
    else:
        return "Code executed successfully. NOTE: If you wanted to see a value, you must use 'print()'."


TOOL_MAPPING: Dict[str, Callable[..., str]] = {
    "python_interpreter": python_interpreter
}


def register_tool(name: str, function: Callable[..., str]) -> None:
    """Register a callable tool for the active runtime session."""
    if not name.isidentifier():
        raise ValueError(f"Tool name must be a valid Python identifier: {name}")
    TOOL_MAPPING[name] = function


def register_compiled_skill(name: str, module_path: Path) -> None:
    """Import a compiled skill module and expose its run(**kwargs) entrypoint."""
    if not name.isidentifier():
        raise ValueError(f"Compiled skill name must be a valid Python identifier: {name}")

    resolved_path = module_path.resolve()
    spec = importlib.util.spec_from_file_location(f"src.compiled_skills.{name}", resolved_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load compiled skill module: {resolved_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    run = getattr(module, "run", None)
    if not callable(run):
        raise ValueError(f"Compiled skill {name} must expose callable run(**kwargs)")

    def compiled_tool(**kwargs) -> str:
        return str(run(**kwargs))

    register_tool(name, compiled_tool)
