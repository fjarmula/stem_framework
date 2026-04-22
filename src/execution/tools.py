from typing import Dict, Callable
import contextlib
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


TOOL_MAPPING: Dict[str, Callable[[str], str]] = {
    "python_interpreter": python_interpreter
}
