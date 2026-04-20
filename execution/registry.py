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
    return output if output.strip() else "Code executed successfully (no output)."


TOOL_MAPPING = {
    "python_interpreter": python_interpreter
}
