import io
import re
import contextlib
import traceback
from evaluation.validation import EnvironmentFeedback
import openai


class EnvironmentSimulator:
    """
    Class to simulate environmental interaction with OpenAI API.
    """

    def __init__(self, api_key: str):
        self.client = openai.AsyncOpenAI(api_key=api_key)

    def _extract_and_execute_code(self, text: str) -> str:
        """Helper method to extract code blocks from the agent's response and execute them."""
        code_blocks = re.findall(r"```python\n(.*?)\n```", text, re.DOTALL)
        if not code_blocks:
            return "No executable code found."
        full_code = "\n".join(code_blocks)
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            try:
                exec(full_code, {"__builtins__": __import__("builtins")}, {})
                execution_result = f.getvalue()
                return f"Execution Success. Output:\n{execution_result}" if execution_result else "Execution Success (No output)."
            except Exception:
                return f"Execution Failed. Traceback:\n{traceback.format_exc()}"

    async def evaluate(self, task: str, agent_output: str) -> EnvironmentFeedback:
        # physical verification
        execution_report = self._extract_and_execute_code(agent_output)

        # using LLM-as-a-judge to check the report and exclude "cheating solutions" (e.g. using 'if' 20 times to get proper answer)
        prompt = f"""
                TASK: {task}
                AGENT_OUTPUT: {agent_output}

                PHYSICAL EXECUTION REPORT:
                {execution_report}

                Evaluate the agent's performance.
                CRITICAL RULES:
                - If the task is mathematical/deterministic (e.g. Fibonacci, prime numbers) and the agent DID NOT use a tool or the code failed, mark as SUCCESS: FALSE.
                - If the agent's answer is correct but was 'guessed' without verification, mark as a FAILURE due to 'lack of deterministic verification'.

                Identify specific gaps like: 'python_interpreter', 'verification_logic', or 'syntax_error'.
                """

        response = await self.client.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are the Objective Environment. You provide harsh but fair feedback."},
                {"role": "user", "content": prompt}
            ],
            response_format=EnvironmentFeedback,
        )
        return response.choices[0].message.parsed
