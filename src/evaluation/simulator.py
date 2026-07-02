import re
from typing import List
from src.evaluation.feedback import EnvironmentFeedback
from src.evaluation.stateful_benchmark import verify_stateful_episode
from src.execution.tools import TOOL_MAPPING
from src.services.llm import LLMService
from src.services.prompts import PromptManager


class EnvironmentSimulator:
    """
    Class to simulate environmental interaction with the configured chat model.
    """

    def __init__(self, llm: LLMService, prompt_manager: PromptManager):
        self.llm = llm
        self.prompt_manager = prompt_manager

    @staticmethod
    def _extract_and_execute_code(text: str) -> str:
        """Helper method to extract code blocks from the agent's response and execute them."""
        code_blocks = re.findall(r"```python\s*\n(.*?)\n\s*```", text, re.DOTALL)
        if not code_blocks:
            return "No executable code found."
        full_code = "\n".join(code_blocks)

        # Use unified execution from registry
        if "python_interpreter" in TOOL_MAPPING:
            result = TOOL_MAPPING["python_interpreter"](full_code)
            # Check if result contains traceback or is an error
            if "Traceback (most recent call last):" in result or "File \"<string>\"" in result:
                return f"Execution Failed. Traceback:\n{result}"
            return f"Execution Success. Output:\n{result}"
        return "Error: python_interpreter not found in registry."

    async def evaluate(self, task: str, agent_output: str) -> EnvironmentFeedback:
        benchmark_feedback = verify_stateful_episode(task, agent_output)
        if benchmark_feedback is not None:
            return benchmark_feedback

        # physical verification
        execution_report = self._extract_and_execute_code(agent_output)

        # using LLM-as-a-judge to check the report and exclude "cheating solutions" (e.g. using 'if' 20 times to get proper answer)
        prompt = self.prompt_manager.get_prompt(
            "env_simulator.txt",
            task=task,
            agent_output=agent_output,
            execution_report=execution_report
        )

        return await self.llm.get_structured_completion(
            "You are the Objective Environment. You provide harsh but fair feedback.",
            prompt,
            EnvironmentFeedback
        )
