import re
from typing import Any, Tuple

from src.evaluation.feedback import EnvironmentFeedback
from src.evaluation.stateful_benchmark import (
    parse_episode_prompt,
    run_stateful_episode,
    verify_stateful_episode,
)
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

    async def evaluate(self, task: str, agent_output: str, turns_taken: int | None = None) -> EnvironmentFeedback:
        benchmark_feedback = verify_stateful_episode(task, agent_output, turns_taken=turns_taken)
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

    async def evaluate_agent(self, agent: Any, task: str) -> Tuple[str, int, EnvironmentFeedback]:
        """
        Execute and evaluate a task.

        Stateful benchmark tasks are run through the physical episode runtime.
        Non-benchmark tasks retain the previous one-shot agent/evaluator path.
        """
        if parse_episode_prompt(task) is not None:
            episode_result = await run_stateful_episode(task, agent.execute_episode_turn)
            if episode_result is None:
                feedback = EnvironmentFeedback(
                    success=False,
                    critique="The benchmark task could not be parsed into an episode.",
                    identified_gaps=["unverifiable_inference"],
                )
                return "", 0, feedback
            return episode_result.output, episode_result.turns_taken, episode_result.feedback

        output, turns = await agent.execute_task(task)
        feedback = await self.evaluate(task, output, turns_taken=turns)
        return output, turns, feedback
