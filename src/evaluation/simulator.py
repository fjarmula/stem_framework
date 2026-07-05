from typing import Any, Tuple

from src.evaluation.feedback import EnvironmentFeedback
from src.evaluation.stateful_benchmark import (
    parse_episode_prompt,
    run_stateful_episode,
    verify_stateful_episode,
)
from src.services.llm import LLMService
from src.services.prompts import PromptManager


class EnvironmentSimulator:
    """
    Class to simulate environmental interaction with the configured chat model.
    """

    def __init__(self, llm: LLMService, prompt_manager: PromptManager):
        self.llm = llm
        self.prompt_manager = prompt_manager

    async def evaluate(self, task: str, agent_output: str, turns_taken: int | None = None) -> EnvironmentFeedback:
        benchmark_feedback = verify_stateful_episode(task, agent_output, turns_taken=turns_taken)
        if benchmark_feedback is not None:
            return benchmark_feedback

        return EnvironmentFeedback(
            success=False,
            critique=(
                "The MVP simulator only evaluates v2 stateful benchmark episodes "
                "through the physical runtime loop."
            ),
            identified_gaps=["unsupported_task_contract", "unverifiable_inference"],
        )

    async def evaluate_agent(self, agent: Any, task: str) -> Tuple[str, int, EnvironmentFeedback]:
        """
        Execute and evaluate a task.

        Stateful benchmark tasks are run through the physical episode runtime.
        Non-benchmark prompts are outside the MVP contract.
        """
        if parse_episode_prompt(task) is None:
            feedback = EnvironmentFeedback(
                success=False,
                critique="Task is not a v2 stateful benchmark episode.",
                identified_gaps=["unsupported_task_contract", "unverifiable_inference"],
            )
            return "", 0, feedback

        episode_result = await run_stateful_episode(task, agent.execute_episode_turn)
        if episode_result is None:
            feedback = EnvironmentFeedback(
                success=False,
                critique="The benchmark task could not be parsed into an episode.",
                identified_gaps=["unverifiable_inference"],
            )
            return "", 0, feedback
        return episode_result.output, episode_result.turns_taken, episode_result.feedback
