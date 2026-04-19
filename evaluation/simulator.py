# evaluation/simulator.py
from evaluation.validation import EnvironmentFeedback
import openai


class EnvironmentSimulator:
    """
    Class to simulate environmental interaction with OpenAI API.
    """

    def __init__(self, api_key: str):
        self.client = openai.AsyncOpenAI(api_key=api_key)

    async def evaluate(self, task: str, agent_output: str) -> EnvironmentFeedback:
        # using a LLM-as-a-judge as for now to evaluate the agent's output against the task. In a more complex implementation, this could be replaced with a more sophisticated evaluation mechanism
        prompt = f"""
        Task: {task}
        Agent's Response: {agent_output}

        Evaluate whether the agent successfully completed the task.
        Return a JSON object with:
        - success: true/false
        - critique: detailed explanation of what worked or failed
        - identified_gaps: list of missing capabilities (e.g., "Inability to perform web search", "No memory of previous steps")
        """

        response = await self.client.chat.completions.parse(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format=EnvironmentFeedback,
        )
        return response.choices[0].message.parsed
