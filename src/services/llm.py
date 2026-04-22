import openai
from typing import Type, List, Optional, Any
from pydantic import BaseModel
from src.config import config


class LLMService:
    """
    Service layer for interacting with OpenAI's language model APIs.

    Provides asynchronous methods to obtain structured (Pydantic‑parsed) responses
    and raw chat completions, abstracting the underlying client configuration.

    Attributes:
        client (openai.AsyncOpenAI): Asynchronous OpenAI client.
        model (str): Model identifier used for all requests (from config).

    Args:
        api_key (str): OpenAI API key for authentication.
    """

    def __init__(self, api_key: str):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = config["llm"]["model"]

    async def get_structured_completion(self, system_prompt: str, user_prompt: str, response_model: Type[BaseModel]) -> BaseModel:
        """Request a structured completion and parse it into a Pydantic model."""
        response = await self.client.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=response_model,
        )
        return response.choices[0].message.parsed

    async def get_chat_completion(self, messages: List[dict], tools: Optional[List[dict]] = None) -> Any:
        """Request a standard (non‑parsed) chat completion, optionally with tool definitions."""
        kwargs = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message
