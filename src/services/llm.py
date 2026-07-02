import json
import os
import asyncio
import uuid
from json import JSONDecodeError
from types import SimpleNamespace
from typing import Type, List, Optional, Any, Dict

import openai
from pydantic import BaseModel, ValidationError
from src.utils.config import config


class LLMRateLimitError(RuntimeError):
    """Raised when the configured model provider reports quota exhaustion."""


class LLMService:
    """
    Service layer for interacting with OpenAI-compatible chat APIs.

    The project can use OpenAI directly or providers that expose an
    OpenAI-compatible endpoint, such as Google's Gemini API.
    """

    DEFAULT_BASE_URLS: Dict[str, Optional[str]] = {
        "openai": None,
        "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    }
    DEFAULT_API_KEY_ENVS: Dict[str, str] = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }

    def __init__(
            self,
            api_key: Optional[str] = None,
            provider: Optional[str] = None,
            model: Optional[str] = None,
            base_url: Optional[str] = None,
            api_key_env: Optional[str] = None,
            structured_output_mode: Optional[str] = None,
    ):
        llm_config = config.get("llm", {})
        self.provider = (provider or llm_config.get("provider") or "openai").lower()
        self.model = model or llm_config["model"]
        self.api_key_env = api_key_env or llm_config.get(
            "api_key_env",
            self.DEFAULT_API_KEY_ENVS.get(self.provider, "OPENAI_API_KEY")
        )
        self.structured_output_mode = (
            structured_output_mode
            or llm_config.get("structured_output_mode")
            or ("native" if self.provider == "openai" else "json")
        ).lower()
        self.tool_call_mode = (
            llm_config.get("tool_call_mode")
            or ("native" if self.provider == "openai" else "manual")
        ).lower()

        resolved_api_key = api_key or os.getenv(self.api_key_env)
        if not resolved_api_key:
            raise ValueError(
                f"{self.api_key_env} is not set. Add it to your environment or .env file."
            )

        resolved_base_url = (
            base_url
            if base_url is not None
            else llm_config.get("base_url", self.DEFAULT_BASE_URLS.get(self.provider))
        )

        client_kwargs: Dict[str, Any] = {"api_key": resolved_api_key}
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url

        self.client = openai.AsyncOpenAI(**client_kwargs)

    @classmethod
    def from_config(cls) -> "LLMService":
        """Create an LLM service from config.yaml and the configured key env var."""
        return cls()

    async def get_structured_completion(self, system_prompt: str, user_prompt: str, response_model: Type[BaseModel]) -> BaseModel:
        """Request a structured completion and validate it into a Pydantic model."""
        if self.structured_output_mode == "native":
            return await self._get_native_structured_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=response_model,
            )

        return await self._get_json_structured_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=response_model,
        )

    async def _get_native_structured_completion(
            self,
            system_prompt: str,
            user_prompt: str,
            response_model: Type[BaseModel],
    ) -> BaseModel:
        """Use OpenAI's native Pydantic parsing helper."""
        try:
            response = await self.client.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=response_model,
            )
        except openai.RateLimitError as exc:
            raise LLMRateLimitError(self._format_rate_limit_error(exc)) from exc
        return response.choices[0].message.parsed

    async def _get_json_structured_completion(
            self,
            system_prompt: str,
            user_prompt: str,
            response_model: Type[BaseModel],
    ) -> BaseModel:
        """Use provider-portable JSON output and validate locally."""
        schema = response_model.model_json_schema()
        schema_text = json.dumps(schema, indent=2)
        json_prompt = self._structured_json_prompt(user_prompt, response_model, schema_text)

        messages = [
            {
                "role": "system",
                "content": (
                    f"{system_prompt}\n\n"
                    "You are being called by software. Your entire response must be valid JSON."
                ),
            },
            {"role": "user", "content": json_prompt}
        ]

        last_error: Optional[Exception] = None
        last_content = ""
        last_data: Any = None

        for attempt in range(2):
            response = await self._create_chat_completion(
                model=self.model,
                messages=messages,
            )
            last_content = self._message_content_to_text(response.choices[0].message.content)
            try:
                last_data = self._extract_json_object(last_content)
                return response_model.model_validate(last_data)
            except (ValidationError, ValueError, TypeError) as exc:
                last_error = exc
                if attempt == 1:
                    break
                messages.append({"role": "assistant", "content": last_content})
                messages.append({
                    "role": "user",
                    "content": self._structured_json_repair_prompt(
                        response_model=response_model,
                        schema_text=schema_text,
                        invalid_data=last_data,
                        validation_error=exc,
                    )
                })

        raise ValueError(
            f"Model did not return valid {response_model.__name__} JSON after repair. "
            f"Last error: {last_error}. Last response: {last_content}"
        )

    @staticmethod
    def _structured_json_prompt(
            user_prompt: str,
            response_model: Type[BaseModel],
            schema_text: str,
    ) -> str:
        return f"""
{user_prompt}

Return only one valid JSON object whose root value is a `{response_model.__name__}`.
The root object must contain the required top-level fields for `{response_model.__name__}`.
Do not return an inner nested schema or a tool parameter schema as the root object.

JSON Schema:
{schema_text}

Do not wrap the JSON in Markdown. Do not include commentary before or after it.
""".strip()

    @staticmethod
    def _structured_json_repair_prompt(
            response_model: Type[BaseModel],
            schema_text: str,
            invalid_data: Any,
            validation_error: Exception,
    ) -> str:
        invalid_json = json.dumps(invalid_data, indent=2, default=str)
        return f"""
Your previous JSON did not validate as `{response_model.__name__}`.

Validation error:
{validation_error}

Previous JSON:
{invalid_json}

Return a corrected root JSON object for `{response_model.__name__}` only.
Do not return a nested tool parameter schema as the root object.

JSON Schema:
{schema_text}
""".strip()

    async def get_chat_completion(self, messages: List[dict], tools: Optional[List[dict]] = None) -> Any:
        """Request a standard (non‑parsed) chat completion, optionally with tool definitions."""
        if tools and self.tool_call_mode == "manual":
            return await self._get_manual_tool_completion(messages=messages, tools=tools)

        kwargs = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        response = await self._create_chat_completion(**kwargs)
        return response.choices[0].message

    async def _get_manual_tool_completion(self, messages: List[dict], tools: List[dict]) -> Any:
        """Provider-portable tool calling using JSON instead of native function calls."""
        response_model = self._build_manual_tool_response(messages=messages, tools=tools)
        prompt_messages = self._with_manual_tool_protocol(messages=messages, tools=tools)
        response = await self._create_chat_completion(
            model=self.model,
            messages=prompt_messages,
        )
        content = self._message_content_to_text(response.choices[0].message.content)
        try:
            data = self._extract_json_object(content)
        except ValueError:
            return SimpleNamespace(content=content, tool_calls=None)

        if not isinstance(data, dict):
            return SimpleNamespace(content=str(data), tool_calls=None)

        tool_call = data.get("tool_call")
        if isinstance(tool_call, dict) and tool_call.get("name"):
            arguments = tool_call.get("arguments") or {}
            return SimpleNamespace(
                content=response_model,
                tool_calls=[
                    SimpleNamespace(
                        id=f"manual_{uuid.uuid4().hex}",
                        type="function",
                        function=SimpleNamespace(
                            name=tool_call["name"],
                            arguments=json.dumps(arguments),
                        ),
                    )
                ],
            )

        final = data.get("final")
        if final is None:
            final = content
        return SimpleNamespace(content=str(final), tool_calls=None)

    @staticmethod
    def _build_manual_tool_response(messages: List[dict], tools: List[dict]) -> str:
        return json.dumps({
            "tool_call": None,
            "final": None,
            "note": "manual tool protocol response"
        })

    @staticmethod
    def _with_manual_tool_protocol(messages: List[dict], tools: List[dict]) -> List[dict]:
        tool_specs = []
        for tool in tools:
            function = tool.get("function", {})
            tool_specs.append({
                "name": function.get("name"),
                "description": function.get("description"),
                "parameters": function.get("parameters"),
            })

        protocol = f"""
You have access to tools, but this provider requires a JSON tool protocol.

Available tools:
{json.dumps(tool_specs, indent=2)}

Respond with exactly one valid JSON object and no Markdown wrapper.

To call a tool:
{{"tool_call": {{"name": "tool_name", "arguments": {{...}}}}, "final": null}}

When you have the tool result and are ready to answer:
{{"tool_call": null, "final": "your complete final answer"}}
""".strip()

        converted: List[dict] = []
        protocol_attached = False
        for message in messages:
            role = message.get("role")
            if role == "system" and not protocol_attached:
                converted.append({
                    "role": "system",
                    "content": f"{message.get('content', '')}\n\n{protocol}",
                })
                protocol_attached = True
            elif role == "tool":
                converted.append({
                    "role": "user",
                    "content": (
                        f"Tool result from {message.get('name')} "
                        f"(tool_call_id={message.get('tool_call_id')}):\n"
                        f"{message.get('content', '')}"
                    ),
                })
            else:
                converted.append({
                    "role": role,
                    "content": message.get("content") or "",
                })

        if not protocol_attached:
            converted.insert(0, {"role": "system", "content": protocol})

        return converted

    async def _create_chat_completion(self, **kwargs: Any) -> Any:
        transient_errors = (
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
        )
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                return await self.client.chat.completions.create(**kwargs)
            except openai.RateLimitError as exc:
                raise LLMRateLimitError(self._format_rate_limit_error(exc)) from exc
            except transient_errors:
                if attempt == max_attempts - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

        raise RuntimeError("Unreachable chat completion retry state.")

    @staticmethod
    def _format_rate_limit_error(exc: openai.RateLimitError) -> str:
        message = str(exc)
        retry_after = "unknown"
        try:
            retry_after = str(exc.response.headers.get("retry-after", "unknown"))
        except Exception:
            pass
        return (
            "The configured LLM provider rate limit or quota was reached. "
            f"Retry-After: {retry_after}. Provider message: {message}"
        )

    @staticmethod
    def _message_content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    parts.append(str(item))
            return "".join(parts)
        if content is None:
            return ""
        return str(content)

    @staticmethod
    def _extract_json_object(text: str) -> Any:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            return json.loads(cleaned)
        except JSONDecodeError:
            pass

        decoder = json.JSONDecoder()
        for index, char in enumerate(cleaned):
            if char not in "[{":
                continue
            try:
                parsed, _ = decoder.raw_decode(cleaned[index:])
                return parsed
            except JSONDecodeError:
                continue

        raise ValueError(f"Model did not return valid JSON: {text}")
