# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Lingxi client implementation."""

import openai
from openai.types.chat import ChatCompletion

from trae_agent.utils.llm_clients.base_client import BaseLLMClient
from trae_agent.utils.llm_clients.llm_basics import LLMMessage, LLMResponse
from trae_agent.utils.llm_clients.openai_compatible_base import (
    OpenAICompatibleBase,
    ProviderConfig,
)
from trae_agent.utils.llm_clients.retry_utils import retry_with


class LingxiConfig(ProviderConfig):
    """Lingxi-specific configuration."""

    def create_client(
        self, api_key: str, base_url: str | None, api_version: str | None
    ) -> openai.OpenAI:
        """Create the OpenAI client for Lingxi."""
        return openai.OpenAI(
            api_key=api_key,
            base_url=base_url or "https://antchat.alipay.com/v1",
        )

    def get_service_name(self) -> str:
        """Get the service name for retry logging."""
        return "lingxi"

    def get_provider_name(self) -> str:
        """Get the provider name."""
        return "lingxi"


class LingxiClient(OpenAICompatibleBase):
    """Lingxi LLM client implementation."""

    def __init__(self, model_config: ModelConfig):
        """Initialize the Lingxi client."""
        super().__init__(
            model_config=model_config,
            provider_config=LingxiConfig(),
        )

    @property
    def supports_structured_output(self) -> bool:
        """Check if the client supports structured output."""
        return True

    @property
    def supports_tools(self) -> bool:
        """Check if the client supports tool calling."""
        return True

    @property
    def supports_system_messages(self) -> bool:
        """Check if the client supports system messages."""
        return True

    @retry_with(max_retries=3, base_delay=1.0, max_delay=60.0)
    async def call(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        """Make an async call to the Lingxi API."""
        # Convert messages to OpenAI format
        openai_messages = self._convert_to_openai_messages(messages)

        # Prepare completion parameters
        completion_params = {
            "model": self.model_config.model,
            "messages": openai_messages,
            "temperature": kwargs.get("temperature", self.model_config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.model_config.max_tokens),
            "top_p": kwargs.get("top_p", self.model_config.top_p),
            "frequency_penalty": kwargs.get(
                "frequency_penalty", self.model_config.frequency_penalty
            ),
            "presence_penalty": kwargs.get(
                "presence_penalty", self.model_config.presence_penalty
            ),
        }

        # Add tools if provided
        tools = kwargs.get("tools")
        if tools:
            completion_params["tools"] = self._convert_tools_to_openai_format(tools)

        # Add tool choice if provided
        tool_choice = kwargs.get("tool_choice")
        if tool_choice:
            completion_params["tool_choice"] = tool_choice

        # Add stream parameter
        if "stream" in kwargs:
            completion_params["stream"] = kwargs["stream"]

        # Make the API call
        completion: ChatCompletion = await self.client.chat.completions.create(
            **completion_params
        )

        # Convert response back to our format
        return self._convert_from_openai_response(completion)

    @retry_with(max_retries=3, base_delay=1.0, max_delay=60.0)
    def call_sync(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        """Make a synchronous call to the Lingxi API."""
        # Convert messages to OpenAI format
        openai_messages = self._convert_to_openai_messages(messages)

        # Prepare completion parameters
        completion_params = {
            "model": self.model_config.model,
            "messages": openai_messages,
            "temperature": kwargs.get("temperature", self.model_config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.model_config.max_tokens),
            "top_p": kwargs.get("top_p", self.model_config.top_p),
            "frequency_penalty": kwargs.get(
                "frequency_penalty", self.model_config.frequency_penalty
            ),
            "presence_penalty": kwargs.get(
                "presence_penalty", self.model_config.presence_penalty
            ),
        }

        # Add tools if provided
        tools = kwargs.get("tools")
        if tools:
            completion_params["tools"] = self._convert_tools_to_openai_format(tools)

        # Add tool choice if provided
        tool_choice = kwargs.get("tool_choice")
        if tool_choice:
            completion_params["tool_choice"] = tool_choice

        # Add stream parameter
        if "stream" in kwargs:
            completion_params["stream"] = kwargs["stream"]

        # Make the API call
        completion: ChatCompletion = self.client.chat.completions.create(
            **completion_params
        )

        # Convert response back to our format
        return self._convert_from_openai_response(completion)
