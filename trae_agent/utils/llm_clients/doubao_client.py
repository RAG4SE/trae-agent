# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Doubao client wrapper with tool integrations"""

import json
import os

from volcenginesdkarkruntime import Ark

from trae_agent.tools.base import Tool, ToolCall, ToolResult
from trae_agent.utils.config import ModelConfig
from trae_agent.utils.llm_clients.base_client import BaseLLMClient
from trae_agent.utils.llm_clients.llm_basics import LLMMessage, LLMResponse, LLMUsage
from trae_agent.utils.llm_clients.retry_utils import retry_with


class DoubaoClient(BaseLLMClient):
    """Doubao client wrapper that uses Ark API directly."""

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)

        # Get API key from environment variable (support both names)
        api_key = os.getenv('DOUBAO_API_KEY') or os.getenv('ARK_API_KEY')
        if not api_key:
            raise ValueError("DOUBAO_API_KEY or ARK_API_KEY environment variable is required")

        # Use official Ark base URL
        base_url = "https://ark.cn-beijing.volces.com/api/v3"

        self.client = Ark(
            api_key=api_key,
            base_url=base_url,
        )
        self.message_history: list[dict] = []

    def set_chat_history(self, messages: list[LLMMessage]) -> None:
        """Set the chat history."""
        self.message_history = self.parse_messages(messages)

    @retry_with(max_retries=3, base_delay=1.0, max_delay=60.0)
    def _create_doubao_response(
        self,
        model_config: ModelConfig,
        tool_schemas: list[dict] | None,
    ) -> Ark.ChatCompletion:
        """Create a response using Doubao Ark API. This method will be decorated with retry logic."""
        completion_params = {
            "model": model_config.model,
            "messages": self.message_history,
            "temperature": model_config.temperature,
            "max_tokens": model_config.max_tokens,
            "top_p": model_config.top_p,
        }

        if tool_schemas:
            completion_params["tools"] = tool_schemas

        return self.client.chat.completions.create(**completion_params)

    def chat(
        self,
        messages: list[LLMMessage],
        model_config: ModelConfig,
        tools: list[Tool] | None = None,
        reuse_history: bool = True,
    ) -> LLMResponse:
        """Send chat messages to Doubao with optional tool support."""
        doubao_messages: list[dict] = self.parse_messages(messages)

        tool_schemas = None
        if tools:
            tool_schemas = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.get_input_schema(),
                    }
                }
                for tool in tools
            ]

        api_call_input: list[dict] = []
        if reuse_history:
            api_call_input.extend(self.message_history)
        api_call_input.extend(doubao_messages)

        self.message_history = api_call_input

        # Apply retry decorator to the API call
        retry_decorator = retry_with(
            func=self._create_doubao_response,
            provider_name="Doubao",
            max_retries=model_config.max_retries,
        )
        completion = retry_decorator(model_config, tool_schemas)

        content = completion.choices[0].message.content or ""
        tool_calls: list[ToolCall] = []

        if completion.choices[0].message.tool_calls:
            for tool_call in completion.choices[0].message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        call_id=tool_call.id,
                        name=tool_call.function.name,
                        arguments=json.loads(tool_call.function.arguments)
                        if tool_call.function.arguments
                        else {},
                        id=tool_call.id,
                    )
                )

        # Update message history
        if content:
            self.message_history.append({
                "role": "assistant",
                "content": content
            })

        if tool_calls:
            for tool_call in tool_calls:
                self.message_history.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tool_call.call_id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": json.dumps(tool_call.arguments)
                        }
                    }]
                })

        usage = None
        if completion.usage:
            usage = LLMUsage(
                input_tokens=completion.usage.prompt_tokens or 0,
                output_tokens=completion.usage.completion_tokens or 0,
                cache_read_input_tokens=0,  # Doubao may not provide this
                reasoning_tokens=0,  # Doubao may not provide this
            )

        llm_response = LLMResponse(
            content=content,
            usage=usage,
            model=completion.model,
            finish_reason=completion.choices[0].finish_reason,
            tool_calls=tool_calls if len(tool_calls) > 0 else None,
        )

        # Record trajectory if recorder is available
        if self.trajectory_recorder:
            self.trajectory_recorder.record_llm_interaction(
                messages=messages,
                response=llm_response,
                provider="doubao",
                model=model_config.model,
                tools=tools,
            )

        return llm_response

    def parse_messages(self, messages: list[LLMMessage]) -> list[dict]:
        """Parse the messages to Doubao format."""
        doubao_messages: list[dict] = []
        for msg in messages:
            if msg.tool_result:
                doubao_messages.append(self.parse_tool_call_result(msg.tool_result))
            elif msg.tool_call:
                doubao_messages.append(self.parse_tool_call(msg.tool_call))
            else:
                if not msg.content:
                    raise ValueError("Message content is required")
                if msg.role == "system":
                    doubao_messages.append({"role": "system", "content": msg.content})
                elif msg.role == "user":
                    doubao_messages.append({"role": "user", "content": msg.content})
                elif msg.role == "assistant":
                    doubao_messages.append({"role": "assistant", "content": msg.content})
                else:
                    raise ValueError(f"Invalid message role: {msg.role}")
        return doubao_messages

    def parse_tool_call(self, tool_call: ToolCall) -> dict:
        """Parse the tool call from the LLM response."""
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tool_call.call_id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(tool_call.arguments)
                }
            }]
        }

    def parse_tool_call_result(self, tool_call_result: ToolResult) -> dict:
        """Parse the tool call result from the LLM response."""
        result_content: str = ""
        if tool_call_result.result is not None:
            result_content += str(tool_call_result.result)
        if tool_call_result.error:
            result_content += f"\nError: {tool_call_result.error}"
        result_content = result_content.strip()

        return {
            "role": "tool",
            "content": result_content,
            "tool_call_id": tool_call_result.call_id,
        }