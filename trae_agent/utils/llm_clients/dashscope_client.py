# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""DashScope client wrapper with tool integrations"""

import openai

from trae_agent.utils.config import ModelConfig
from trae_agent.utils.llm_clients.openai_compatible_base import (
    OpenAICompatibleClient,
    ProviderConfig,
)


class DashScopeProvider(ProviderConfig):
    """DashScope provider configuration."""

    def create_client(
        self, api_key: str, base_url: str | None, api_version: str | None = None
    ) -> openai.OpenAI:
        """Create OpenAI client with DashScope base URL."""
        # DashScope typically uses a specific base URL
        default_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        final_base_url = base_url or default_base_url
        return openai.OpenAI(base_url=final_base_url, api_key=api_key)

    def get_service_name(self) -> str:
        """Get the service name for retry logging."""
        return "DashScope"

    def get_provider_name(self) -> str:
        """Get the provider name for trajectory recording."""
        return "dashscope"

    def get_extra_headers(self) -> dict[str, str]:
        """Get DashScope-specific headers."""
        # DashScope may require specific headers for authentication or other purposes
        return {}

    def supports_tool_calling(self, model_name: str) -> bool:
        """Check if the model supports tool calling."""
        # DashScope models like qwen-turbo, qwen-plus, qwen-max generally support tool calling
        # You may want to check specific model capabilities
        _ = model_name  # Avoid unused parameter warning
        return True


class DashScopeClient(OpenAICompatibleClient):
    """DashScope client wrapper that maintains compatibility while using the new architecture."""

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config, DashScopeProvider())