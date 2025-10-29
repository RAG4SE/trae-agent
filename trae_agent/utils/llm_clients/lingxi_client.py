# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Lingxi client wrapper with tool integrations"""

import openai
import os

from trae_agent.utils.config import ModelConfig
from trae_agent.utils.llm_clients.openai_compatible_base import (
    OpenAICompatibleClient,
    ProviderConfig,
)


class LingxiProvider(ProviderConfig):
    """Lingxi provider configuration."""

    def create_client(
        self, api_key: str | None = None, base_url: str | None = None, api_version: str | None = None
    ) -> openai.OpenAI:
        """Create OpenAI client with Lingxi base URL."""
        # Lingxi typically uses a specific base URL
        default_base_url = "https://antchat.alipay.com/v1"
        final_base_url = base_url or default_base_url
        return openai.OpenAI(base_url=final_base_url, api_key=api_key or os.getenv('LINGXI_API_KEY'))

    def get_service_name(self) -> str:
        """Get the service name for retry logging."""
        return "Lingxi"

    def get_provider_name(self) -> str:
        """Get the provider name for trajectory recording."""
        return "lingxi"

    def get_extra_headers(self) -> dict[str, str]:
        """Get Lingxi-specific headers."""
        # Lingxi may require specific headers for authentication or other purposes
        return {}

    def supports_tool_calling(self, model_name: str) -> bool:
        """Check if the model supports tool calling."""
        # Lingxi models generally support tool calling
        # You may want to check specific model capabilities
        _ = model_name  # Avoid unused parameter warning
        return True


class LingxiClient(OpenAICompatibleClient):
    """Lingxi client wrapper that maintains compatibility while using the new architecture."""

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config, LingxiProvider())