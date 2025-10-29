# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Kimi client wrapper with tool integrations"""

import openai
import os

from trae_agent.utils.config import ModelConfig
from trae_agent.utils.llm_clients.openai_compatible_base import (
    OpenAICompatibleClient,
    ProviderConfig,
)


class KimiProvider(ProviderConfig):
    """Kimi provider configuration."""

    def create_client(
        self, api_key: str | None = None, base_url: str | None = None, api_version: str | None = None
    ) -> openai.OpenAI:
        """Create OpenAI client with Kimi base URL."""
        # Kimi typically uses a specific base URL
        default_base_url = "https://api.moonshot.cn/v1"
        final_base_url = base_url or default_base_url
        return openai.OpenAI(base_url=final_base_url, api_key=api_key or os.getenv('MOONSHOT_API_KEY') or os.getenv("KIMI_API_KEY"))

    def get_service_name(self) -> str:
        """Get the service name for retry logging."""
        return "Kimi"

    def get_provider_name(self) -> str:
        """Get the provider name for trajectory recording."""
        return "kimi"

    def get_extra_headers(self) -> dict[str, str]:
        """Get Kimi-specific headers."""
        # Kimi may require specific headers for authentication or other purposes
        return {}

    def supports_tool_calling(self, model_name: str) -> bool:
        """Check if the model supports tool calling."""
        # Kimi models generally support tool calling
        # You may want to check specific model capabilities
        _ = model_name  # Avoid unused parameter warning
        return True


class KimiClient(OpenAICompatibleClient):
    """Kimi client wrapper that maintains compatibility while using the new architecture."""

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config, KimiProvider())