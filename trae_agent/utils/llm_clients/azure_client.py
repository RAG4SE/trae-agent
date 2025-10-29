# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Azure client wrapper with tool integrations"""

import os
import openai

from trae_agent.utils.config import ModelConfig
from trae_agent.utils.llm_clients.openai_compatible_base import (
    OpenAICompatibleClient,
    ProviderConfig,
)


class AzureProvider(ProviderConfig):
    """Azure OpenAI provider configuration."""

    def create_client(
        self, api_key: str | None = None, base_url: str | None = None, api_version: str | None = None
    ) -> openai.AzureOpenAI:
        """Create Azure OpenAI client."""
        # Get Azure endpoint from environment variable (required)
        azure_endpoint = base_url or os.getenv('AZURE_OPENAI_ENDPOINT')
        if not azure_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is required for Azure OpenAI")

        # Get API key from environment variable (required)
        final_api_key = api_key or os.getenv('AZURE_OPENAI_API_KEY')
        if not final_api_key:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable is required for Azure OpenAI")

        # Get API version from environment variable or use default
        final_api_version = api_version or os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')

        return openai.AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_version=final_api_version,
            api_key=final_api_key,
        )

    def get_service_name(self) -> str:
        """Get the service name for retry logging."""
        return "Azure OpenAI"

    def get_provider_name(self) -> str:
        """Get the provider name for trajectory recording."""
        return "azure"

    def get_extra_headers(self) -> dict[str, str]:
        """Get Azure-specific headers (none needed)."""
        return {}

    def supports_tool_calling(self, model_name: str) -> bool:
        """Check if the model supports tool calling."""
        # Azure OpenAI models generally support tool calling
        return True


class AzureClient(OpenAICompatibleClient):
    """Azure client wrapper that maintains compatibility while using the new architecture."""

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config, AzureProvider())
