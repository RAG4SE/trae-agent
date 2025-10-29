# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import os
from dataclasses import dataclass, field

import yaml


class ConfigError(Exception):
    pass


@dataclass
class ModelProvider:
    """
    Model provider configuration. For official model providers such as OpenAI and Anthropic,
    the base_url is optional. api_version is required for Azure.
    """

    api_key: str
    provider: str
    base_url: str | None = None
    api_version: str | None = None


@dataclass
class ModelConfig:
    """
    Model configuration.
    """

    model: str
    model_provider: str
    temperature: float
    top_p: float
    top_k: int
    parallel_tool_calls: bool
    max_retries: int
    max_tokens: int | None = None  # Legacy max_tokens parameter, optional
    supports_tool_calling: bool = True
    candidate_count: int | None = None  # Gemini specific field
    stop_sequences: list[str] | None = None
    max_completion_tokens: int | None = None  # Azure OpenAI specific field

    def get_max_tokens_param(self) -> int:
        """Get the maximum tokens parameter value.Prioritizes max_completion_tokens, falls back to max_tokens if not available."""
        if self.max_completion_tokens is not None:
            return self.max_completion_tokens
        elif self.max_tokens is not None:
            return self.max_tokens
        else:
            # Return default value if neither is set
            return 4096

    def should_use_max_completion_tokens(self) -> bool:
        """Determine whether to use the max_completion_tokens parameter.Primarily used for Azure OpenAI's newer models (e.g., gpt-5)."""
        return (
            self.max_completion_tokens is not None
            and self.model_provider == "azure"
            and ("gpt-5" in self.model or "o3" in self.model or "o4-mini" in self.model)
        )

@dataclass
class MCPServerConfig:
    # For stdio transport
    command: str | None = None
    args: list[str] | None = None
    env: dict[str, str] | None = None
    cwd: str | None = None

    # For sse transport
    url: str | None = None

    # For streamable http transport
    http_url: str | None = None
    headers: dict[str, str] | None = None

    # For websocket transport
    tcp: str | None = None

    # Common
    timeout: int | None = None
    trust: bool | None = None

    # Metadata
    description: str | None = None


@dataclass
class AgentConfig:
    """
    Base class for agent configurations.
    """

    allow_mcp_servers: list[str]
    mcp_servers_config: dict[str, MCPServerConfig]
    max_steps: int
    model: ModelConfig
    tools: list[str]


@dataclass
class TraeAgentConfig(AgentConfig):
    """
    Trae agent configuration.
    """

    enable_lakeview: bool = True
    tools: list[str] = field(
        default_factory=lambda: [
            "bash",
            "str_replace_based_edit_tool",
            "sequentialthinking",
            "task_done",
            "json_formatter"
        ]
    )



@dataclass
class LakeviewConfig:
    """
    Lakeview configuration.
    """

    model: ModelConfig


@dataclass
class Config:
    """
    Configuration class for agents, models and model providers.
    """

    lakeview: LakeviewConfig | None = None
    models: dict[str, ModelConfig] | None = None

    trae_agent: TraeAgentConfig | None = None

    @classmethod
    def create(
        cls,
        *,
        config_dict: dict | None = None,
        config_file: str | None = None,
        config_string: str | None = None,
    ) -> "Config":
        if config_file and config_string:
            raise ConfigError("Only one of config_file or config_string should be provided")

        # Parse YAML config from file or string
        try:
            if config_dict is not None:
                yaml_config = config_dict
            elif config_file is not None:
                if config_file.endswith(".json"):
                    return cls.create_from_legacy_config(config_file=config_file)
                with open(config_file, "r") as f:
                    yaml_config = yaml.safe_load(f)
            elif config_string is not None:
                yaml_config = yaml.safe_load(config_string)
            else:
                raise ConfigError("No config file or config string provided")
        except yaml.YAMLError as e:
            raise ConfigError(f"Error parsing YAML config: {e}") from e
    
        config = cls()

        # Parse models and populate model_provider fields
        models = yaml_config.get("models", None)
        if models is not None and len(models.keys()) > 0:
            config_models: dict[str, ModelConfig] = {}
            for model_name, model_config in models.items():
                config_models[model_name] = ModelConfig(**model_config)

                # config_models[model_name].model_provider = config_model_providers[
                #     model_config["model_provider"]
                # ]
            config.models = config_models
        else:
            raise ConfigError("No models provided")

        # Parse lakeview config
        lakeview = yaml_config.get("lakeview", None)
        if lakeview is not None:
            lakeview_model_name = lakeview.get("model", None)
            if lakeview_model_name is None:
                raise ConfigError("No model provided for lakeview")
            lakeview_model = config_models[lakeview_model_name]
            config.lakeview = LakeviewConfig(
                model=lakeview_model,
            )
        else:
            config.lakeview = None

        mcp_servers_config = {
            k: MCPServerConfig(**v) for k, v in yaml_config.get("mcp_servers", {}).items()
        }
        allow_mcp_servers = yaml_config.get("allow_mcp_servers", [])

        # Parse agents
        agents = yaml_config.get("agents", None)
        if agents is not None and len(agents.keys()) > 0:
            for agent_name, agent_config in agents.items():
                agent_model_name = agent_config.get("model", None)
                if agent_model_name is None:
                    raise ConfigError(f"No model provided for {agent_name}")
                try:
                    agent_model = config_models[agent_model_name]
                except KeyError as e:
                    raise ConfigError(f"Model {agent_model_name} not found") from e
                match agent_name:
                    case "trae_agent":
                        trae_agent_config = TraeAgentConfig(
                            **agent_config,
                            mcp_servers_config=mcp_servers_config,
                            allow_mcp_servers=allow_mcp_servers,
                        )
                        trae_agent_config.model = agent_model
                        if trae_agent_config.enable_lakeview and config.lakeview is None:
                            raise ConfigError("Lakeview is enabled but no lakeview config provided")
                        config.trae_agent = trae_agent_config
                    case _:
                        raise ConfigError(f"Unknown agent: {agent_name}")
        else:
            raise ConfigError("No agent configs provided")
        return config

def resolve_config_value(
    *,
    cli_value: int | str | float | None,
    config_value: int | str | float | None,
    env_var: str | None = None,
) -> int | str | float | None:
    """Resolve configuration value with priority: CLI > ENV > Config > Default."""
    if cli_value is not None:
        return cli_value

    if env_var and os.getenv(env_var):
        return os.getenv(env_var)

    if config_value is not None:
        return config_value

    return None
