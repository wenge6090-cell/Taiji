"""Create LLM providers from config."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vingobot.config.schema import Config
from vingobot.providers.base import GenerationSettings, LLMProvider
from vingobot.providers.registry import find_by_name


@dataclass(frozen=True)
class ProviderSnapshot:
    provider: LLMProvider
    model: str
    context_window_tokens: int
    signature: tuple[object, ...]


def make_provider(config: Config, model: str | None = None) -> LLMProvider:
    """Create the LLM provider implied by config.

    Args:
        config: Full configuration object.
        model: Optional model override.  Falls back to
            ``config.agents.defaults.model`` when not provided
            or ``None``.
    """
    model = model or config.agents.defaults.model
    provider_name = config.get_provider_name(model)
    p = config.get_provider(model)
    spec = find_by_name(provider_name) if provider_name else None
    backend = spec.backend if spec else "openai_compat"

    if backend == "azure_openai":
        if not p or not p.api_key or not p.api_base:
            raise ValueError("Azure OpenAI requires api_key and api_base in config.")
    elif backend == "openai_compat" and not model.startswith("bedrock/"):
        needs_key = not (p and p.api_key)
        exempt = spec and (spec.is_oauth or spec.is_local or spec.is_direct)
        if needs_key and not exempt:
            raise ValueError(f"No API key configured for provider '{provider_name}'.")

    if backend == "openai_codex":
        from vingobot.providers.openai_codex_provider import OpenAICodexProvider

        provider = OpenAICodexProvider(default_model=model)
    elif backend == "azure_openai":
        from vingobot.providers.azure_openai_provider import AzureOpenAIProvider

        provider = AzureOpenAIProvider(
            api_key=p.api_key,
            api_base=p.api_base,
            default_model=model,
        )
    elif backend == "github_copilot":
        from vingobot.providers.github_copilot_provider import GitHubCopilotProvider

        provider = GitHubCopilotProvider(default_model=model)
    elif backend == "anthropic":
        from vingobot.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(
            api_key=p.api_key if p else None,
            api_base=config.get_api_base(model),
            default_model=model,
            extra_headers=p.extra_headers if p else None,
        )
    elif backend == "bedrock":
        from vingobot.providers.bedrock_provider import BedrockProvider

        provider = BedrockProvider(
            api_key=p.api_key if p else None,
            api_base=p.api_base if p else None,
            default_model=model,
            region=getattr(p, "region", None) if p else None,
            profile=getattr(p, "profile", None) if p else None,
            extra_body=p.extra_body if p else None,
        )
    else:
        from vingobot.providers.openai_compat_provider import OpenAICompatProvider

        provider = OpenAICompatProvider(
            api_key=p.api_key if p else None,
            api_base=config.get_api_base(model),
            default_model=model,
            extra_headers=p.extra_headers if p else None,
            spec=spec,
            extra_body=p.extra_body if p else None,
        )

    defaults = config.agents.defaults
    provider.generation = GenerationSettings(
        temperature=defaults.temperature,
        max_tokens=defaults.max_tokens,
        reasoning_effort=defaults.reasoning_effort,
        disable_deepseek_thinking=defaults.disable_deepseek_thinking,
    )
    return provider


def provider_signature(config: Config) -> tuple[object, ...]:
    """Return the config fields that affect the primary LLM provider."""
    model = config.agents.defaults.model
    defaults = config.agents.defaults
    p = config.get_provider(model)
    return (
        model,
        defaults.provider,
        config.get_provider_name(model),
        config.get_api_key(model),
        config.get_api_base(model),
        p.extra_headers if p else None,
        p.extra_body if p else None,
        getattr(p, "region", None) if p else None,
        getattr(p, "profile", None) if p else None,
        defaults.max_tokens,
        defaults.temperature,
        defaults.reasoning_effort,
        defaults.context_window_tokens,
        defaults.disable_deepseek_thinking,
    )


def build_provider_snapshot(config: Config) -> ProviderSnapshot:
    return ProviderSnapshot(
        provider=make_provider(config),
        model=config.agents.defaults.model,
        context_window_tokens=config.agents.defaults.context_window_tokens,
        signature=provider_signature(config),
    )


def build_sixiang_provider_snapshot(config: Config, agent_name: str) -> ProviderSnapshot:
    """Create a provider snapshot for a specific sixiang agent.

    If the agent has a per-agent override in
    ``config.agents.defaults.sixiang.agents[agent_name]``, use that
    model and generation settings.  Otherwise fall back to the global
    defaults.

    Args:
        config: Full configuration.
        agent_name: One of ``mingjue``, ``weaver``, ``yang``, ``yin``, ``action``, ``anqu``.
    """
    agent_cfg = config.agents.defaults.sixiang.agents.get(agent_name)
    model = agent_cfg.model if (agent_cfg and agent_cfg.model) else config.agents.defaults.model
    provider = make_provider(config, model=model)

    if agent_cfg:
        # GenerationSettings is frozen — build a new instance if overrides exist
        temp = agent_cfg.temperature if agent_cfg.temperature is not None else provider.generation.temperature
        mt = agent_cfg.max_tokens if agent_cfg.max_tokens is not None else provider.generation.max_tokens
        re_effort = agent_cfg.reasoning_effort if agent_cfg.reasoning_effort is not None else provider.generation.reasoning_effort
        provider.generation = GenerationSettings(
            temperature=temp,
            max_tokens=mt,
            reasoning_effort=re_effort,
            disable_deepseek_thinking=provider.generation.disable_deepseek_thinking,
        )

    return ProviderSnapshot(
        provider=provider,
        model=model,
        context_window_tokens=config.agents.defaults.context_window_tokens,
        signature=(),
    )


def load_provider_snapshot(config_path: Path | None = None) -> ProviderSnapshot:
    from vingobot.config.loader import load_config, resolve_config_env_vars

    return build_provider_snapshot(resolve_config_env_vars(load_config(config_path)))
