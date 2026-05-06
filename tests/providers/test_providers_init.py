"""Tests for lazy provider exports from vingobot.providers."""

from __future__ import annotations

import importlib
import sys


def test_importing_providers_package_is_lazy(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "vingobot.providers", raising=False)
    monkeypatch.delitem(sys.modules, "vingobot.providers.anthropic_provider", raising=False)
    monkeypatch.delitem(sys.modules, "vingobot.providers.openai_compat_provider", raising=False)
    monkeypatch.delitem(sys.modules, "vingobot.providers.openai_codex_provider", raising=False)
    monkeypatch.delitem(sys.modules, "vingobot.providers.github_copilot_provider", raising=False)
    monkeypatch.delitem(sys.modules, "vingobot.providers.azure_openai_provider", raising=False)
    monkeypatch.delitem(sys.modules, "vingobot.providers.bedrock_provider", raising=False)

    providers = importlib.import_module("vingobot.providers")

    assert "vingobot.providers.anthropic_provider" not in sys.modules
    assert "vingobot.providers.openai_compat_provider" not in sys.modules
    assert "vingobot.providers.openai_codex_provider" not in sys.modules
    assert "vingobot.providers.github_copilot_provider" not in sys.modules
    assert "vingobot.providers.azure_openai_provider" not in sys.modules
    assert "vingobot.providers.bedrock_provider" not in sys.modules
    assert providers.__all__ == [
        "LLMProvider",
        "LLMResponse",
        "AnthropicProvider",
        "OpenAICompatProvider",
        "OpenAICodexProvider",
        "GitHubCopilotProvider",
        "AzureOpenAIProvider",
        "BedrockProvider",
    ]


def test_explicit_provider_import_still_works(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "vingobot.providers", raising=False)
    monkeypatch.delitem(sys.modules, "vingobot.providers.anthropic_provider", raising=False)

    namespace: dict[str, object] = {}
    exec("from vingobot.providers import AnthropicProvider", namespace)

    assert namespace["AnthropicProvider"].__name__ == "AnthropicProvider"
    assert "vingobot.providers.anthropic_provider" in sys.modules


def test_openai_codex_supports_progress_deltas() -> None:
    from vingobot.providers.openai_codex_provider import OpenAICodexProvider

    assert OpenAICodexProvider.supports_progress_deltas is True
