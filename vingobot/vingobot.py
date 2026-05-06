"""High-level programmatic interface to vingobot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vingobot.agent.hook import AgentHook
from vingobot.agent.loop import AgentLoop
from vingobot.bus.queue import MessageBus


@dataclass(slots=True)
class RunResult:
    """Result of a single agent run."""

    content: str
    tools_used: list[str]
    messages: list[dict[str, Any]]


class vingobot:
    """Programmatic facade for running the vingobot agent.

    Usage::

        bot = vingobot.from_config()
        result = await bot.run("Summarize this repo", hooks=[MyHook()])
        print(result.content)

    Sixiang (goal-driven) mode::

        bot = vingobot.from_config()
        await bot.start_sixiang(workers=3)
    """

    def __init__(self, loop: "AgentLoop") -> None:
        self._loop = loop
        self._sixiang_pool: Any = None

    @classmethod
    def from_config(
        cls,
        config_path: str | Path | None = None,
        *,
        workspace: str | Path | None = None,
    ) -> "vingobot":
        """Create a vingobot instance from a config file.

        Args:
            config_path: Path to ``config.json``.  Defaults to
                ``<project-root>/.vingobot/config.json``.
            workspace: Override the workspace directory from config.
        """
        from vingobot.config.loader import load_config, resolve_config_env_vars
        from vingobot.config.schema import Config

        resolved: Path | None = None
        if config_path is not None:
            resolved = Path(config_path).expanduser().resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Config not found: {resolved}")

        config: Config = resolve_config_env_vars(load_config(resolved))
        if workspace is not None:
            config.agents.defaults.workspace = str(Path(workspace).expanduser().resolve())

        provider = _make_provider(config)
        bus = MessageBus()
        defaults = config.agents.defaults

        loop = AgentLoop(
            bus=bus,
            provider=provider,
            workspace=config.workspace_path,
            model=defaults.model,
            max_iterations=defaults.max_tool_iterations,
            context_window_tokens=defaults.context_window_tokens,
            context_block_limit=defaults.context_block_limit,
            max_tool_result_chars=defaults.max_tool_result_chars,
            provider_retry_mode=defaults.provider_retry_mode,
            web_config=config.tools.web,
            exec_config=config.tools.exec,
            restrict_to_workspace=config.tools.restrict_to_workspace,
            mcp_servers=config.tools.mcp_servers,
            timezone=defaults.timezone,
            unified_session=defaults.unified_session,
            disabled_skills=defaults.disabled_skills,
            session_ttl_minutes=defaults.session_ttl_minutes,
            consolidation_ratio=defaults.consolidation_ratio,
            tools_config=config.tools,
        )

        # Store sixiang config
        self_cls = cls(loop)
        self_cls._config = config
        self_cls._sixiang_cfg = defaults.sixiang
        self_cls._provider = provider
        # Connect TPN tool to the vingobot instance for sixiang pool control
        loop._tpn_bot = self_cls
        return self_cls

    async def start_sixiang(self, *, workers: int | None = None) -> Any:
        """Start the sixiang (六爻) goal-driven coroutine pool.

        Args:
            workers: Number of concurrent workers.  Defaults to config value.

        Returns:
            The ``WorkerPool`` instance (call ``.stop()`` to shut down).
        """
        from vingobot.goal.coroutine import WorkerPool
        from vingobot.goal.sixiang_loop import execute_complete_sixiang_loop
        from vingobot.goal.dialogue_target import _inject_provider
        from vingobot.core.workspace import init_workspace
        from vingobot.config.paths import get_workspace_path as _get_ws

        # Init sixiang workspace
        ws_path = _get_ws()
        init_workspace(ws_path / ".taiji")

        n = (
            workers
            if workers is not None
            else getattr(getattr(self, "_sixiang_cfg", None), "max_concurrent_workers", 3)
        )

        provider = getattr(self, "_provider", None)
        sixiang_cfg = getattr(self, "_sixiang_cfg", None)
        has_per_agent = bool(sixiang_cfg and getattr(sixiang_cfg, "agents", None))

        if has_per_agent:
            # Per-agent model config exists — create independent providers
            from vingobot.goal.dialogue_target import _inject_sixiang_providers

            _inject_sixiang_providers(getattr(self, '_config', None))
        elif provider is not None:
            # No per-agent config — inject the single provider (backward compat)
            _inject_provider(provider)

        self._sixiang_pool = WorkerPool(
            max_workers=n,
            run_task_fn=execute_complete_sixiang_loop,
        )
        await self._sixiang_pool.start()
        return self._sixiang_pool

    async def stop_sixiang(self) -> None:
        """Stop the sixiang coroutine pool if running."""
        if self._sixiang_pool is not None:
            await self._sixiang_pool.stop()
            self._sixiang_pool = None

    @property
    def sixiang_running(self) -> bool:
        """Check if the sixiang pool is active."""
        return self._sixiang_pool is not None and self._sixiang_pool.running

    async def run(
        self,
        message: str,
        *,
        session_key: str = "sdk:default",
        hooks: list[AgentHook] | None = None,
    ) -> RunResult:
        """Run the agent once and return the result.

        Args:
            message: The user message to process.
            session_key: Session identifier for conversation isolation.
                Different keys get independent history.
            hooks: Optional lifecycle hooks for this run.
        """
        prev = self._loop._extra_hooks
        if hooks is not None:
            self._loop._extra_hooks = list(hooks)
        try:
            response = await self._loop.process_direct(
                message,
                session_key=session_key,
            )
        finally:
            self._loop._extra_hooks = prev

        content = (response.content if response else None) or ""
        return RunResult(content=content, tools_used=[], messages=[])


def _make_provider(config: Any) -> Any:
    """Create the LLM provider from config (extracted from CLI)."""
    from vingobot.providers.factory import make_provider

    return make_provider(config)
