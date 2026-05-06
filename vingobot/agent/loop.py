"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import time
from contextlib import AsyncExitStack, nullcontext, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from vingobot.agent.autocompact import AutoCompact
from vingobot.agent.context import ContextBuilder
from vingobot.agent.hook import AgentHook, AgentHookContext, CompositeHook
from vingobot.agent.memory import Consolidator, Dream
from vingobot.agent.runner import _MAX_INJECTIONS_PER_TURN, AgentRunner, AgentRunSpec
from vingobot.agent.skills import _resolve_builtin_skills_dir
from vingobot.agent.subagent import SubagentManager
from vingobot.agent.tools.ask import (
    AskUserTool,
    ask_user_options_from_messages,
    ask_user_outbound,
    ask_user_tool_result_messages,
    pending_ask_user_id,
)
from vingobot.agent.tools.cron import CronTool
from vingobot.agent.tools.dmn import DmnTool
from vingobot.agent.tools.file_state import FileStateStore, bind_file_states, reset_file_states
from vingobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from vingobot.agent.tools.message import MessageTool
from vingobot.agent.tools.notebook import NotebookEditTool
from vingobot.agent.tools.registry import ToolRegistry
from vingobot.agent.tools.search import GlobTool, GrepTool
from vingobot.agent.tools.self import MyTool
from vingobot.agent.tools.shell import ExecTool
from vingobot.agent.tools.spawn import SpawnTool
from vingobot.agent.tools.tpn import TpnTool
from vingobot.agent.tools.web import WebFetchTool, WebSearchTool
from vingobot.bus.events import InboundMessage, OutboundMessage
from vingobot.bus.queue import MessageBus
from vingobot.command import CommandContext, CommandRouter, register_builtin_commands
from vingobot.config.schema import AgentDefaults
from vingobot.providers.base import LLMProvider
from vingobot.providers.factory import ProviderSnapshot
from vingobot.session.manager import Session, SessionManager
from vingobot.utils.document import extract_documents
from vingobot.utils.helpers import image_placeholder_text
from vingobot.utils.helpers import truncate_text as truncate_text_fn
from vingobot.utils.progress_events import (
    build_tool_event_finish_payloads,
    build_tool_event_start_payload,
    invoke_on_progress,
    on_progress_accepts_tool_events,
)
from vingobot.utils.runtime import EMPTY_FINAL_RESPONSE_MESSAGE

if TYPE_CHECKING:
    from vingobot.config.schema import ChannelsConfig, ExecToolConfig, ToolsConfig, WebToolsConfig
    from vingobot.cron.service import CronService


UNIFIED_SESSION_KEY = "unified:default"


class _LoopHook(AgentHook):
    """Core hook for the main loop."""

    def __init__(
        self,
        agent_loop: AgentLoop,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        *,
        channel: str = "cli",
        chat_id: str = "direct",
        message_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        session_key: str | None = None,
    ) -> None:
        super().__init__(reraise=True)
        self._loop = agent_loop
        self._on_progress = on_progress
        self._on_stream = on_stream
        self._on_stream_end = on_stream_end
        self._channel = channel
        self._chat_id = chat_id
        self._message_id = message_id
        self._metadata = metadata or {}
        self._session_key = session_key
        self._stream_buf = ""

    def wants_streaming(self) -> bool:
        return self._on_stream is not None

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        from vingobot.utils.helpers import strip_think

        prev_clean = strip_think(self._stream_buf)
        self._stream_buf += delta
        new_clean = strip_think(self._stream_buf)
        incremental = new_clean[len(prev_clean) :]
        if incremental and self._on_stream:
            await self._on_stream(incremental)

    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
        if self._on_stream_end:
            await self._on_stream_end(resuming=resuming)
        self._stream_buf = ""

    async def before_iteration(self, context: AgentHookContext) -> None:
        self._loop._current_iteration = context.iteration

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        if self._on_progress:
            if not self._on_stream and not context.streamed_content:
                thought = self._loop._strip_think(
                    context.response.content if context.response else None
                )
                if thought:
                    await self._on_progress(thought)
            tool_hint = self._loop._strip_think(self._loop._tool_hint(context.tool_calls))
            tool_events = [build_tool_event_start_payload(tc) for tc in context.tool_calls]
            await invoke_on_progress(
                self._on_progress,
                tool_hint,
                tool_hint=True,
                tool_events=tool_events,
            )
        for tc in context.tool_calls:
            args_str = json.dumps(tc.arguments, ensure_ascii=False)
            logger.info("Tool call: {}({})", tc.name, args_str[:200])
        self._loop._set_tool_context(
            self._channel,
            self._chat_id,
            self._message_id,
            self._metadata,
            session_key=self._session_key,
        )

    async def after_iteration(self, context: AgentHookContext) -> None:
        if (
            self._on_progress
            and context.tool_calls
            and context.tool_events
            and on_progress_accepts_tool_events(self._on_progress)
        ):
            tool_events = build_tool_event_finish_payloads(context)
            if tool_events:
                await invoke_on_progress(
                    self._on_progress,
                    "",
                    tool_hint=False,
                    tool_events=tool_events,
                )
        u = context.usage or {}
        logger.debug(
            "LLM usage: prompt={} completion={} cached={}",
            u.get("prompt_tokens", 0),
            u.get("completion_tokens", 0),
            u.get("cached_tokens", 0),
        )

    def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
        return self._loop._strip_think(content)


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _RUNTIME_CHECKPOINT_KEY = "runtime_checkpoint"
    _PENDING_USER_TURN_KEY = "pending_user_turn"

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int | None = None,
        context_window_tokens: int | None = None,
        context_block_limit: int | None = None,
        max_tool_result_chars: int | None = None,
        provider_retry_mode: str = "standard",
        web_config: WebToolsConfig | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
        timezone: str | None = None,
        session_ttl_minutes: int = 0,
        consolidation_ratio: float = 0.5,
        max_messages: int = 120,
        hooks: list[AgentHook] | None = None,
        unified_session: bool = False,
        disabled_skills: list[str] | None = None,
        tools_config: ToolsConfig | None = None,
        provider_snapshot_loader: Callable[[], ProviderSnapshot] | None = None,
        provider_signature: tuple[object, ...] | None = None,
    ):
        from vingobot.config.schema import ExecToolConfig, ToolsConfig, WebToolsConfig

        _tc = tools_config or ToolsConfig()
        defaults = AgentDefaults()
        self.bus = bus
        self.channels_config = channels_config
        self.provider = provider
        self._provider_snapshot_loader = provider_snapshot_loader
        self._provider_signature = provider_signature
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = (
            max_iterations if max_iterations is not None else defaults.max_tool_iterations
        )
        self.context_window_tokens = (
            context_window_tokens
            if context_window_tokens is not None
            else defaults.context_window_tokens
        )
        self.context_block_limit = context_block_limit
        self.max_tool_result_chars = (
            max_tool_result_chars
            if max_tool_result_chars is not None
            else defaults.max_tool_result_chars
        )
        self.provider_retry_mode = provider_retry_mode
        self.web_config = web_config or WebToolsConfig()
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self._start_time = time.time()
        self._last_usage: dict[str, int] = {}
        self._extra_hooks: list[AgentHook] = hooks or []

        self.context = ContextBuilder(workspace, timezone=timezone, disabled_skills=disabled_skills)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        # One file-read/write tracker per logical session. The tool registry is
        # shared by this loop, so tools resolve the active state via contextvars.
        self._file_state_store = FileStateStore()
        self.runner = AgentRunner(provider)
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            web_config=self.web_config,
            max_tool_result_chars=self.max_tool_result_chars,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
            disabled_skills=disabled_skills,
            max_iterations=self.max_iterations,
        )
        self._unified_session = unified_session
        self._max_messages = max_messages if max_messages > 0 else 120
        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stacks: dict[str, AsyncExitStack] = {}
        self._mcp_connected = False
        self._mcp_connecting = False
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._background_tasks: list[asyncio.Task] = []
        self._dmn_consumer_task: asyncio.Task | None = None
        # Track the user's last active channel for DMN progress reporting.
        self._last_user_route: tuple[str, str] = ("cli", "direct")  # (channel, chat_id)
        # DMN goal-state cache: periodically refreshed snapshot of all goals
        self._goal_states: dict[str, dict[str, object]] = {}
        self._goal_states_at: float = 0.0  # monotonic timestamp of last scan
        self._goal_states_lock = asyncio.Lock()
        self._session_locks: dict[str, asyncio.Lock] = {}
        # Per-session pending queues for mid-turn message injection.
        # When a session has an active task, new messages for that session
        # are routed here instead of creating a new task.
        self._pending_queues: dict[str, asyncio.Queue] = {}
        # VINGOBOT_MAX_CONCURRENT_REQUESTS: <=0 means unlimited; default 3.
        _max = int(os.environ.get("VINGOBOT_MAX_CONCURRENT_REQUESTS", "3"))
        self._concurrency_gate: asyncio.Semaphore | None = (
            asyncio.Semaphore(_max) if _max > 0 else None
        )
        # DMN gets its own gate so it never competes with dialogue for LLM slots.
        # Default 1: at most one DMN task runs at a time.
        _dmn_max = int(os.environ.get("VINGOBOT_DMN_MAX_CONCURRENT", "1"))
        self._dmn_gate: asyncio.Semaphore | None = (
            asyncio.Semaphore(_dmn_max) if _dmn_max > 0 else None
        )
        self.consolidator = Consolidator(
            store=self.context.memory,
            provider=provider,
            model=self.model,
            sessions=self.sessions,
            context_window_tokens=self.context_window_tokens,
            build_messages=self.context.build_messages,
            get_tool_definitions=self.tools.get_definitions,
            max_completion_tokens=provider.generation.max_tokens,
            consolidation_ratio=consolidation_ratio,
            session_locks=self._session_locks,
        )
        self.auto_compact = AutoCompact(
            sessions=self.sessions,
            consolidator=self.consolidator,
            session_ttl_minutes=session_ttl_minutes,
        )
        self.dream = Dream(
            store=self.context.memory,
            provider=provider,
            model=self.model,
        )
        self._register_default_tools()
        if _tc.my.enable:
            self.tools.register(MyTool(loop=self, modify_allowed=_tc.my.allow_set))
        if _tc.tpn.enable:
            self.tools.register(TpnTool(loop=self))
        if _tc.dmn.enable:
            self.tools.register(DmnTool(loop=self))
        self._runtime_vars: dict[str, Any] = {}
        self._current_iteration: int = 0
        self._tpn_bot: Any = None  # Set by vingobot for sixiang pool control
        self.commands = CommandRouter()
        register_builtin_commands(self.commands)

    def _sync_subagent_runtime_limits(self) -> None:
        """Keep subagent runtime limits aligned with mutable loop settings."""
        self.subagents.max_iterations = self.max_iterations

    def _apply_provider_snapshot(self, snapshot: ProviderSnapshot) -> None:
        """Swap model/provider for future turns without disturbing an active one."""
        provider = snapshot.provider
        model = snapshot.model
        context_window_tokens = snapshot.context_window_tokens
        if self.provider is provider and self.model == model:
            return
        old_model = self.model
        self.provider = provider
        self.model = model
        self.context_window_tokens = context_window_tokens
        self.runner.provider = provider
        self.subagents.set_provider(provider, model)
        self.consolidator.set_provider(provider, model, context_window_tokens)
        self.dream.set_provider(provider, model)
        self._provider_signature = snapshot.signature
        logger.info("Runtime model switched for next turn: {} -> {}", old_model, model)

    def _refresh_provider_snapshot(self) -> None:
        if self._provider_snapshot_loader is None:
            return
        try:
            snapshot = self._provider_snapshot_loader()
        except Exception:
            logger.exception("Failed to refresh provider config")
            return
        if snapshot.signature == self._provider_signature:
            return
        self._apply_provider_snapshot(snapshot)

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = (
            self.workspace if (self.restrict_to_workspace or self.exec_config.sandbox) else None
        )
        extra_read = [_resolve_builtin_skills_dir()] if allowed_dir else None
        self.tools.register(AskUserTool())
        self.tools.register(
            ReadFileTool(
                workspace=self.workspace,
                allowed_dir=allowed_dir,
                extra_allowed_dirs=extra_read,
            )
        )
        for cls in (WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        for cls in (GlobTool, GrepTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(NotebookEditTool(workspace=self.workspace, allowed_dir=allowed_dir))
        if self.exec_config.enable:
            self.tools.register(
                ExecTool(
                    working_dir=str(self.workspace),
                    timeout=self.exec_config.timeout,
                    restrict_to_workspace=self.restrict_to_workspace,
                    sandbox=self.exec_config.sandbox,
                    path_append=self.exec_config.path_append,
                    allowed_env_keys=self.exec_config.allowed_env_keys,
                )
            )
        if self.web_config.enable:
            self.tools.register(
                WebSearchTool(
                    config=self.web_config.search,
                    proxy=self.web_config.proxy,
                    user_agent=self.web_config.user_agent,
                )
            )
            self.tools.register(
                WebFetchTool(
                    config=self.web_config.fetch,
                    proxy=self.web_config.proxy,
                    user_agent=self.web_config.user_agent,
                )
            )
        self.tools.register(
            MessageTool(send_callback=self.bus.publish_outbound, workspace=self.workspace)
        )
        self.tools.register(SpawnTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(
                CronTool(self.cron_service, default_timezone=self.context.timezone or "UTC")
            )

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from vingobot.agent.tools.mcp import connect_mcp_servers

        try:
            self._mcp_stacks = await connect_mcp_servers(self._mcp_servers, self.tools)
            if self._mcp_stacks:
                self._mcp_connected = True
            else:
                logger.warning("No MCP servers connected successfully (will retry next message)")
        except asyncio.CancelledError:
            logger.warning("MCP connection cancelled (will retry next message)")
            self._mcp_stacks.clear()
        except BaseException as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            self._mcp_stacks.clear()
        finally:
            self._mcp_connecting = False

    def _set_tool_context(
        self,
        channel: str,
        chat_id: str,
        message_id: str | None = None,
        metadata: dict | None = None,
        session_key: str | None = None,
    ) -> None:
        """Update context for all tools that need routing info."""
        # When the caller threads a thread-scoped session_key (e.g. slack with
        # reply_in_thread: true), honor it so spawn announces route back to
        # the originating thread session. Falls back to unified mode or
        # channel:chat_id for callers that don't have a thread-scoped key.
        if session_key is not None:
            effective_key = session_key
        elif self._unified_session:
            effective_key = UNIFIED_SESSION_KEY
        else:
            effective_key = f"{channel}:{chat_id}"
        for name in ("message", "spawn", "cron", "my"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    if name == "spawn":
                        tool.set_context(channel, chat_id, effective_key=effective_key)
                        if hasattr(tool, "set_origin_message_id"):
                            tool.set_origin_message_id(message_id)
                    elif name == "cron":
                        tool.set_context(
                            channel, chat_id, metadata=metadata, session_key=session_key
                        )
                    elif name == "message":
                        tool.set_context(channel, chat_id, message_id, metadata=metadata)
                    else:
                        tool.set_context(channel, chat_id)

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        from vingobot.utils.helpers import strip_think

        return strip_think(text) or None

    @staticmethod
    def _runtime_chat_id(msg: InboundMessage) -> str:
        """Return the chat id shown in runtime metadata for the model."""
        return str(msg.metadata.get("context_chat_id") or msg.chat_id)

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hints with smart abbreviation."""
        from vingobot.utils.tool_hints import format_tool_hints

        return format_tool_hints(tool_calls)

    async def _dispatch_command_inline(
        self,
        msg: InboundMessage,
        key: str,
        raw: str,
        dispatch_fn: Callable[[CommandContext], Awaitable[OutboundMessage | None]],
    ) -> None:
        """Dispatch a command directly from the run() loop and publish the result."""
        ctx = CommandContext(msg=msg, session=None, key=key, raw=raw, loop=self)
        result = await dispatch_fn(ctx)
        if result:
            await self.bus.publish_outbound(result)
        else:
            logger.warning("Command '{}' matched but dispatch returned None", raw)

    async def _cancel_active_tasks(self, key: str) -> int:
        """Cancel and await all active tasks and subagents for *key*.

        Returns the total number of cancelled tasks + subagents.
        """
        tasks = self._active_tasks.pop(key, [])
        cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
        for t in tasks:
            with suppress(asyncio.CancelledError, Exception):
                await t
        sub_cancelled = await self.subagents.cancel_by_session(key)
        return cancelled + sub_cancelled

    def _effective_session_key(self, msg: InboundMessage) -> str:
        """Return the session key used for task routing and mid-turn injections."""
        if self._unified_session and not msg.session_key_override:
            return UNIFIED_SESSION_KEY
        return msg.session_key

    def _replay_token_budget(self) -> int:
        """Derive a token budget for session history replay from the context window."""
        if self.context_window_tokens <= 0:
            return 0
        max_output = getattr(getattr(self.provider, "generation", None), "max_tokens", 4096)
        try:
            reserved_output = int(max_output)
        except (TypeError, ValueError):
            reserved_output = 4096
        budget = self.context_window_tokens - max(1, reserved_output) - 1024
        return budget if budget > 0 else max(128, self.context_window_tokens // 2)

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        on_retry_wait: Callable[[str], Awaitable[None]] | None = None,
        *,
        session: Session | None = None,
        channel: str = "cli",
        chat_id: str = "direct",
        message_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        session_key: str | None = None,
        pending_queue: asyncio.Queue | None = None,
    ) -> tuple[str | None, list[str], list[dict], str, bool]:
        """Run the agent iteration loop.

        *on_stream*: called with each content delta during streaming.
        *on_stream_end(resuming)*: called when a streaming session finishes.
        ``resuming=True`` means tool calls follow (spinner should restart);
        ``resuming=False`` means this is the final response.

        Returns (final_content, tools_used, messages, stop_reason, had_injections).
        """
        self._sync_subagent_runtime_limits()

        loop_hook = _LoopHook(
            self,
            on_progress=on_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
            channel=channel,
            chat_id=chat_id,
            message_id=message_id,
            metadata=metadata,
            session_key=session_key,
        )
        hook: AgentHook = (
            CompositeHook([loop_hook] + self._extra_hooks) if self._extra_hooks else loop_hook
        )

        async def _checkpoint(payload: dict[str, Any]) -> None:
            if session is None:
                return
            self._set_runtime_checkpoint(session, payload)

        async def _drain_pending(*, limit: int = _MAX_INJECTIONS_PER_TURN) -> list[dict[str, Any]]:
            """Drain follow-up messages from the pending queue.

            When no messages are immediately available but sub-agents
            spawned in this dispatch are still running, blocks until at
            least one result arrives (or timeout).  This keeps the runner
            loop alive so subsequent sub-agent completions are consumed
            in-order rather than dispatched separately.
            """
            if pending_queue is None:
                return []

            def _to_user_message(pending_msg: InboundMessage) -> dict[str, Any]:
                content = pending_msg.content
                media = pending_msg.media if pending_msg.media else None
                if media:
                    content, media = extract_documents(content, media)
                    media = media or None
                user_content = self.context._build_user_content(content, media)
                runtime_ctx = self.context._build_runtime_context(
                    pending_msg.channel,
                    self._runtime_chat_id(pending_msg),
                    self.context.timezone,
                )
                if isinstance(user_content, str):
                    merged: str | list[dict[str, Any]] = f"{runtime_ctx}\n\n{user_content}"
                else:
                    merged = [{"type": "text", "text": runtime_ctx}] + user_content
                return {"role": "user", "content": merged}

            items: list[dict[str, Any]] = []
            while len(items) < limit:
                try:
                    items.append(_to_user_message(pending_queue.get_nowait()))
                except asyncio.QueueEmpty:
                    break

            # Block if nothing drained but sub-agents spawned in this dispatch
            # are still running.  Keeps the runner loop alive so subsequent
            # completions are injected in-order rather than dispatched separately.
            if (
                not items
                and session is not None
                and self.subagents.get_running_count_by_session(session.key) > 0
            ):
                try:
                    msg = await asyncio.wait_for(pending_queue.get(), timeout=300)
                except asyncio.TimeoutError:
                    logger.warning(
                        "Timeout waiting for sub-agent completion in session {}",
                        session.key,
                    )
                    return items
                items.append(_to_user_message(msg))
                while len(items) < limit:
                    try:
                        items.append(_to_user_message(pending_queue.get_nowait()))
                    except asyncio.QueueEmpty:
                        break

            return items

        active_session_key = session.key if session else session_key
        file_state_token = bind_file_states(self._file_state_store.for_session(active_session_key))
        try:
            result = await self.runner.run(
                AgentRunSpec(
                    initial_messages=initial_messages,
                    tools=self.tools,
                    model=self.model,
                    max_iterations=self.max_iterations,
                    max_tool_result_chars=self.max_tool_result_chars,
                    hook=hook,
                    error_message="Sorry, I encountered an error calling the AI model.",
                    concurrent_tools=True,
                    workspace=self.workspace,
                    session_key=session.key if session else None,
                    context_window_tokens=self.context_window_tokens,
                    context_block_limit=self.context_block_limit,
                    provider_retry_mode=self.provider_retry_mode,
                    progress_callback=on_progress,
                    retry_wait_callback=on_retry_wait,
                    checkpoint_callback=_checkpoint,
                    injection_callback=_drain_pending,
                )
            )
        finally:
            reset_file_states(file_state_token)
        self._last_usage = result.usage
        if result.stop_reason == "max_iterations":
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            # Push final content through stream so streaming channels (e.g. Feishu)
            # update the card instead of leaving it empty.
            if on_stream and on_stream_end:
                await on_stream(result.final_content or "")
                await on_stream_end(resuming=False)
        elif result.stop_reason == "error":
            logger.error("LLM returned error: {}", (result.final_content or "")[:200])
        return (
            result.final_content,
            result.tools_used,
            result.messages,
            result.stop_reason,
            result.had_injections,
        )

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        # ── Start DMN consumer: polls pending/ for cognitive evolution tasks ──
        self._dmn_consumer_task = asyncio.create_task(
            self._run_dmn_consumer(),
            name="dmn-consumer",
        )

        try:
            while self._running:
                try:
                    msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
                except asyncio.TimeoutError:
                    self.auto_compact.check_expired(
                        self._schedule_background,
                        active_session_keys=self._pending_queues.keys(),
                    )
                    continue
                except asyncio.CancelledError:
                    # Preserve real task cancellation so shutdown can complete cleanly.
                    # Only ignore non-task CancelledError signals that may leak from integrations.
                    if not self._running or asyncio.current_task().cancelling():
                        raise
                    continue
                except Exception as e:
                    logger.warning("Error consuming inbound message: {}, continuing...", e)
                    continue

                raw = msg.content.strip()
                if self.commands.is_priority(raw):
                    await self._dispatch_command_inline(
                        msg,
                        msg.session_key,
                        raw,
                        self.commands.dispatch_priority,
                    )
                    continue
                effective_key = self._effective_session_key(msg)
                # If this session already has an active pending queue (i.e. a task
                # is processing this session), route the message there for mid-turn
                # injection instead of creating a competing task.
                if effective_key in self._pending_queues:
                    # Non-priority commands must not be queued for injection;
                    # dispatch them directly (same pattern as priority commands).
                    if self.commands.is_dispatchable_command(raw):
                        await self._dispatch_command_inline(
                            msg,
                            effective_key,
                            raw,
                            self.commands.dispatch,
                        )
                        continue
                    pending_msg = msg
                    if effective_key != msg.session_key:
                        pending_msg = dataclasses.replace(
                            msg,
                            session_key_override=effective_key,
                        )
                    try:
                        self._pending_queues[effective_key].put_nowait(pending_msg)
                    except asyncio.QueueFull:
                        logger.warning(
                            "Pending queue full for session {}, falling back to queued task",
                            effective_key,
                        )
                    else:
                        logger.info(
                            "Routed follow-up message to pending queue for session {}",
                            effective_key,
                        )
                        continue
                # Compute the effective session key before dispatching
                # This ensures /stop command can find tasks correctly when unified session is enabled
                task = asyncio.create_task(self._dispatch(msg))
                self._active_tasks.setdefault(effective_key, []).append(task)
                task.add_done_callback(
                    lambda t, k=effective_key: (
                        self._active_tasks.get(k, []) and self._active_tasks[k].remove(t)
                        if t in self._active_tasks.get(k, [])
                        else None
                    )
                )
        finally:
            if self._dmn_consumer_task is not None and not self._dmn_consumer_task.done():
                self._dmn_consumer_task.cancel()
                try:
                    await asyncio.wait_for(self._dmn_consumer_task, timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                logger.info("DMN consumer task cancelled")

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message: per-session serial, cross-session concurrent."""
        session_key = self._effective_session_key(msg)
        if session_key != msg.session_key:
            msg = dataclasses.replace(msg, session_key_override=session_key)
        lock = self._session_locks.setdefault(session_key, asyncio.Lock())
        gate = self._concurrency_gate or nullcontext()

        # Remember the user's route for DMN progress reporting (skip DMN itself).
        if session_key != "_dmn":
            self._last_user_route = (msg.channel, msg.chat_id)

        # Register a pending queue so follow-up messages for this session are
        # routed here (mid-turn injection) instead of spawning a new task.
        pending = asyncio.Queue(maxsize=20)
        self._pending_queues[session_key] = pending

        try:
            async with lock, gate:
                try:
                    on_stream = on_stream_end = None
                    if msg.metadata.get("_wants_stream"):
                        # Split one answer into distinct stream segments.
                        stream_base_id = f"{msg.session_key}:{time.time_ns()}"
                        stream_segment = 0

                        def _current_stream_id() -> str:
                            return f"{stream_base_id}:{stream_segment}"

                        async def on_stream(delta: str) -> None:
                            meta = dict(msg.metadata or {})
                            meta["_stream_delta"] = True
                            meta["_stream_id"] = _current_stream_id()
                            await self.bus.publish_outbound(
                                OutboundMessage(
                                    channel=msg.channel,
                                    chat_id=msg.chat_id,
                                    content=delta,
                                    metadata=meta,
                                )
                            )

                        async def on_stream_end(*, resuming: bool = False) -> None:
                            nonlocal stream_segment
                            meta = dict(msg.metadata or {})
                            meta["_stream_end"] = True
                            meta["_resuming"] = resuming
                            meta["_stream_id"] = _current_stream_id()
                            await self.bus.publish_outbound(
                                OutboundMessage(
                                    channel=msg.channel,
                                    chat_id=msg.chat_id,
                                    content="",
                                    metadata=meta,
                                )
                            )
                            stream_segment += 1

                    response = await self._process_message(
                        msg,
                        on_stream=on_stream,
                        on_stream_end=on_stream_end,
                        pending_queue=pending,
                        _already_locked=True,
                    )
                    if response is not None:
                        await self.bus.publish_outbound(response)
                    elif msg.channel == "cli":
                        await self.bus.publish_outbound(
                            OutboundMessage(
                                channel=msg.channel,
                                chat_id=msg.chat_id,
                                content="",
                                metadata=msg.metadata or {},
                            )
                        )
                except asyncio.CancelledError:
                    logger.info("Task cancelled for session {}", session_key)
                    # Preserve partial context from the interrupted turn so
                    # the user does not lose tool results and assistant
                    # messages accumulated before /stop.  The checkpoint was
                    # already persisted to session metadata by
                    # _emit_checkpoint during tool execution; materializing
                    # it into session history now makes it visible in the
                    # next conversation turn.
                    try:
                        key = self._effective_session_key(msg)
                        session = self.sessions.get_or_create(key)
                        if self._restore_runtime_checkpoint(session):
                            self._clear_pending_user_turn(session)
                            self.sessions.save(session)
                            logger.info(
                                "Restored partial context for cancelled session {}",
                                key,
                            )
                    except Exception:
                        logger.debug(
                            "Could not restore checkpoint for cancelled session {}",
                            session_key,
                            exc_info=True,
                        )
                    raise
                except Exception:
                    logger.exception("Error processing message for session {}", session_key)
                    await self.bus.publish_outbound(
                        OutboundMessage(
                            channel=msg.channel,
                            chat_id=msg.chat_id,
                            content="Sorry, I encountered an error.",
                        )
                    )
        finally:
            # Drain any messages still in the pending queue and re-publish
            # them to the bus so they are processed as fresh inbound messages
            # rather than silently lost.
            queue = self._pending_queues.pop(session_key, None)
            if queue is not None:
                leftover = 0
                while True:
                    try:
                        item = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    await self.bus.publish_inbound(item)
                    leftover += 1
                if leftover:
                    logger.info(
                        "Re-published {} leftover message(s) to bus for session {}",
                        leftover,
                        session_key,
                    )

    async def close_mcp(self) -> None:
        """Drain pending background archives, then close MCP connections."""
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
        for name, stack in self._mcp_stacks.items():
            try:
                await stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                logger.debug("MCP server '{}' cleanup error (can be ignored)", name)
        self._mcp_stacks.clear()

    def _schedule_background(self, coro) -> None:
        """Schedule a coroutine as a tracked background task (drained on shutdown)."""
        task = asyncio.create_task(coro)
        self._background_tasks.append(task)
        task.add_done_callback(self._background_tasks.remove)

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        if self._dmn_consumer_task is not None and not self._dmn_consumer_task.done():
            self._dmn_consumer_task.cancel()
        logger.info("Agent loop stopping")

    # ------------------------------------------------------------------
    # DMN — goal-state scanner (read-only snapshot for context injection)
    # ------------------------------------------------------------------

    async def _scan_goal_states(self) -> None:
        """Read every goal's ``meta.json`` and cache a lightweight summary.

        This is a pure read operation — it never writes to the file system
        and never triggers side effects.  The cached summary is later
        injected into the runtime context so the LLM is aware of TPN state.
        """
        from datetime import datetime, timezone

        from vingobot.core.goal_meta import get_all_goals

        try:
            all_goals = get_all_goals()
        except Exception:
            logger.exception("[DMN] 扫描目标状态失败")
            return

        now = datetime.now(timezone.utc)
        snapshot: dict[str, dict[str, object]] = {}

        for meta in all_goals:
            # Compute relative "last active" for human readability
            try:
                last_dt = datetime.fromisoformat(meta.last_active) if meta.last_active else None
                if last_dt:
                    delta = now - last_dt
                    if delta.total_seconds() < 60:
                        ago = "刚刚"
                    elif delta.total_seconds() < 3600:
                        ago = f"{int(delta.total_seconds() / 60)}分钟前"
                    elif delta.total_seconds() < 86400:
                        ago = f"{int(delta.total_seconds() / 3600)}小时前"
                    else:
                        ago = f"{delta.days}天前"
                else:
                    ago = "从未活跃"
            except (ValueError, TypeError):
                ago = "未知"

            snapshot[meta.id] = {
                "name": meta.name or meta.id,
                "status": meta.status,
                "priority": meta.priority,
                "last_active_ago": ago,
                "warnings": list(meta.warnings),
                "rounds_completed": meta.rounds_completed,
            }

        async with self._goal_states_lock:
            self._goal_states = snapshot
            self._goal_states_at = time.monotonic()

    def _format_goal_summary(self) -> str:
        """Format the cached goal-state snapshot as a concise text block.

        Returns an empty string when no goals are cached (first scan not yet
        completed or workspace has no goals).
        """
        states = self._goal_states  # atomic reference capture
        if not states:
            return ""

        lines: list[str] = []
        # Sort: active first, then by priority (higher first)
        sorted_ids = sorted(
            states,
            key=lambda gid: (
                0 if states[gid].get("status") == "active" else 1,
                -int(states[gid].get("priority", 5)),
            ),
        )

        for gid in sorted_ids:
            gs = states[gid]
            status = str(gs.get("status", "?"))
            name = str(gs.get("name", gid))
            ago = str(gs.get("last_active_ago", ""))
            warnings = list(gs.get("warnings") or [])
            rounds = int(gs.get("rounds_completed", 0))

            status_icon = {"active": "●", "paused": "◐", "completed": "○", "archived": "✕"}.get(
                status, "?"
            )
            line = f"{status_icon} **{name}** ({gid}): {status}"
            if ago:
                line += f", {ago}"
            if rounds:
                line += f", {rounds}轮完成"
            if warnings:
                line += f", ⚠{len(warnings)}个警告"
            lines.append(line)

        if not lines:
            return ""

        return "# 当前目标状态\n\n" + "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # DMN consumer — polls pending/ for cognitive evolution tasks
    # ------------------------------------------------------------------

    async def _notify_dmn(self, content: str) -> None:
        """Send a DMN progress report to the user's active channel.

        Non-blocking: failures are silently logged so a disconnected
        channel never stalls the DMN consumer.
        """
        channel, chat_id = self._last_user_route
        try:
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=channel,
                    chat_id=chat_id,
                    content=f"🧠 **DMN 认知网络**\n{content}",
                )
            )
        except Exception:
            logger.debug("[DMN] 汇报失败 (channel={})", channel)

    async def _run_dmn_consumer(self) -> None:
        """DMN consumer: polls pending/ for cognitive evolution tasks.

        Works like WorkerPool but only consumes ``cognition-evolution__*``
        and ``dmn__*`` prefixed ``.task`` files.  Each task is processed
        by ``_run_dmn_turn`` which injects DMN system prompts and reuses
        the existing AgentLoop (LLM + tools).

        Also periodically refreshes the goal-state cache (every 30 s) so
        dialogue-mode turns can inject up-to-date TPN awareness.
        """
        from vingobot.core.pending_queue import PendingQueue

        queue = PendingQueue()
        logger.info("[DMN消费者] 已启动")
        await self._notify_dmn("🟢 DMN 消费者已启动，开始监听认知演化任务。")

        _scan_interval = 30.0  # seconds between goal-state snapshots

        while self._running:
            # ── Consume cognitive-evolution / DMN tasks ──────────
            claimed = queue.try_consume_by_prefix("cognition-evolution")
            if claimed is None:
                claimed = queue.try_consume_by_prefix("dmn")

            if claimed is not None:
                task, file_path = claimed
                desc = task.description[:100]
                logger.info("[DMN] 消费学习任务: {}…", desc)

                await self._notify_dmn(f"🔍 开始处理: {desc}")
                try:
                    await self._run_dmn_turn(task.description)
                    await self._notify_dmn(f"✅ 任务完成: {desc}")
                except asyncio.CancelledError:
                    await self._notify_dmn(f"⏹ 任务已取消: {desc}")
                    logger.info("[DMN] 学习任务被取消")
                except Exception as exc:
                    await self._notify_dmn(f"❌ 任务异常: {desc}\n> {exc}")
                    logger.exception("[DMN] 学习任务异常")
                finally:
                    queue.delete_task_file(file_path)

                # Still refresh goal-state cache periodically even when busy
                now = time.monotonic()
                if now - self._goal_states_at >= _scan_interval:
                    await self._scan_goal_states()
                continue

            # ── Idle: refresh goal-state cache periodically ─────
            now = time.monotonic()
            if now - self._goal_states_at >= _scan_interval:
                await self._scan_goal_states()

            try:
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break

        logger.info("[DMN消费者] 已退出")
        await self._notify_dmn("⚪ DMN 消费者已停止。")

    async def _run_dmn_turn(self, task_description: str) -> None:
        """Execute one DMN task by reusing the AgentLoop.

        Injects a DMN identity section into the system prompt so the LLM
        acts as the system's cognitive steward.  Uses a dedicated ``_dmn``
        session to keep DMN context isolated from user conversation state.

        If the task description follows the structured cognition-evolution
        format (produced by ``_build_evolution_task_description``), the
        handler first asks the LLM to generate the asset content, then
        calls ``cognition_evolver.execute_evolution_action()`` to create
        the structured asset on disk.  Otherwise falls back to free-form
        LLM-driven execution.
        """
        # ── Try structured cognition-evolution path ────────────────
        parsed = self._parse_evolution_task(task_description)
        if parsed is not None:
            try:
                success = await self._execute_evolution_task(parsed, task_description)
                if not success:
                    raise RuntimeError("结构化认知演化任务内部失败")
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[DMN] 结构化认知演化任务异常")
            return

        # ── Fallback: free-form DMN execution ──────────────────────
        dmn_gate = self._dmn_gate or nullcontext()
        async with dmn_gate:
            dmn_injection = (
                "## 全局认知维护（DMN）模式\n\n"
                "你当前处于全局认知维护模式，身份是系统的认知管家。\n"
                "职责：\n"
                "1. **认知演化** — 创建/更新 L1 技能、L2 模型、L3 格栅\n"
                "2. **全局维护** — 检查 meta.json、归档、暂停、成长日志\n\n"
                "**约束**:\n"
                "- 所有文件写入必须在 `.taiji/` 目录下\n"
                "- 不能修改 `.taiji/cognition/truths/` 下的 L4 不可变真理"
            )

            # Build system prompt with DMN identity appended
            base_prompt = self.context.build_system_prompt(channel="system")
            system_prompt = f"{base_prompt}\n\n---\n\n{dmn_injection}"

            key = "_dmn"
            session = self.sessions.get_or_create(key)
            history = session.get_history(max_messages=3, max_tokens=2048)

            messages = [
                {"role": "system", "content": system_prompt},
                *history,
                {"role": "user", "content": task_description},
            ]

            try:
                final_content, _, all_msgs, _, _ = await self._run_agent_loop(
                    messages,
                    session=session,
                    channel="_dmn",
                    chat_id="_dmn",
                    session_key=key,
                )
                self._save_turn(session, all_msgs, 1 + len(history))
                session.retain_recent_legal_suffix(15)
                self.sessions.save(session)
                logger.info("[DMN] 任务完成: {}…", (final_content or "")[:120])
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[DMN] 任务执行异常")

    @staticmethod
    def _parse_evolution_task(
        task_description: str,
    ) -> dict[str, str] | None:
        """Parse a structured cognition-evolution task description.

        Returns a dict with ``action``, ``target_name``, ``description``,
        ``context`` (JSON string), and ``source_task_id`` if the task
        follows the format produced by ``_build_evolution_task_description``.
        Returns ``None`` for unstructured / free-form tasks.
        """
        import re

        lines = task_description.strip().split("\n")
        if not lines or not lines[0].startswith("# 认知演化任务:"):
            return None

        action = lines[0].replace("# 认知演化任务:", "").strip()
        if not action:
            return None

        result: dict[str, str] = {"action": action}

        for line in lines[1:]:
            stripped = line.strip()
            if stripped.startswith("目标:"):
                result["target_name"] = stripped.replace("目标:", "").strip()
            elif stripped.startswith("描述:"):
                result["description"] = stripped.replace("描述:", "").strip()
            elif stripped.startswith("来源任务:"):
                result["source_task_id"] = stripped.replace("来源任务:", "").strip()
            elif stripped.startswith("来源目标:"):
                result["source_goal_id"] = stripped.replace("来源目标:", "").strip()
            elif stripped.startswith("## "):
                break  # stop at next section header

        # Extract context JSON if present
        ctx_match = re.search(r"## 上下文\s*\n(.*?)(?:\n## |\n# |$)", task_description, re.DOTALL)
        if ctx_match:
            result["context"] = ctx_match.group(1).strip()

        # Validate required fields
        if not result.get("target_name"):
            logger.warning("[DMN] 演化任务缺少目标名称，回退到自由执行")
            return None

        return result

    async def _execute_evolution_task(
        self,
        parsed: dict[str, str],
        task_description: str,
    ) -> bool:
        """Execute a structured cognition-evolution task.

        1. Ask the LLM to generate the asset content (skill body, model text, etc.)
        2. Call ``cognition_evolver.execute_evolution_action()`` to create the file
        3. Invalidate caches so the next Weaver round discovers new assets
        4. Refresh skill tool registry for newly created L1 skills
        """
        from vingobot.goal import cognition_evolver, weaver
        from vingobot.goal.cognition_tools import update_nav_index
        from vingobot.goal.grid_types import CognitionEvolutionAction

        action = parsed["action"]
        target_name = parsed.get("target_name", "")
        description = parsed.get("description", "")
        ctx_json = parsed.get("context", "{}")

        import json as _json

        try:
            ctx = _json.loads(ctx_json) if ctx_json else {}
        except _json.JSONDecodeError:
            ctx = {}

        logger.info(
            "[DMN] 结构化认知演化: action={}, target={}",
            action,
            target_name,
        )

        # ── Step 1: Ask LLM to generate asset content ─────────────
        await self._notify_dmn(
            f"📝 正在生成认知资产: {action} → {target_name}\n> {description[:80]}"
        )
        # Use a focused prompt instructing the LLM to produce the content
        content_prompt = (
            f"你正在创建认知资产。请生成以下内容：\n\n"
            f"动作: {action}\n"
            f"目标名称: {target_name}\n"
            f"描述: {description}\n\n"
            f"原始任务:\n{task_description}\n\n"
            f"请生成具体的内容定义（不要创建文件，只需输出内容文本）：\n"
            f"- 如果是 learn_skill 或 precipitate_skill: 输出技能的具体步骤和工具定义\n"
            f"- 如果是 precipitate_model: 输出经验模型的抽象模式和关键要点\n"
            f"- 如果是 create_grid: 输出该认知领域的 L1/L2 资产建议列表\n\n"
            f"请用中文回答，保持简洁。"
        )

        base_prompt = self.context.build_system_prompt(channel="system")
        system_prompt = (
            f"{base_prompt}\n\n---\n\n"
            f"## 认知演化内容生成器\n\n"
            f"你是系统的认知架构师，负责生成高质量认知资产内容。"
        )

        key = "_dmn_evo"
        session = self.sessions.get_or_create(key)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_prompt},
        ]

        llm_content = ""
        dmn_gate = self._dmn_gate or nullcontext()
        async with dmn_gate:
            try:
                final_content, _, all_msgs, _, _ = await self._run_agent_loop(
                    messages,
                    session=session,
                    channel="_dmn",
                    chat_id="_dmn_evo",
                    session_key=key,
                )
                llm_content = final_content or ""
                self._save_turn(session, all_msgs, 1)
                session.retain_recent_legal_suffix(15)
                self.sessions.save(session)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[DMN] 演化内容生成失败")
                return False

        # ── Step 2: Inject LLM-generated content into context ─────
        if action in ("precipitate_model",) and llm_content:
            ctx["insight"] = llm_content
        if action in ("learn_skill", "precipitate_skill") and llm_content:
            if "description" not in ctx or not ctx["description"]:
                ctx["description"] = llm_content[:200]

        # ── Step 3: Execute structured creation ───────────────────
        ea = CognitionEvolutionAction(
            action=action,
            target_name=target_name,
            description=description,
            source_task_id=parsed.get("source_task_id", ""),
            source_goal_id=parsed.get("source_goal_id", ""),
            context=ctx,
        )

        try:
            created = await cognition_evolver.execute_evolution_action(ea)
            status = "创建" if created else "已存在(跳过)"
            logger.info("[DMN] 认知演化结果: {} {} → {}", action, target_name, status)
            await self._notify_dmn(
                f"{'✅ 已' + status if created else '⏭ ' + status}: {action} → {target_name}"
            )
        except Exception:
            logger.exception("[DMN] 认知演化执行失败: {} {}", action, target_name)
            await self._notify_dmn(f"❌ 演化失败: {action} → {target_name}")
            return False

        # ── Step 4: Invalidate caches ────────────────────────────
        try:
            weaver.invalidate_cognition_caches()
            update_nav_index()
            logger.debug("[DMN] 已刷新认知缓存和导航索引")
        except Exception:
            logger.exception("[DMN] 缓存刷新失败")

        # ── Step 5: Refresh skill tool registry ──────────────────
        if action in ("learn_skill", "precipitate_skill"):
            try:
                from vingobot.core.workspace import get_workspace_paths
                from vingobot.goal.skill_parser import (
                    parse_skill_md,
                    register_skill_tools_from_meta,
                )

                wp = get_workspace_paths()
                skill_md = wp.skills / target_name / "SKILL.md"
                if skill_md.is_file():
                    meta = parse_skill_md(skill_md)
                    if meta:
                        register_skill_tools_from_meta(meta)
                        logger.info("[DMN] 已注册新技能工具: {}", target_name)
                        await self._notify_dmn(f"🔧 已注册新技能工具: `{target_name}`")
            except Exception:
                logger.exception("[DMN] 技能工具注册失败")

        return True

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        pending_queue: asyncio.Queue | None = None,
        _already_locked: bool = False,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        self._refresh_provider_snapshot()
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (
                msg.chat_id.split(":", 1) if ":" in msg.chat_id else ("cli", msg.chat_id)
            )
            logger.info("Processing system message from {}", msg.sender_id)
            # Honor session_key_override so subagent announces from threaded
            # callers route to the originating thread session, not the
            # channel-level session derived from chat_id.
            key = msg.session_key_override or f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            if self._restore_runtime_checkpoint(session):
                self.sessions.save(session)
            if self._restore_pending_user_turn(session):
                self.sessions.save(session)

            session, pending = self.auto_compact.prepare_session(session, key)

            await self.consolidator.maybe_consolidate_by_tokens(
                session,
                session_summary=pending,
                _already_locked=_already_locked,
            )
            # Persist subagent follow-ups into durable history BEFORE prompt
            # assembly. ContextBuilder merges adjacent same-role messages for
            # provider compatibility, which previously caused the follow-up to
            # disappear from session.messages while still being visible to the
            # LLM via the merged prompt. See _persist_subagent_followup.
            is_subagent = msg.sender_id == "subagent"
            if is_subagent and self._persist_subagent_followup(session, msg):
                self.sessions.save(session)
            self._set_tool_context(
                channel,
                chat_id,
                msg.metadata.get("message_id"),
                msg.metadata,
                session_key=key,
            )
            _hist_kwargs: dict[str, Any] = {
                "max_messages": self._max_messages,
                "max_tokens": self._replay_token_budget(),
                "include_timestamps": True,
            }
            history = session.get_history(**_hist_kwargs)
            current_role = "assistant" if is_subagent else "user"

            # Subagent content is already in `history` above; passing it again
            # as current_message would double-project it into the prompt.
            messages = self.context.build_messages(
                history=history,
                current_message="" if is_subagent else msg.content,
                channel=channel,
                chat_id=chat_id,
                session_summary=pending,
                current_role=current_role,
                sender_id=msg.sender_id,
                goal_summary=self._format_goal_summary(),
            )
            final_content, _, all_msgs, stop_reason, _ = await self._run_agent_loop(
                messages,
                session=session,
                channel=channel,
                chat_id=chat_id,
                message_id=msg.metadata.get("message_id"),
                metadata=msg.metadata,
                session_key=key,
                pending_queue=pending_queue,
            )
            self._save_turn(session, all_msgs, 1 + len(history))
            session.enforce_file_cap(on_archive=self.context.memory.raw_archive)
            self._clear_runtime_checkpoint(session)
            self.sessions.save(session)
            self._schedule_background(self.consolidator.maybe_consolidate_by_tokens(session))
            options = ask_user_options_from_messages(all_msgs) if stop_reason == "ask_user" else []
            content, buttons = ask_user_outbound(
                final_content or "Background task completed.",
                options,
                channel,
            )
            # Reconstruct channel-specific metadata from session.key so the
            # outbound reply lands in the originating thread (not the channel
            # top-level). The announce InboundMessage carries only
            # injected_event metadata; we recover thread_ts from the session
            # key, which slack writes as "slack:<chat_id>:<thread_ts>".
            outbound_metadata: dict[str, Any] = {}
            if channel == "slack" and key.startswith("slack:") and key.count(":") >= 2:
                outbound_metadata["slack"] = {"thread_ts": key.split(":", 2)[2]}
            if origin_message_id := msg.metadata.get("origin_message_id"):
                outbound_metadata["origin_message_id"] = origin_message_id
            return OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content=content,
                buttons=buttons,
                metadata=outbound_metadata,
            )

        # Extract document text from media at the processing boundary so all
        # channels benefit without format-specific logic in ContextBuilder.
        if msg.media:
            new_content, image_only = extract_documents(msg.content, msg.media)
            msg = dataclasses.replace(msg, content=new_content, media=image_only)

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)
        if self._restore_runtime_checkpoint(session):
            self.sessions.save(session)
        if self._restore_pending_user_turn(session):
            self.sessions.save(session)

        session, pending = self.auto_compact.prepare_session(session, key)

        # Slash commands
        raw = msg.content.strip()
        ctx = CommandContext(msg=msg, session=session, key=key, raw=raw, loop=self)
        if result := await self.commands.dispatch(ctx):
            return result

        await self.consolidator.maybe_consolidate_by_tokens(
            session,
            session_summary=pending,
            _already_locked=_already_locked,
        )

        self._set_tool_context(
            msg.channel,
            msg.chat_id,
            msg.metadata.get("message_id"),
            msg.metadata,
            session_key=key,
        )
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        _hist_kwargs: dict[str, Any] = {
            "max_messages": self._max_messages,
            "max_tokens": self._replay_token_budget(),
            "include_timestamps": True,
        }
        history = session.get_history(**_hist_kwargs)

        pending_ask_id = pending_ask_user_id(history)
        if pending_ask_id:
            initial_messages = ask_user_tool_result_messages(
                self.context.build_system_prompt(channel=msg.channel),
                history,
                pending_ask_id,
                msg.content,
            )
        else:
            initial_messages = self.context.build_messages(
                history=history,
                current_message=msg.content,
                session_summary=pending,
                media=msg.media if msg.media else None,
                channel=msg.channel,
                chat_id=self._runtime_chat_id(msg),
                sender_id=msg.sender_id,
                goal_summary=self._format_goal_summary(),
            )

        async def _bus_progress(
            content: str,
            *,
            tool_hint: bool = False,
            tool_events: list[dict[str, Any]] | None = None,
        ) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            if tool_events:
                meta["_tool_events"] = tool_events
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=content,
                    metadata=meta,
                )
            )

        async def _on_retry_wait(content: str) -> None:
            meta = dict(msg.metadata or {})
            meta["_retry_wait"] = True
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=content,
                    metadata=meta,
                )
            )

        # Persist the triggering user message up front so a mid-turn crash
        # doesn't silently lose the prompt on recovery. ``media`` rides along
        # as raw on-disk paths — sanitized image blocks are stripped from
        # JSONL, and webui replay needs the paths to mint signed URLs.
        user_persisted_early = False
        media_paths = [p for p in (msg.media or []) if isinstance(p, str) and p]
        has_text = isinstance(msg.content, str) and msg.content.strip()
        if not pending_ask_id and (has_text or media_paths):
            extra: dict[str, Any] = {"media": list(media_paths)} if media_paths else {}
            text = msg.content if isinstance(msg.content, str) else ""
            session.add_message("user", text, **extra)
            self._mark_pending_user_turn(session)
            self.sessions.save(session)
            user_persisted_early = True

        final_content, _, all_msgs, stop_reason, had_injections = await self._run_agent_loop(
            initial_messages,
            on_progress=on_progress or _bus_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
            on_retry_wait=_on_retry_wait,
            session=session,
            channel=msg.channel,
            chat_id=msg.chat_id,
            message_id=msg.metadata.get("message_id"),
            metadata=msg.metadata,
            session_key=key,
            pending_queue=pending_queue,
        )

        if final_content is None or not final_content.strip():
            final_content = EMPTY_FINAL_RESPONSE_MESSAGE

        # Skip the already-persisted user message when saving the turn
        save_skip = 1 + len(history) + (1 if user_persisted_early else 0)
        self._save_turn(session, all_msgs, save_skip)
        session.enforce_file_cap(on_archive=self.context.memory.raw_archive)
        self._clear_pending_user_turn(session)
        self._clear_runtime_checkpoint(session)
        self.sessions.save(session)
        self._schedule_background(self.consolidator.maybe_consolidate_by_tokens(session))

        # When follow-up messages were injected mid-turn, a later natural
        # language reply may address those follow-ups and should not be
        # suppressed just because MessageTool was used earlier in the turn.
        # However, if the turn falls back to the empty-final-response
        # placeholder, suppress it when the real user-visible output already
        # came from MessageTool.
        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            if not had_injections or stop_reason == "empty_final_response":
                return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)

        meta = dict(msg.metadata or {})
        final_content, buttons = ask_user_outbound(
            final_content,
            ask_user_options_from_messages(all_msgs) if stop_reason == "ask_user" else [],
            msg.channel,
        )
        if on_stream is not None and stop_reason not in {"ask_user", "error"}:
            meta["_streamed"] = True
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=meta,
            buttons=buttons,
        )

    def _sanitize_persisted_blocks(
        self,
        content: list[dict[str, Any]],
        *,
        should_truncate_text: bool = False,
        drop_runtime: bool = False,
    ) -> list[dict[str, Any]]:
        """Strip volatile multimodal payloads before writing session history."""
        filtered: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                filtered.append(block)
                continue

            if (
                drop_runtime
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
                and block["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG)
            ):
                continue

            if block.get("type") == "image_url" and block.get("image_url", {}).get(
                "url", ""
            ).startswith("data:image/"):
                path = (block.get("_meta") or {}).get("path", "")
                filtered.append({"type": "text", "text": image_placeholder_text(path)})
                continue

            if block.get("type") == "text" and isinstance(block.get("text"), str):
                text = block["text"]
                if should_truncate_text and len(text) > self.max_tool_result_chars:
                    text = truncate_text_fn(text, self.max_tool_result_chars)
                filtered.append({**block, "text": text})
                continue

            filtered.append(block)

        return filtered

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime

        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            if role == "tool":
                if isinstance(content, str) and len(content) > self.max_tool_result_chars:
                    entry["content"] = truncate_text_fn(content, self.max_tool_result_chars)
                elif isinstance(content, list):
                    filtered = self._sanitize_persisted_blocks(content, should_truncate_text=True)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            elif role == "user":
                if isinstance(content, str) and content.startswith(
                    ContextBuilder._RUNTIME_CONTEXT_TAG
                ):
                    # Strip the entire runtime-context block (including any session summary).
                    # The block is bounded by _RUNTIME_CONTEXT_TAG and _RUNTIME_CONTEXT_END.
                    end_marker = ContextBuilder._RUNTIME_CONTEXT_END
                    end_pos = content.find(end_marker)
                    if end_pos >= 0:
                        after = content[end_pos + len(end_marker) :].lstrip("\n")
                        if after:
                            entry["content"] = after
                        else:
                            continue
                    else:
                        # Fallback: no end marker found, strip the tag prefix
                        after_tag = content[len(ContextBuilder._RUNTIME_CONTEXT_TAG) :].lstrip("\n")
                        if after_tag.strip():
                            entry["content"] = after_tag
                        else:
                            continue
                if isinstance(content, list):
                    filtered = self._sanitize_persisted_blocks(content, drop_runtime=True)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    def _persist_subagent_followup(self, session: Session, msg: InboundMessage) -> bool:
        """Persist subagent follow-ups before prompt assembly so history stays durable.

        Returns True if a new entry was appended; False if the follow-up was
        deduped (same ``subagent_task_id`` already in session) or carries no
        content worth persisting.
        """
        if not msg.content:
            return False
        task_id = msg.metadata.get("subagent_task_id") if isinstance(msg.metadata, dict) else None
        if task_id and any(
            m.get("injected_event") == "subagent_result" and m.get("subagent_task_id") == task_id
            for m in session.messages
        ):
            return False
        session.add_message(
            "assistant",
            msg.content,
            sender_id=msg.sender_id,
            injected_event="subagent_result",
            subagent_task_id=task_id,
        )
        return True

    def _set_runtime_checkpoint(self, session: Session, payload: dict[str, Any]) -> None:
        """Persist the latest in-flight turn state into session metadata."""
        session.metadata[self._RUNTIME_CHECKPOINT_KEY] = payload
        self.sessions.save(session)

    def _mark_pending_user_turn(self, session: Session) -> None:
        session.metadata[self._PENDING_USER_TURN_KEY] = True

    def _clear_pending_user_turn(self, session: Session) -> None:
        session.metadata.pop(self._PENDING_USER_TURN_KEY, None)

    def _clear_runtime_checkpoint(self, session: Session) -> None:
        if self._RUNTIME_CHECKPOINT_KEY in session.metadata:
            session.metadata.pop(self._RUNTIME_CHECKPOINT_KEY, None)

    @staticmethod
    def _checkpoint_message_key(message: dict[str, Any]) -> tuple[Any, ...]:
        return (
            message.get("role"),
            message.get("content"),
            message.get("tool_call_id"),
            message.get("name"),
            message.get("tool_calls"),
            message.get("reasoning_content"),
            message.get("thinking_blocks"),
        )

    def _restore_runtime_checkpoint(self, session: Session) -> bool:
        """Materialize an unfinished turn into session history before a new request."""
        from datetime import datetime

        checkpoint = session.metadata.get(self._RUNTIME_CHECKPOINT_KEY)
        if not isinstance(checkpoint, dict):
            return False

        assistant_message = checkpoint.get("assistant_message")
        completed_tool_results = checkpoint.get("completed_tool_results") or []
        pending_tool_calls = checkpoint.get("pending_tool_calls") or []

        restored_messages: list[dict[str, Any]] = []
        if isinstance(assistant_message, dict):
            restored = dict(assistant_message)
            restored.setdefault("timestamp", datetime.now().isoformat())
            restored_messages.append(restored)
        for message in completed_tool_results:
            if isinstance(message, dict):
                restored = dict(message)
                restored.setdefault("timestamp", datetime.now().isoformat())
                restored_messages.append(restored)
        for tool_call in pending_tool_calls:
            if not isinstance(tool_call, dict):
                continue
            tool_id = tool_call.get("id")
            name = ((tool_call.get("function") or {}).get("name")) or "tool"
            restored_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": name,
                    "content": "Error: Task interrupted before this tool finished.",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        overlap = 0
        max_overlap = min(len(session.messages), len(restored_messages))
        for size in range(max_overlap, 0, -1):
            existing = session.messages[-size:]
            restored = restored_messages[:size]
            if all(
                self._checkpoint_message_key(left) == self._checkpoint_message_key(right)
                for left, right in zip(existing, restored)
            ):
                overlap = size
                break
        session.messages.extend(restored_messages[overlap:])

        self._clear_pending_user_turn(session)
        self._clear_runtime_checkpoint(session)
        return True

    def _restore_pending_user_turn(self, session: Session) -> bool:
        """Close a turn that only persisted the user message before crashing."""
        from datetime import datetime

        if not session.metadata.get(self._PENDING_USER_TURN_KEY):
            return False

        if session.messages and session.messages[-1].get("role") == "user":
            session.messages.append(
                {
                    "role": "assistant",
                    "content": "Error: Task interrupted before a response was generated.",
                    "timestamp": datetime.now().isoformat(),
                }
            )
            session.updated_at = datetime.now()

        self._clear_pending_user_turn(session)
        return True

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        media: list[str] | None = None,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a message directly and return the outbound payload."""
        await self._connect_mcp()
        msg = InboundMessage(
            channel=channel,
            sender_id="user",
            chat_id=chat_id,
            content=content,
            media=media or [],
        )
        return await self._process_message(
            msg,
            session_key=session_key,
            on_progress=on_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
        )
