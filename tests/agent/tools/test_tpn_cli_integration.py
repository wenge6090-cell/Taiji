"""Test: CLI paths properly bind _tpn_bot and TPN tool can control sixiang pool."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_config():
    """模拟 CLI 中 _load_runtime_config 返回的 Config 对象."""
    from vingobot.config.schema import ToolsConfig, TpnToolConfig

    cfg = MagicMock()
    cfg.workspace_path = Path.cwd()

    # Nested mock for cfg.agents.defaults
    sixiang_cfg = MagicMock()
    sixiang_cfg.max_concurrent_workers = 3

    defaults = MagicMock()
    defaults.sixiang = sixiang_cfg
    defaults.model = "test-model"
    defaults.max_tool_iterations = 20
    defaults.context_window_tokens = 8000
    defaults.provider_retry_mode = "standard"
    defaults.timezone = "Asia/Shanghai"
    defaults.unified_session = False
    defaults.disabled_skills = None
    defaults.session_ttl_minutes = 0
    defaults.consolidation_ratio = 0.5
    defaults.max_messages = 120

    cfg.agents = MagicMock()
    cfg.agents.defaults = defaults

    cfg.tools = ToolsConfig(tpn=TpnToolConfig(enable=True))
    cfg.channels = MagicMock()
    cfg.tools.restrict_to_workspace = False
    cfg.tools.mcp_servers = {}
    cfg.tools.web = MagicMock()
    cfg.tools.exec = MagicMock()
    return cfg


class TestCliTpnBot:
    """Verify _tpn_bot binding and TPN tool pool control in CLI-like paths."""

    def test_agent_path_binds_tpn_bot(self, mock_config):
        """模拟 agent 入口路径：创建 AgentLoop 后 _tpn_bot 应该已绑定."""
        from vingobot.agent.loop import AgentLoop
        from vingobot.bus.queue import MessageBus
        from vingobot.vingobot import vingobot as _vingobot

        cfg = mock_config
        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"

        # ── 模拟 agent 入口的流程 ──
        agent_loop = AgentLoop(
            bus=bus,
            provider=provider,
            workspace=cfg.workspace_path,
            model=cfg.agents.defaults.model,
            max_iterations=cfg.agents.defaults.max_tool_iterations,
            context_window_tokens=cfg.agents.defaults.context_window_tokens,
            provider_retry_mode=cfg.agents.defaults.provider_retry_mode,
            timezone=cfg.agents.defaults.timezone,
            unified_session=cfg.agents.defaults.unified_session,
            disabled_skills=cfg.agents.defaults.disabled_skills,
            session_ttl_minutes=cfg.agents.defaults.session_ttl_minutes,
            consolidation_ratio=cfg.agents.defaults.consolidation_ratio,
            max_messages=cfg.agents.defaults.max_messages,
            tools_config=cfg.tools,
        )

        # 模拟 CLI 补充的绑定代码
        _tpn_bot = _vingobot(agent_loop)
        _tpn_bot._sixiang_cfg = cfg.agents.defaults.sixiang
        _tpn_bot._provider = provider
        agent_loop._tpn_bot = _tpn_bot

        # ── 验证 _tpn_bot 已绑定 ──
        assert agent_loop._tpn_bot is not None
        assert agent_loop._tpn_bot is _tpn_bot

        # ── 验证 TPN 工具能识别池状态 ──
        tpn_tool = agent_loop.tools.get("tpn")
        assert tpn_tool is not None

        # 初始状态：池未运行
        assert tpn_tool._pool_running() is False

        status_result = tpn_tool._do_status()
        assert "未初始化" not in status_result  # 不再显示"不可用"
        assert "已停止" in status_result

    def test_tpn_start_stop_pool(self, mock_config):
        """TPN 工具的 start/stop 能正确控制六爻协程池."""
        from vingobot.agent.loop import AgentLoop
        from vingobot.bus.queue import MessageBus
        from vingobot.vingobot import vingobot as _vingobot

        cfg = mock_config
        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"

        agent_loop = AgentLoop(
            bus=bus,
            provider=provider,
            workspace=cfg.workspace_path,
            model=cfg.agents.defaults.model,
            max_iterations=cfg.agents.defaults.max_tool_iterations,
            context_window_tokens=cfg.agents.defaults.context_window_tokens,
            provider_retry_mode=cfg.agents.defaults.provider_retry_mode,
            timezone=cfg.agents.defaults.timezone,
            unified_session=cfg.agents.defaults.unified_session,
            disabled_skills=cfg.agents.defaults.disabled_skills,
            session_ttl_minutes=cfg.agents.defaults.session_ttl_minutes,
            consolidation_ratio=cfg.agents.defaults.consolidation_ratio,
            max_messages=cfg.agents.defaults.max_messages,
            tools_config=cfg.tools,
        )

        _tpn_bot = _vingobot(agent_loop)
        _tpn_bot._sixiang_cfg = cfg.agents.defaults.sixiang
        _tpn_bot._provider = provider
        agent_loop._tpn_bot = _tpn_bot

        tpn_tool = agent_loop.tools.get("tpn")
        assert tpn_tool is not None

        # TPN start
        result = asyncio.run(tpn_tool.execute(action="start"))
        assert "六爻协程池已启动" in result
        assert agent_loop._tpn_bot.sixiang_running is True
        assert tpn_tool._pool_running() is True

        # TPN status 应该显示运行中
        status_result = tpn_tool._do_status()
        assert "🟢 运行中" in status_result

        # TPN stop
        result = asyncio.run(tpn_tool.execute(action="stop"))
        assert "六爻协程池已停止" in result
        assert agent_loop._tpn_bot.sixiang_running is False
        assert tpn_tool._pool_running() is False

        # TPN status 应该显示已停止
        status_result = tpn_tool._do_status()
        assert "已停止" in status_result

    def test_tpn_start_when_already_running(self, mock_config):
        """重复 start 应该提示已在运行."""
        from vingobot.agent.loop import AgentLoop
        from vingobot.bus.queue import MessageBus
        from vingobot.vingobot import vingobot as _vingobot

        cfg = mock_config
        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"

        agent_loop = AgentLoop(
            bus=bus,
            provider=provider,
            workspace=cfg.workspace_path,
            model=cfg.agents.defaults.model,
            max_iterations=cfg.agents.defaults.max_tool_iterations,
            context_window_tokens=cfg.agents.defaults.context_window_tokens,
            provider_retry_mode=cfg.agents.defaults.provider_retry_mode,
            timezone=cfg.agents.defaults.timezone,
            unified_session=cfg.agents.defaults.unified_session,
            disabled_skills=cfg.agents.defaults.disabled_skills,
            session_ttl_minutes=cfg.agents.defaults.session_ttl_minutes,
            consolidation_ratio=cfg.agents.defaults.consolidation_ratio,
            max_messages=cfg.agents.defaults.max_messages,
            tools_config=cfg.tools,
        )

        _tpn_bot = _vingobot(agent_loop)
        _tpn_bot._sixiang_cfg = cfg.agents.defaults.sixiang
        _tpn_bot._provider = provider
        agent_loop._tpn_bot = _tpn_bot

        tpn_tool = agent_loop.tools.get("tpn")

        async def run():
            r1 = await tpn_tool.execute(action="start")
            r2 = await tpn_tool.execute(action="start")
            return r1, r2

        r1, r2 = asyncio.run(run())
        assert "已启动" in r1
        assert "已经在运行中" in r2

        # 清理
        asyncio.run(tpn_tool.execute(action="stop"))

    def test_tpn_stop_when_not_running(self, mock_config):
        """未运行时 stop 应该提示未在运行."""
        from vingobot.agent.loop import AgentLoop
        from vingobot.bus.queue import MessageBus
        from vingobot.vingobot import vingobot as _vingobot

        cfg = mock_config
        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"

        agent_loop = AgentLoop(
            bus=bus,
            provider=provider,
            workspace=cfg.workspace_path,
            model=cfg.agents.defaults.model,
            max_iterations=cfg.agents.defaults.max_tool_iterations,
            context_window_tokens=cfg.agents.defaults.context_window_tokens,
            provider_retry_mode=cfg.agents.defaults.provider_retry_mode,
            timezone=cfg.agents.defaults.timezone,
            unified_session=cfg.agents.defaults.unified_session,
            disabled_skills=cfg.agents.defaults.disabled_skills,
            session_ttl_minutes=cfg.agents.defaults.session_ttl_minutes,
            consolidation_ratio=cfg.agents.defaults.consolidation_ratio,
            max_messages=cfg.agents.defaults.max_messages,
            tools_config=cfg.tools,
        )

        _tpn_bot = _vingobot(agent_loop)
        _tpn_bot._sixiang_cfg = cfg.agents.defaults.sixiang
        _tpn_bot._provider = provider
        agent_loop._tpn_bot = _tpn_bot

        tpn_tool = agent_loop.tools.get("tpn")

        result = asyncio.run(tpn_tool.execute(action="stop"))
        assert "未在运行" in result
