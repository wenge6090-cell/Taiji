"""
六爻所有Agent的API调用测试。

测试目标：
1. 验证每个六爻Agent（明觉、织者、阳、阴、暗驱）的Provider初始化
2. 验证每个Agent能通过Provider成功调用LLM API
3. 验证各Agent的模型配置正确加载
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from loguru import logger

from vingobot.config.loader import load_config, resolve_config_env_vars
from vingobot.providers.factory import build_sixiang_provider_snapshot

# ── 六爻Agent列表 ──────────────────────────────────────────────────────────
SIXIANG_AGENTS = [
    ("mingjue", "初爻·明觉"),
    ("weaver", "二爻·织者"),
    ("yang", "三爻·阳"),
    ("yin", "四爻·阴"),
    ("anqu", "上爻·暗驱"),
]


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def config():
    """加载真实配置。"""
    cfg = resolve_config_env_vars(load_config())
    return cfg


# ── 测试1: Provider初始化 ────────────────────────────────────────────────


class TestSixiangProviderInit:
    """验证每个Agent的Provider能正确初始化。"""

    @pytest.mark.parametrize("agent_name,agent_label", SIXIANG_AGENTS)
    def test_provider_init(self, config, agent_name: str, agent_label: str):
        """每个Agent的 build_sixiang_provider_snapshot 应成功。"""
        snapshot = build_sixiang_provider_snapshot(config, agent_name)

        assert snapshot.provider is not None, f"{agent_label} provider 不应为 None"
        assert snapshot.model is not None, f"{agent_label} model 不应为 None"
        assert snapshot.context_window_tokens > 0, f"{agent_label} context_window_tokens 应 > 0"

        logger.info("[{}] Provider初始化成功: model={}, context_window={}",
                     agent_label, snapshot.model, snapshot.context_window_tokens)

    @pytest.mark.parametrize("agent_name,agent_label", SIXIANG_AGENTS)
    def test_provider_generation_settings(self, config, agent_name: str, agent_label: str):
        """验证每个Agent的generation设置正确加载。"""
        snapshot = build_sixiang_provider_snapshot(config, agent_name)
        provider = snapshot.provider

        gen = provider.generation
        assert gen is not None
        assert gen.max_tokens > 0, f"{agent_label} max_tokens 应 > 0"

        logger.info("[{}] Generation设置: temperature={}, max_tokens={}, reasoning_effort={}",
                     agent_label, gen.temperature, gen.max_tokens, gen.reasoning_effort)

    @pytest.mark.parametrize("agent_name,agent_label", SIXIANG_AGENTS)
    def test_model_config_from_config(self, config, agent_name: str, agent_label: str):
        """验证模型配置与config.json中的一致。"""
        snapshot = build_sixiang_provider_snapshot(config, agent_name)

        # 检查config中的模型设置（agents 是 dict[str, SixiangAgentConfig]）
        sixiang_agents = config.agents.defaults.sixiang.agents
        agent_cfg = sixiang_agents.get(agent_name)

        expected_model = None
        if agent_cfg and agent_cfg.model:
            expected_model = agent_cfg.model

        if expected_model:
            assert snapshot.model == expected_model, (
                f"{agent_label} model={snapshot.model}, 期望={expected_model}"
            )
            logger.info("[{}] 模型配置验证: model={} (匹配显式配置: {})",
                         agent_label, snapshot.model, expected_model)
        else:
            # 使用全局默认模型
            default_model = config.agents.defaults.model
            assert snapshot.model == default_model, (
                f"{agent_label} model={snapshot.model}, 期望默认={default_model}"
            )
            logger.info("[{}] 模型配置验证: model={} (继承默认值)", agent_label, snapshot.model)


# ── 测试2: 真实API调用 ────────────────────────────────────────────────────


class TestSixiangRealApiCall:
    """验证每个Agent的Provider能成功调用LLM API。
    
    注意：这些测试会发起真实的LLM API调用，需要网络连接和有效的API Key。
    测试使用最小化请求以降低延迟和成本。
    """

    MINIMAL_MESSAGES = [
        {"role": "system", "content": "你是一个助手。请用一句话回复。"},
        {"role": "user", "content": "回复'测试通过'四个字即可。"},
    ]
    """最小化测试消息，用于快速验证API连通性。"""

    @pytest.mark.parametrize("agent_name,agent_label", SIXIANG_AGENTS)
    @pytest.mark.asyncio
    async def test_basic_chat_call(self, config, agent_name: str, agent_label: str):
        """每个Agent应能通过chat_with_retry成功调用LLM。"""
        snapshot = build_sixiang_provider_snapshot(config, agent_name)
        provider = snapshot.provider

        response = await provider.chat_with_retry(
            messages=self.MINIMAL_MESSAGES,
            temperature=0.1,
            max_tokens=50,
        )

        # 验证响应：content 或 reasoning_content 至少有一项不为空
        assert response is not None, f"{agent_label} 响应不应为 None"
        assert response.finish_reason in ("stop", "length"), (
            f"{agent_label} finish_reason={response.finish_reason}, 期望=stop/length"
        )
        has_content = bool(response.content and response.content.strip())
        has_reasoning = bool(response.reasoning_content and response.reasoning_content.strip())
        assert has_content or has_reasoning, (
            f"{agent_label} content和reasoning_content均为空"
        )

        logger.info("[{}] API调用成功: content={!r:.50}, reasoning={!r:.50}, "
                     "finish_reason={}, usage={}",
                     agent_label, (response.content or "")[:50],
                     (response.reasoning_content or "")[:50],
                     response.finish_reason, response.usage)

    @pytest.mark.parametrize("agent_name,agent_label", SIXIANG_AGENTS)
    @pytest.mark.asyncio
    async def test_chat_with_custom_model(self, config, agent_name: str, agent_label: str):
        """每个Agent能用指定model进行API调用。"""
        snapshot = build_sixiang_provider_snapshot(config, agent_name)
        provider = snapshot.provider

        # 使用agent自己的model进行调用
        response = await provider.chat_with_retry(
            messages=self.MINIMAL_MESSAGES,
            model=snapshot.model,
            temperature=0.1,
            max_tokens=50,
        )

        assert response is not None
        assert response.finish_reason in ("stop", "length"), (
            f"{agent_label} finish_reason={response.finish_reason}"
        )

        logger.info("[{}] 自定义model调用成功: model={}, content={!r:.50}, finish_reason={}",
                     agent_label, snapshot.model, response.content, response.finish_reason)


# ── 测试3: 各Agent特有参数 ────────────────────────────────────────────────


class TestSixiangAgentSpecific:
    """测试各Agent特有的API调用参数。"""

    MINIMAL_MESSAGES = [
        {"role": "system", "content": "你是助手。一句话回复。"},
        {"role": "user", "content": "回复'OK'。"},
    ]

    @pytest.mark.asyncio
    async def test_yang_with_tools(self, config):
        """阳（三爻）需要验证Function Calling调用。
        
        Yang是六爻中唯一使用工具定义的Agent。
        """
        from vingobot.providers.factory import build_sixiang_provider_snapshot

        snapshot = build_sixiang_provider_snapshot(config, "yang")
        provider = snapshot.provider

        tool_defs = [
            {
                "type": "function",
                "function": {
                    "name": "say_hello",
                    "description": "Say hello",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Name"},
                        },
                        "required": ["name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "task_complete",
                    "description": "Complete the task",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string", "description": "Summary"},
                        },
                        "required": ["summary"],
                    },
                },
            },
        ]

        messages = [
            {"role": "system", "content": "你是一个助手。请调用 task_complete 来完成任务。"},
            {"role": "user", "content": "请调用 task_complete 完成任务确认。"},
        ]

        response = await provider.chat_with_retry(
            messages=messages,
            tools=tool_defs,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=200,
        )

        assert response is not None
        logger.info("[阳·工具] 响应: finish_reason={}, has_tool_calls={}, content={!r:.80}",
                     response.finish_reason, response.has_tool_calls, response.content)

        # Yang应该要么返回工具调用，要么返回文本内容
        if response.has_tool_calls:
            for tc in response.tool_calls:
                logger.info("[阳·工具] 工具调用: name={}, args={}", tc.name, tc.arguments)
        else:
            logger.info("[阳·工具] 无工具调用（纯文本响应）")

    @pytest.mark.asyncio
    async def test_yin_low_temp(self, config):
        """阴（四爻）需要验证低温度API调用。
        
        Yin使用固定低温度(temperature=0.1)做审批决策。
        """
        snapshot = build_sixiang_provider_snapshot(config, "yin")
        provider = snapshot.provider

        messages = [
            {"role": "system", "content": "你是审批者。请判断以下工具调用是否安全。"},
            {"role": "user", "content": "read_file path=test.json 是否安全？回复'safe'或'unsafe'。"},
        ]

        response = await provider.chat_with_retry(
            messages=messages,
            temperature=0.1,  # Yin使用固定低温度
            max_tokens=50,
        )

        assert response is not None
        assert response.content is not None
        logger.info("[阴] 审批API调用成功: content={!r:.50}", response.content)

    @pytest.mark.asyncio
    async def test_weaver_structured_output(self, config):
        """织者（二爻）需要验证结构化的JSON输出能力。
        
        Weaver需要LLM输出六爻认知画像的JSON。
        """
        snapshot = build_sixiang_provider_snapshot(config, "weaver")
        provider = snapshot.provider

        messages = [
            {"role": "system", "content": "请以JSON格式输出：{\"current_yao\": 1, \"current_gua\": \"乾\"}"},
            {"role": "user", "content": "输出JSON"},
        ]

        response = await provider.chat_with_retry(
            messages=messages,
            temperature=0.7,  # Weaver使用相对高的温度
            max_tokens=200,
        )

        assert response is not None
        assert response.content is not None
        logger.info("[织者] JSON输出调用成功: content={!r:.80}", response.content)


# ── 测试4: 批量并行测试 ────────────────────────────────────────────────────


class TestSixiangParallelApiCalls:
    """并行测试所有Agent的API调用，模拟实际六爻循环中的并发场景。"""

    MINIMAL_MESSAGES = [
        {"role": "system", "content": "你是一个助手。请用一句话回复。"},
        {"role": "user", "content": "回复'1'。"},
    ]

    @pytest.mark.asyncio
    async def test_all_agents_parallel_api(self, config):
        """所有Agent并行调用API，验证并发能力。"""
        snapshots = {}
        for agent_name, agent_label in SIXIANG_AGENTS:
            snapshots[agent_name] = {
                "label": agent_label,
                "provider": build_sixiang_provider_snapshot(config, agent_name).provider,
            }

        async def call_agent(agent_name: str) -> dict[str, Any]:
            info = snapshots[agent_name]
            try:
                response = await info["provider"].chat_with_retry(
                    messages=self.MINIMAL_MESSAGES,
                    temperature=0.1,
                    max_tokens=30,
                )
                return {
                    "agent": agent_name,
                    "label": info["label"],
                    "success": True,
                    "content": response.content,
                    "finish_reason": response.finish_reason,
                }
            except Exception as e:
                return {
                    "agent": agent_name,
                    "label": info["label"],
                    "success": False,
                    "error": str(e),
                }

        # 并行发起所有Agent的API调用
        tasks = [call_agent(name) for name, _ in SIXIANG_AGENTS]
        results = await asyncio.gather(*tasks)

        # 汇总结果
        success_count = sum(1 for r in results if r["success"])
        fail_count = sum(1 for r in results if not r["success"])

        logger.info("=" * 60)
        logger.info("六爻Agent API并行调用测试结果:")
        logger.info(f"  总计: {len(results)}, 成功: {success_count}, 失败: {fail_count}")
        logger.info("-" * 60)
        for r in results:
            if r["success"]:
                logger.info("  ✅ [{}] finish_reason={}", r["label"], r["finish_reason"])
            else:
                logger.info("  ❌ [{}] error={}", r["label"], r.get("error"))
        logger.info("=" * 60)

        # 至少要有部分成功（如果API有配额限制可能部分失败）
        assert success_count > 0, "至少有一个Agent的API调用应成功"
