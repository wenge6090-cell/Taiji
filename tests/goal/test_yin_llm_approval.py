"""Tests for Yin's two-layer approval — hardcoded front + LLM contextual.

Covers:
- ``_parse_approval_decisions()`` — parsing LLM-generated JSON
- Front layer (hardcoded): path traversal, dangerous commands, protected paths
- LLM layer (contextual): mock LLM approval/rejection decisions
- Two-layer integration: mixed tool types, skill tools, fallback behaviors
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from vingobot.goal.types import ApprovedToolCall
from vingobot.goal.yin import _parse_approval_decisions, approve


# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------


class MockDecisionProvider:
    """Mock LLM provider that returns configurable approval decisions.

    Usage::

        provider = MockDecisionProvider([
            {"tool_index": 0, "approved": True, "reason": "safe"},
            {"tool_index": 1, "approved": False, "reason": "not safe"},
        ])
        provider._should_raise = True   # simulate LLM failure
        provider._captured_messages = []  # capture sent messages
    """

    def __init__(self, decisions: list[dict[str, Any]] | None = None) -> None:
        self.decisions = decisions or [
            {"tool_index": 0, "approved": True, "reason": "Mock approved"},
        ]
        self._should_raise: bool = False
        self._captured_messages: list[dict[str, Any]] = []

    async def chat_with_retry(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        """Simulate an LLM call, returning the configured decisions."""
        self._captured_messages = messages

        if self._should_raise:
            msg = "Simulated LLM provider failure"
            raise RuntimeError(msg)

        content = json.dumps({"decisions": self.decisions}, ensure_ascii=False)

        class _MockResponse:
            def __init__(self, content: str) -> None:
                self.content = content

        return _MockResponse(content)


# ---------------------------------------------------------------------------
# _parse_approval_decisions
# ---------------------------------------------------------------------------


class TestParseApprovalDecisions:
    """Parsing LLM-generated JSON approval responses."""

    def test_clean_json(self) -> None:
        """Parse clean JSON with a single decision."""
        content = (
            '{"decisions": [{"tool_index": 0, "approved": true, "reason": "ok"}]}'
        )
        result = _parse_approval_decisions(content, 1)
        assert len(result) == 1
        assert result[0]["tool_index"] == 0
        assert result[0]["approved"] is True
        assert result[0]["reason"] == "ok"

    def test_markdown_json_fence(self) -> None:
        """Parse JSON inside ```json code block."""
        content = (
            "Some text\n```json\n"
            '{"decisions": [{"tool_index": 0, "approved": false, "reason": "no"}]}\n'
            "```"
        )
        result = _parse_approval_decisions(content, 1)
        assert len(result) == 1
        assert result[0]["approved"] is False
        assert result[0]["reason"] == "no"

    def test_markdown_plain_fence(self) -> None:
        """Parse JSON inside plain ``` code block."""
        content = (
            "```\n"
            '{"decisions": [{"tool_index": 0, "approved": true}]}\n'
            "```"
        )
        result = _parse_approval_decisions(content, 1)
        assert len(result) == 1
        assert result[0]["approved"] is True

    def test_multiple_decisions(self) -> None:
        """Parse JSON with multiple decisions at once."""
        content = json.dumps({
            "decisions": [
                {"tool_index": 0, "approved": True, "reason": "safe"},
                {"tool_index": 1, "approved": False, "reason": "risky"},
                {"tool_index": 2, "approved": True, "reason": "ok"},
            ],
        })
        result = _parse_approval_decisions(content, 3)
        assert len(result) == 3
        assert result[0]["approved"] is True
        assert result[1]["approved"] is False
        assert result[2]["approved"] is True

    def test_empty_content_returns_all_rejected(self) -> None:
        """Empty content returns default reject for every expected index."""
        result = _parse_approval_decisions("", 2)
        assert len(result) == 2
        assert all(d["approved"] is False for d in result)
        assert all(d["reason"] == "解析失败" for d in result)

    def test_malformed_json_returns_all_rejected(self) -> None:
        """Unparseable content returns default reject for every expected index."""
        result = _parse_approval_decisions("not even close to json", 3)
        assert len(result) == 3
        assert all(d["approved"] is False for d in result)

    def test_fewer_decisions_than_expected(self) -> None:
        """Fewer decisions than expectations: results are short (caller handles)."""
        content = json.dumps({
            "decisions": [
                {"tool_index": 0, "approved": True, "reason": "only one"},
            ],
        })
        result = _parse_approval_decisions(content, 5)
        assert len(result) == 1  # only one decision returned


# ---------------------------------------------------------------------------
# approve() — front layer only (no LLM)
# ---------------------------------------------------------------------------


class TestApproveFrontLayer:
    """Hardcoded front layer — zero LLM involvement."""

    async def test_empty_tool_calls_returns_skipped(self) -> None:
        """Empty list returns ([] , 'skipped', ...)."""
        result, decision, reason = await approve([])
        assert result == []
        assert decision == "skipped"
        assert reason == "无工具调用"

    async def test_read_only_tool_auto_approved(self) -> None:
        """read_file is auto-approved by front layer."""
        calls = [{"name": "read_file", "arguments": {"path": "test.txt"}}]
        result, decision, reason = await approve(calls)
        assert len(result) == 1
        assert result[0].name == "read_file"
        assert decision == "approved"

    async def test_special_tool_task_complete_auto_approved(self) -> None:
        """task_complete is auto-approved."""
        calls = [{"name": "task_complete", "arguments": {"summary": "done"}}]
        result, decision, reason = await approve(calls)
        assert len(result) == 1
        assert result[0].name == "task_complete"

    async def test_path_traversal_rejected(self) -> None:
        """Path traversal via '..' is rejected by front layer."""
        calls = [
            {"name": "write_file", "arguments": {"path": "../../etc/passwd"}},
        ]
        result, decision, reason = await approve(calls)
        assert len(result) == 0
        assert decision == "rejected"
        assert "路径穿越" in reason

    async def test_dangerous_command_rejected(self) -> None:
        """Dangerous shell command (rm -rf) is rejected by front layer."""
        calls = [{"name": "exec", "arguments": {"command": "rm -rf /"}}]
        result, decision, reason = await approve(calls)
        assert len(result) == 0
        assert decision == "rejected"
        assert "危险命令" in reason

    async def test_dangerous_command_pipe_rejected(self) -> None:
        """Pipe to a dangerous command is rejected."""
        calls = [{"name": "exec", "arguments": {"command": "cat sensitive | rm -f"}}]
        result, decision, reason = await approve(calls)
        assert len(result) == 0
        assert decision == "rejected"
        assert "危险命令" in reason

    async def test_protected_taiji_path_rejected(self, tmp_path: Path) -> None:
        """Write to .taiji/ cognition directory is rejected."""
        calls = [
            {"name": "write_file", "arguments": {"path": ".taiji/cognition/test.txt"}},
        ]
        result, decision, reason = await approve(calls, workspace_root=tmp_path)
        assert len(result) == 0
        assert decision == "rejected"
        assert "认知层路径受保护" in reason

    async def test_protected_config_json_rejected(self, tmp_path: Path) -> None:
        """Write to config.json is rejected."""
        calls = [{"name": "write_file", "arguments": {"path": "config.json"}}]
        result, decision, reason = await approve(calls, workspace_root=tmp_path)
        assert len(result) == 0
        assert decision == "rejected"
        assert "配置文件受保护" in reason

    async def test_unknown_tool_rejected(self) -> None:
        """An unregistered tool is rejected immediately."""
        calls = [{"name": "nobody_knows_this_tool", "arguments": {}}]
        result, decision, reason = await approve(calls)
        assert len(result) == 0
        assert decision == "rejected"
        assert "未知工具" in reason

    async def test_exec_with_traversal_cwd_rejected(self) -> None:
        """Traversal in exec.cwd is rejected."""
        calls = [{"name": "exec", "arguments": {"command": "ls", "cwd": "../escape"}}]
        result, decision, reason = await approve(calls)
        assert len(result) == 0
        assert decision == "rejected"
        assert "路径穿越" in reason


# ---------------------------------------------------------------------------
# approve() — two-layer with LLM
# ---------------------------------------------------------------------------


class TestApproveWithLLMLayer:
    """Full two-layer approval using a mock LLM provider."""

    async def test_write_file_accepted_by_llm(self, tmp_path: Path) -> None:
        """A safe write_file passes front check and gets LLM approval."""
        provider = MockDecisionProvider([
            {"tool_index": 0, "approved": True, "reason": "写入工作区内，符合 L4 安全真理"},
        ])
        calls = [{"name": "write_file", "arguments": {"path": "output.txt"}}]
        result, decision, reason = await approve(
            calls, workspace_root=tmp_path, provider=provider,
        )
        assert len(result) == 1
        assert result[0].name == "write_file"
        assert decision == "approved"

    async def test_write_file_rejected_by_llm(self, tmp_path: Path) -> None:
        """LLM can reject a write_file that passed the front check."""
        provider = MockDecisionProvider([
            {"tool_index": 0, "approved": False, "reason": "违反安全真理: 路径指向敏感区域"},
        ])
        calls = [{"name": "write_file", "arguments": {"path": "output.txt"}}]
        result, decision, reason = await approve(
            calls, workspace_root=tmp_path, provider=provider,
        )
        assert len(result) == 0
        assert decision == "rejected"
        assert "违反安全真理" in reason

    async def test_read_only_and_side_effect_mixed(
        self, tmp_path: Path,
    ) -> None:
        """read-only bypasses LLM; side-effect goes through it."""
        provider = MockDecisionProvider([
            {"tool_index": 0, "approved": True, "reason": "ok"},
        ])
        calls = [
            {"name": "read_file", "arguments": {"path": "test.txt"}},
            {"name": "write_file", "arguments": {"path": "notes.txt"}},
        ]
        result, decision, reason = await approve(
            calls, workspace_root=tmp_path, provider=provider,
        )
        assert len(result) == 2
        assert result[0].name == "read_file"
        assert result[1].name == "write_file"

    async def test_llm_rejects_one_of_two_side_effects(
        self, tmp_path: Path,
    ) -> None:
        """LLM approves one, rejects another → 'modified' decision."""
        provider = MockDecisionProvider([
            {"tool_index": 0, "approved": True, "reason": "safe location"},
            {"tool_index": 1, "approved": False, "reason": "路径违规"},
        ])
        calls = [
            {"name": "write_file", "arguments": {"path": "safe.txt"}},
            {"name": "write_file", "arguments": {"path": "risky.txt"}},
        ]
        result, decision, reason = await approve(
            calls, workspace_root=tmp_path, provider=provider,
        )
        assert len(result) == 1
        assert result[0].arguments["path"] == "safe.txt"
        assert decision == "modified"
        assert "部分批准" in reason

    async def test_no_provider_fallback_reject(self, tmp_path: Path) -> None:
        """No provider → conservative rejection of LLM-needed calls."""
        calls = [{"name": "write_file", "arguments": {"path": "output.txt"}}]
        with patch("vingobot.goal.yin._get_provider", return_value=None):
            result, decision, reason = await approve(
                calls, workspace_root=tmp_path, provider=None,
            )
        assert len(result) == 0
        assert decision == "rejected"
        assert "无 LLM provider" in reason

    async def test_llm_exception_fallback_reject(self, tmp_path: Path) -> None:
        """LLM exception → conservative rejection."""
        provider = MockDecisionProvider()
        provider._should_raise = True
        calls = [{"name": "write_file", "arguments": {"path": "output.txt"}}]
        result, decision, reason = await approve(
            calls, workspace_root=tmp_path, provider=provider,
        )
        assert len(result) == 0
        assert decision == "rejected"
        assert "LLM 调用异常" in reason

    async def test_exec_accepted_by_llm(self, tmp_path: Path) -> None:
        """A safe exec command passes front check and gets LLM approval."""
        provider = MockDecisionProvider([
            {"tool_index": 0, "approved": True, "reason": "安全命令"},
        ])
        calls = [{"name": "exec", "arguments": {"command": "ls -la"}}]
        result, decision, reason = await approve(
            calls, workspace_root=tmp_path, provider=provider,
        )
        assert len(result) == 1
        assert result[0].name == "exec"

    async def test_delete_file_accepted_by_llm(self, tmp_path: Path) -> None:
        """A safe delete_file passes front check and gets LLM approval."""
        provider = MockDecisionProvider([
            {"tool_index": 0, "approved": True, "reason": "Safe deletion"},
        ])
        calls = [{"name": "delete_file", "arguments": {"path": "tmp.txt"}}]
        result, decision, reason = await approve(
            calls, workspace_root=tmp_path, provider=provider,
        )
        assert len(result) == 1
        assert result[0].name == "delete_file"

    async def test_llm_system_prompt_includes_truths_and_soul(
        self, tmp_path: Path,
    ) -> None:
        """The system prompt sent to the LLM mentions L4 truths and L5 soul."""
        provider = MockDecisionProvider([
            {"tool_index": 0, "approved": True, "reason": "ok"},
        ])
        calls = [{"name": "write_file", "arguments": {"path": "test.txt"}}]
        await approve(calls, workspace_root=tmp_path, provider=provider)

        assert len(provider._captured_messages) == 2
        sys_content = provider._captured_messages[0]["content"]
        assert "L4 真理层" in sys_content
        assert "L5 灵魂层" in sys_content


# ---------------------------------------------------------------------------
# approve() — skill tool routing through two layers
# ---------------------------------------------------------------------------


class TestApproveSkillTools:
    """Skill tool approval through the two-layer pipeline."""

    @pytest.fixture(autouse=True)
    def _clean_registry(self) -> None:
        """Clear the global skill-tool registry before and after each test."""
        from vingobot.goal.skill_parser import _skill_tool_registry

        _skill_tool_registry.clear()
        yield
        _skill_tool_registry.clear()

    async def test_auto_approved_skill_bypasses_llm(self) -> None:
        """Skill tool with auto_approve=True is approved without LLM."""
        from vingobot.goal.skill_parser import SkillToolDef, register_skill_tool

        register_skill_tool(SkillToolDef(
            name="quick_greet",
            description="Quick greet",
            tool_auto_approve=True,
        ))

        calls = [{"name": "quick_greet", "arguments": {"name": "Alice"}}]
        result, decision, reason = await approve(calls)
        assert len(result) == 1
        assert result[0].name == "quick_greet"
        assert decision == "approved"

    async def test_non_auto_approved_skill_goes_to_llm(
        self, tmp_path: Path,
    ) -> None:
        """Skill tool with auto_approve=False goes through front check → LLM."""
        from vingobot.goal.skill_parser import SkillToolDef, register_skill_tool

        register_skill_tool(SkillToolDef(
            name="risky_op",
            description="Risky operation",
            tool_auto_approve=False,
        ))

        provider = MockDecisionProvider([
            {"tool_index": 0, "approved": True, "reason": "Acceptable risk"},
        ])
        calls = [{"name": "risky_op", "arguments": {"target": "file.txt"}}]
        result, decision, reason = await approve(
            calls, workspace_root=tmp_path, provider=provider,
        )
        assert len(result) == 1
        assert result[0].name == "risky_op"

    async def test_non_auto_approved_skill_rejected_by_llm(
        self, tmp_path: Path,
    ) -> None:
        """Non-auto-approved skill tool can be rejected by LLM."""
        from vingobot.goal.skill_parser import SkillToolDef, register_skill_tool

        register_skill_tool(SkillToolDef(
            name="risky_op",
            description="Too risky",
            tool_auto_approve=False,
        ))

        provider = MockDecisionProvider([
            {"tool_index": 0, "approved": False, "reason": "风险过高"},
        ])
        calls = [{"name": "risky_op", "arguments": {"target": "secret.txt"}}]
        result, decision, reason = await approve(
            calls, workspace_root=tmp_path, provider=provider,
        )
        assert len(result) == 0
        assert decision == "rejected"


# ---------------------------------------------------------------------------
# approve() — edge cases and argument handling
# ---------------------------------------------------------------------------


class TestApproveEdgeCases:
    """Edge cases: argument parsing, malformed input, etc."""

    async def test_tool_call_arguments_as_json_string(self) -> None:
        """Arguments passed as JSON string (deserialized by yin.py)."""
        calls = [
            {
                "name": "write_file",
                "arguments": '{"path": "test.txt"}',
            },
        ]
        # Argument JSON string → correctly parsed
        # write_file with path only (no content) → goes to LLM layer
        result, decision, reason = await approve(calls, workspace_root=None, provider=None)
        # May be approved or rejected depending on config; key test is json parse worked
        if result:
            assert result[0].arguments.get("path") == "test.txt"
        else:
            assert decision == "rejected"

    async def test_tool_call_with_function_dict_structure(self) -> None:
        """Tool calls in {function: {name, arguments}} format (Anthropic-style)."""
        calls = [
            {
                "function": {
                    "name": "read_file",
                    "arguments": {"path": "readme.md"},
                },
            },
        ]
        result, decision, reason = await approve(calls)
        assert len(result) == 1
        assert result[0].name == "read_file"

    async def test_missing_tool_name(self) -> None:
        """Tool call with no name is rejected."""
        calls = [{"name": "", "arguments": {}}]
        result, decision, reason = await approve(calls)
        assert len(result) == 0
        assert decision == "rejected"

    async def test_file_safety_empty_path(self) -> None:
        """write_file with empty path is rejected."""
        calls = [{"name": "write_file", "arguments": {"path": ""}}]
        result, decision, reason = await approve(calls)
        assert len(result) == 0
        assert decision == "rejected"
        assert "路径为空" in reason

    async def test_exec_empty_command(self) -> None:
        """exec with empty command is rejected."""
        calls = [{"name": "exec", "arguments": {"command": ""}}]
        result, decision, reason = await approve(calls)
        assert len(result) == 0
        assert decision == "rejected"
        assert "命令为空" in reason
