"""
二爻·编织器 — Dynamic orchestration layer.

The Weaver is responsible for preparing everything Yang needs for a
single round of the inner loop:

1. **Cognitive Profile** — Weaver LLM determines the cognitive posture
   (yao, sixiang, gua, temperature, etc.) by reading meta-cognitive grids
   and execution history.
2. **System prompt** — Built from eternal context (with cognitive posture
   injected), goal context, and round facts.
3. **Tool definitions** — Base tools (trigram-filtered) + L3 grid-discovered
   skill tools.
4. **Loop detection** — Monitors for repeated identical tool calls across
   rounds and flags potential infinite loops.

L3 Grid integration:
- ``_discover_skill_tools()`` reads the trigram's grid JSON and loads
  tool definitions from referenced L1 skills (their ``SKILL.md``).
  L2 models are handled by 思变 (Sibian), not by Weaver.
- Workflow steps from the grid are injected into the system prompt as
  execution guidance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from loguru import logger

from vingobot.goal.grid_types import (
    BaguaGrid,
    LiuyaoGrid,
    SixiangGrid,
    parse_bagua_grid,
    parse_liuyao_grid,
    parse_sixiang_grid,
)
from vingobot.goal.types import (
    CognitiveProfile,
    MetaCognitionState,
    MingjueOutput,
    RoundExecutionFact,
    WeaverOutput,
)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _human_size(size: int) -> str:
    """Format a byte count into a human-readable string."""
    if size < 1024:
        return f"{size} B"
    for unit in ("KB", "MB", "GB"):
        size /= 1024.0
        if size < 1024:
            return f"{size:.1f} {unit}"
    return f"{size:.1f} TB"


# ---------------------------------------------------------------------------
# Trigram → tool set mapping (base tools only — skill tools are discovered)
# ---------------------------------------------------------------------------

_TRIGRAM_TOOLS: dict[str, list[str]] = {
    "qian": [
        "read_file",
        "list_directory",
        "write_file",
        "edit_file",
        "web_search",
        "web_fetch",
        "query_capabilities",
        "task_complete",
    ],
    "kun": [
        "read_file",
        "list_directory",
        "write_file",
        "edit_file",
        "exec",
        "web_search",
        "web_fetch",
        "query_capabilities",
        "task_complete",
    ],
    "zhen": [
        "read_file",
        "list_directory",
        "write_file",
        "edit_file",
        "exec",
        "web_search",
        "web_fetch",
        "query_capabilities",
        "task_complete",
    ],
    "xun": [
        "read_file",
        "list_directory",
        "write_file",
        "edit_file",
        "web_search",
        "web_fetch",
        "query_capabilities",
        "task_complete",
    ],
    "kan": [
        "read_file",
        "list_directory",
        "write_file",
        "edit_file",
        "exec",
        "web_search",
        "web_fetch",
        "query_capabilities",
        "task_complete",
    ],
    "li": [
        "read_file",
        "list_directory",
        "write_file",
        "edit_file",
        "exec",
        "web_search",
        "web_fetch",
        "query_capabilities",
        "task_complete",
    ],
    "gen": [
        "read_file",
        "list_directory",
        "write_file",
        "edit_file",
        "query_capabilities",
        "task_complete",
    ],
    "dui": [
        "read_file",
        "list_directory",
        "write_file",
        "edit_file",
        "exec",
        "web_search",
        "web_fetch",
        "query_capabilities",
        "task_complete",
    ],
}

# Base tool definitions (minimal — full definitions fetched from registry at runtime)
_BASE_TOOL_DEFS: dict[str, dict[str, Any]] = {
    "read_file": {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative or absolute file path"},
                    "start_line": {
                        "type": "integer",
                        "description": "Optional start line (1-based)",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Optional end line (1-based, inclusive)",
                    },
                },
                "required": ["path"],
            },
        },
    },
    "list_directory": {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to list"},
                },
                "required": ["path"],
            },
        },
    },
    "write_file": {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, creating it if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write to"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    "edit_file": {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit a file by replacing old_string with new_string. "
            "Surgical alternative to write_file — read the target file, "
            "find-and-replace the first occurrence of old_string, and "
            "write back. Use this for targeted changes instead of "
            "overwriting the whole file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to edit"},
                    "old_string": {"type": "string", "description": "Text to replace"},
                    "new_string": {"type": "string", "description": "Replacement text"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    "exec": {
        "type": "function",
        "function": {
            "name": "exec",
            "description": "Execute a shell command and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "cwd": {"type": "string", "description": "Optional working directory"},
                },
                "required": ["command"],
            },
        },
    },
    "web_search": {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for up-to-date information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    },
    "web_fetch": {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch and read the content of a web page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                },
                "required": ["url"],
            },
        },
    },
    "query_capabilities": {
        "type": "function",
        "function": {
            "name": "query_capabilities",
            "description": "Query the current execution environment capabilities: "
            "concurrency limits, read/write permissions, context budget, "
            "cross-round features, etc. Use this when unsure whether you "
            "can safely perform an action or how many resources you have.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    "task_complete": {
        "type": "function",
        "function": {
            "name": "task_complete",
            "description": "Mark the current task as complete. Call this when you have "
            "fully achieved the task goal. Provide a summary and structured results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "One-line summary of what was accomplished",
                    },
                    "concrete_goal": {
                        "type": "string",
                        "description": "Detailed task description including expected outcomes and constraints (optional, used by task decomposition)",
                    },
                    "trigram": {
                        "type": "string",
                        "enum": ["qian", "kun", "zhen", "xun", "kan", "li", "gen", "dui"],
                        "description": "Bagua trigram for the task nature (optional, used by task decomposition)",
                    },
                    "trigram_reason": {
                        "type": "string",
                        "description": "Reason for choosing this trigram (optional)",
                    },
                    "goal_progress_pct": {
                        "type": "integer",
                        "description": "Goal completion percentage 0-100 (optional)",
                        "minimum": 0,
                        "maximum": 100,
                    },
                },
                "required": ["summary"],
            },
        },
    },
    "search_codebase": {
        "type": "function",
        "function": {
            "name": "search_codebase",
            "description": "Semantic search across the codebase for relevant code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query"},
                },
                "required": ["query"],
            },
        },
    },
}

# ---------------------------------------------------------------------------
# L3 Grid discovery cache
# ---------------------------------------------------------------------------

_grid_discovery_cache: dict[str, tuple[list[dict[str, Any]], str, list[str]]] = {}


def _discover_skill_tools(trigram: str) -> tuple[list[dict[str, Any]], str, list[str]]:
    """Discover L1 skill tools from the trigram's L3 grid.

    Reads the grid JSON at ``<workspace>/.taiji/cognition/grids/<trigram相关的grid>.json``,
    resolves referenced skills (tool definitions) only.
    L2 models are handled by 思变 (Sibian), not by Weaver.

    Returns:
        (tool_definitions, workflow_guidance, skill_names)
    """
    # Check cache first (per trigram, per session)
    cached = _grid_discovery_cache.get(trigram)
    if cached is not None:
        return cached  # type: ignore[return-value]

    discovered_tools: list[dict[str, Any]] = []
    workflow_guidance = ""
    skill_names: list[str] = []

    try:
        from vingobot.core.workspace import get_workspace_paths
        from vingobot.goal.cognition_tools import parse_grid

        wp = get_workspace_paths()
        grids_dir = wp.grids

        if not grids_dir.is_dir():
            return [], "", []

        # Try to find grid files matching this trigram
        grid_files = list(grids_dir.glob("*.json"))
        trigram_grid = None

        for gf in grid_files:
            try:
                grid = parse_grid(gf)
                if grid and grid.trigram == trigram:
                    trigram_grid = grid
                    break
            except Exception:
                continue

        if trigram_grid is None:
            _grid_discovery_cache[trigram] = ([], "", [])
            return [], "", []

        # Build workflow guidance
        if trigram_grid.workflow:
            steps = []
            for ws in trigram_grid.workflow:
                skill_hint = f" [skills: {', '.join(ws.skills)}]" if ws.skills else ""
                steps.append(f"{ws.step}. {ws.description}{skill_hint}")
            workflow_guidance = f"\n## 推荐工作流 ({trigram_grid.domain})\n" + "\n".join(steps)

        # Discover skill-based tools from grid's skill references using skill_parser
        skills_dir = wp.skills
        for skill_ref in trigram_grid.skills:
            skill_name = skill_ref.name
            skill_dir = skills_dir / skill_name

            if not skill_dir.is_dir():
                logger.debug("[编织] L3网格引用的技能 '{}' 不存在", skill_name)
                continue

            tool_defs = _load_skill_tools(skill_name, skill_dir)
            discovered_tools.extend(tool_defs)
            skill_names.append(skill_name)

    except Exception as exc:
        logger.warning("[编织] 发现L3格栅技能工具失败: {}", exc)

    result = (discovered_tools, workflow_guidance, skill_names)
    _grid_discovery_cache[trigram] = result
    return result


def _load_skill_tools(
    skill_name: str,
    skill_dir: Any,
) -> list[dict[str, Any]]:
    """Load tool definitions from a skill's SKILL.md using skill_parser.

    Parses the YAML frontmatter, converts each tool to OpenAI schema,
    registers them in the global skill tool registry for Executor routing,
    and loads the skill's ``_executor.py`` if present.
    """
    from vingobot.goal.skill_parser import parse_skill_md, register_skill_tools_from_meta

    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        return []

    # ── Load the skill's executor module (auto-registers on import) ──
    _try_load_skill_executor(skill_dir)

    meta = parse_skill_md(skill_md)
    if meta is None:
        return []

    # Register tools so Executor can route calls
    register_skill_tools_from_meta(meta)

    # Convert each tool to OpenAI schema
    return [t.to_openai_tool_def() for t in meta.tools]


def _try_load_skill_executor(skill_dir: Any) -> None:
    """Attempt to import a skill's ``_executor`` module for auto-registration.

    Skills can ship an ``_executor.py`` file that auto-registers async
    executor callables on import via ``register_skill_executor()``.
    This function triggers that import so the executors are available
    when Yang calls the skill's tools.
    """
    executor_file = Path(str(skill_dir)) / "_executor.py"
    if not executor_file.is_file():
        return

    # Derive the dotted module path from the file path
    # e.g. .../skills/remotion-video/_executor.py → _executor
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            f"vingobot_skill_{executor_file.parent.name}",
            str(executor_file),
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            logger.debug("[编织] 已加载技能执行器: {}", executor_file.parent.name)
    except Exception as exc:
        logger.debug("[编织] 加载技能执行器失败 {}: {}", executor_file.parent.name, exc)


def clear_grid_discovery_cache() -> None:
    """Clear the L3 grid discovery cache (e.g. after a grid is updated)."""
    _grid_discovery_cache.clear()


def invalidate_cognition_caches() -> None:
    """Invalidate all cognition caches.

    Called after DMN cognitive evolution tasks complete (new skills,
    models, or grids created) so that the next Weaver round discovers
    the updated assets.
    """
    _grid_discovery_cache.clear()
    logger.debug("[编织] 已刷新认知缓存")


# ---------------------------------------------------------------------------
# Loop detection constants
# ---------------------------------------------------------------------------

_MAX_IDENTICAL_ROUNDS = 4  # trigger warning after this many identical calls


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def weave(
    mingjue: MingjueOutput,
    facts: list[RoundExecutionFact],
    goal_context: Any,
    round_num: int = 1,
    previous_invoke_results: str = "",
    read_only_round_count: int = 0,
    had_successful_write: bool = False,
) -> WeaverOutput:
    """Weave context, tools, and cognitive profile for one round.

    Note: previous_yang_content injection is now handled by task_inner_loop
    directly before calling run_yang — it is not part of Weaver's responsibility.
    """
    trigram = mingjue.trigram or "kun"
    tool_names = _TRIGRAM_TOOLS.get(trigram, _TRIGRAM_TOOLS["kun"])

    # 0. Load meta-cognitive grids
    sixiang_grid, liuyao_grid, bagua_grid = _load_meta_grids()

    # 1. Build current MetaCognitionState from facts (or defaults)
    state = _build_state_from_facts(facts, mingjue)

    # 2. Get cognitive profile from Weaver LLM
    profile = await _decide_cognitive_profile(
        mingjue, state, facts, round_num,
        sixiang_grid, liuyao_grid, bagua_grid,
    )

    # 3. Build eternal context using CognitiveProfile (for Yang)
    eternal = _build_eternal_context(
        mingjue, profile, liuyao_grid, bagua_grid, sixiang_grid,
        round_num,
    )

    # 4. Discover L3 grid skill tools (L1 only; L2 models are handled by 思变)
    skill_tools, workflow_guidance, discovered_skill_names = _discover_skill_tools(trigram)

    # 5. Build full system prompt
    system_prompt = _build_system_prompt_v2(
        eternal, mingjue, goal_context, facts, round_num,
        workflow_guidance, previous_invoke_results,
    )

    # 6. Build tool definitions
    tool_defs = [_BASE_TOOL_DEFS[name] for name in tool_names if name in _BASE_TOOL_DEFS]
    tool_defs.extend(skill_tools)

    # 7. Inject cognitive posture info (informational, not imperative)
    yao_keys = ["初爻", "二爻", "三爻", "四爻", "五爻", "上爻"]
    yao_name = yao_keys[profile.current_yao - 1] if 1 <= profile.current_yao <= 6 else "初爻"
    system_prompt += (
        f"\n\n## 本轮认知姿态\n"
        f"- 爻位: {yao_name}({profile.yao_reasoning})\n"
        f"- 卦象: {profile.current_gua}卦({profile.gua_reasoning})\n"
        f"- 四象: {profile.sixiang_selected}({profile.sixiang_reasoning})\n"
        f"- 参数: T={profile.temperature}, top_p={profile.top_p}, top_k={profile.top_k}, rep_pen={profile.repetition_penalty}\n"
        f"\n（以上为 Weaver 认知决策引擎的输出，供你参考当前任务阶段的执行姿态，不构成强制指令。）"
    )

    # 7a. Cognitive asset path guidance (mandatory skill usage)
    trigram_label = mingjue.trigram or "kun"
    if discovered_skill_names:
        skill_list = "\n".join(
            f"  - **{name}**: `read_file .vingobot/.taiji/cognition/skills/{name}/SKILL.md` 查看技能定义"
            for name in discovered_skill_names
        )
        system_prompt += (
            f"\n\n## 认知资产指引（强制性——认知库优先）\n"
            f"当前卦象 **{trigram_label}卦** 已注入以下技能工具（可直接调用）：\n"
            f"{skill_list}\n\n"
            f"**优先级规则（不可跳过）**：\n"
            f"1. ⭐ 这些技能是本卦默认方案——你的**首轮必须先 read_file 以上技能的 SKILL.md**，了解已有模板/脚本/流程\n"
            f"2. 如果技能提供了现成模板（如 remotion_init_project），**必须调用技能工具**，禁止自己从零搭建（npm init / pip install / npx create）\n"
            f"3. 如果技能不适用 → 在 execution-facts 中写明理由（`skill_bypass_reason: ...`），然后走自己的方案\n"
            f"4. 技能工具调用失败 → 查看错误输出后用不同参数重试一次，不要放弃改用 ad-hoc 方案\n"
        )
    else:
        system_prompt += (
            f"\n\n## 认知资产指引（强制性）\n"
            f"当前卦象 {trigram_label}卦 无匹配的预注入技能。"
            f"检查工具列表，如存在以 `skill_` 或 `remotion_` 前缀开头的工具，说明 Weaver 已注入可使用。\n"
            f"## 认知库路径\n"
            f"- L3 网格: `list_directory` 浏览 grids/ 目录，`read_file` 读取匹配 {trigram_label} 卦的 JSON 网格文件获取完整 workflow\n"
            f"- L2 模型: `list_directory` 浏览 models/ 目录查看可用经验模型，`read_file` 读取 .json 或 .md 文件\n"
            f"- L1 技能: `list_directory` 浏览 skills/ 目录查看已有技能，`read_file` 读取 SKILL.md\n"
        )

    # 8. Loop detection
    loop_warning = _detect_loop(facts)
    if loop_warning:
        system_prompt += f"\n\n## ⚠️ 死循环检测\n{loop_warning}"

    # 8a. Termination stats (informational, not imperative)
    term_directive = _build_termination_directive(round_num, read_only_round_count, had_successful_write)
    if term_directive:
        system_prompt += f"\n\n{term_directive}"

    # 9. Use discovered skill names directly (from grid, not tool-name inference)
    grid_skills = list(discovered_skill_names)

    return WeaverOutput(
        system_prompt=system_prompt,
        tool_definitions=tool_defs,
        cognitive_profile=profile,
        grid_domain=_get_discovered_grid_domain(trigram),
        grid_skills=grid_skills,
    )



def _build_system_prompt_v2(
    eternal: str,
    mingjue: MingjueOutput,
    goal_context: Any,
    facts: list[RoundExecutionFact],
    round_num: int,
    workflow_guidance: str = "",
    previous_invoke_results: str = "",
) -> str:
    """Build the system prompt from pre-computed eternal context.

    Note: Yang-side injections (previous_yang_content, terminate directives,
    invoke results window, execution history path reference) are now handled
    by task_inner_loop, not by Weaver.
    """
    parts = [eternal]
    l2 = _layer_goal_context(mingjue, goal_context, workflow_guidance)
    if l2:
        parts.append(l2)
    if previous_invoke_results:
        parts.append(f"## 上一轮工具执行结果\n{previous_invoke_results[:15000]}")
    l4 = _layer_round_facts(facts, round_num)
    if l4:
        parts.append(l4)

    # ── Exec failure recovery guidance ───────────────────────
    exec_failures = sum(
        1 for f in facts
        if getattr(f, "execution_status", "") == "exec_failed"
        and not getattr(f, "is_verification_round", False)
    )
    if exec_failures >= 2:
        parts.append(
            f"## ⚠️ exec 连续失败 ({exec_failures} 次)\n"
            f"检测到连续 exec 失败（超时或非零退出码）。"
            f"**禁止再次执行同一脚本！**\n"
            f"降级方案：用 write_file 写入手工分析报告/文档作为替代交付物。\n"
            f"如需修复脚本，用 edit_file 一次性修复后用 task_complete 提交结果。\n"
            f"**不要**用 read_file 读取自己的执行事实文件来试图\"理解错误\"——直接行动。"
        )
    elif exec_failures == 1:
        parts.append(
            f"## ⚠️ exec 执行失败\n"
            f"上一轮 exec 执行失败。\n"
            f"1. 若脚本有 bug，用 edit_file 修复后重试一次\n"
            f"2. 若无法修复，降级为 write_file 手工产出\n"
            f"3. 禁止反复读取执行事实文件——直接行动，不要观察。"
        )

    # ── Failable operation enforcement ─────────────────────────
    _FAILABLE_THRESHOLD = 3
    non_failable_streak = 0
    for f in reversed(facts):
        if not getattr(f, "had_failable_op", False):
            non_failable_streak += 1
        else:
            break
    if non_failable_streak >= _FAILABLE_THRESHOLD:
        if exec_failures >= 2:
            action_items = (
                f"1. 用 write_file 写入产出，然后立刻 read_file 验证内容正确\n"
                f"2. 用 web_search 获取外部信息（多源交叉验证）\n"
            )
        else:
            action_items = (
                f"1. 用 write_file 写入产出，然后立刻 read_file 验证内容正确\n"
                f"2. 用 exec 执行脚本，观察真实运行结果而非继续阅读\n"
                f"3. 用 web_search 获取外部信息（多源交叉验证）\n"
            )
        parts.append(
            f"## ⚠️ 连续 {non_failable_streak} 轮无验证性操作\n"
            f"你已连续 {non_failable_streak} 轮只做 read_file/list_directory，"
            f"未执行任何**可失败操作**（exec / web_search / web_fetch / write_file / edit_file）。\n\n"
            f"**本轮强制要求**：必须执行以下至少一项之后才能继续读取文件：\n"
            f"{action_items}\n"
            f"**禁止**继续读自己的执行事实文件或已读过的产出文件——你已经有足够信息，"
            f"现在需要行动起来，用不可预知的真实结果来校准你的理解。"
        )

    return "\n\n---\n\n".join(p for p in parts if p)



def _layer_goal_context(
    mingjue: MingjueOutput,
    goal_context: Any,
    workflow_guidance: str = "",
) -> str:
    """Layer 2: Goal-level context, navigation hints, and grid workflow."""
    parts: list[str] = []

    # Task description
    trigram_label = mingjue.trigram or "kun"
    parts.append(f"## 当前任务\n{mingjue.concrete_goal or mingjue.summary}")
    parts.append(f"\n八卦卦象: {trigram_label} — {mingjue.trigram_reason}")

    # Goal metadata
    if goal_context is not None:
        try:
            bp = getattr(goal_context, "blueprint_summary", "") or ""
            if bp:
                parts.append(f"\n## 目标蓝图\n{bp[:2000]}")
            mem = getattr(goal_context, "memory_summary", "") or ""
            if mem:
                parts.append(f"\n## 目标记忆\n{mem[:800]}")

            # Goal status & priority from meta
            meta = getattr(goal_context, "meta", None)
            if meta is not None:
                status = getattr(meta, "status", "")
                priority = getattr(meta, "priority", "")
                self_driven = getattr(meta, "self_driven", False)
                meta_parts = []
                if status:
                    meta_parts.append(f"状态: {status}")
                if priority:
                    meta_parts.append(f"优先级: {priority}/10")
                if self_driven:
                    meta_parts.append("自驱执行: 是")
                if meta_parts:
                    parts.append(f"\n## 目标状态\n{' | '.join(meta_parts)}")

            # ── Known traps (per-goal anti-pattern table) ──
            known_traps = getattr(goal_context, "known_traps_text", "") or ""
            if known_traps:
                parts.append(f"\n{known_traps}")

            # Trajectory snapshot
            ts = getattr(goal_context, "trajectory_snapshot", "") or ""
            if ts:
                parts.append(f"\n## 目标轨迹\n{ts}")

            # Recent task statuses
            recent = getattr(goal_context, "recent_task_statuses", None) or []
            if recent:
                recent_lines = []
                for t in recent[:5]:
                    recent_lines.append(f"- {t.task_id}: {t.status} | {t.summary_snippet[:200]}")
                parts.append("\n## 近期任务\n" + "\n".join(recent_lines))

        except Exception:
            pass

    # Goal directory file listing (read-only reference)
    goal_dir_path = mingjue.context.goal_dir if mingjue.context else ""
    if goal_dir_path:
        try:
            gd = Path(goal_dir_path)
            if gd.is_dir():
                available = []
                for f in sorted(gd.iterdir()):
                    if f.is_file():
                        try:
                            size = f.stat().st_size
                            available.append(f"- {f.name} ({size} B)")
                        except OSError:
                            available.append(f"- {f.name}")
                    elif f.is_dir():
                        available.append(f"- {f.name}/ (目录)")
                if available:
                    parts.append("\n## 目标目录文件清单（可只读访问）")
                    parts.append("以下文件位于目标目录，你可以使用 read_file 只读访问它们：")
                    parts.extend(available)

            # Phase 1 report
            phase1 = gd / "phase1-report.md" if gd.is_dir() else None
            if phase1 and phase1.is_file():
                try:
                    p1_text = phase1.read_text(encoding="utf-8")[:2000]
                    parts.append(f"\n## 阶段报告\n{p1_text}")
                except OSError:
                    pass

            # ── Goal deliverables directory ─────────────────────────
            dlv_dir = gd / "deliverables"
            if dlv_dir.is_dir():
                dlv_files = sorted(dlv_dir.iterdir())
                if dlv_files:
                    dlv_lines = [
                        "\n## 目标共享产出（可跨任务复用）",
                        "以下文件位于 `deliverables/` 目录，由前序任务写入，",
                        "供当前及后续任务直接复用。你应该优先 read_file 此目录下的文件。",
                        "",
                    ]
                    for df in dlv_files:
                        try:
                            ds = df.stat().st_size
                            dlv_lines.append(f"- `{df}` ({_human_size(ds)}, {df.suffix or '无后缀'})")
                        except OSError:
                            dlv_lines.append(f"- `{df}`")
                    parts.append("\n".join(dlv_lines))
                # Always inject guidance on where to write reusable outputs
                parts.append(
                    "\n## 产出文件写入规范\n"
                    "可跨任务复用的产出（分析报告、代码、数据集、设计文档等）应写入 "
                    f"`{dlv_dir.resolve()}` 目录。仅轮次级的执行日志/反思写入 "
                    "当前任务目录下的 `outputs/`。这样后续任务可以直接复用你的成果，"
                    "无需从零开始。"
                )
        except Exception:
            pass

    # L3 grid workflow guidance (path reference — Yang reads grid JSON for full workflow)
    if workflow_guidance:
        parts.append(f"\n## 网格工作流摘要\n当前领域网格包含推荐工作流，完整步骤请用 `read_file` 读取网格 JSON 文件。\n摘要：\n{workflow_guidance[:500]}")

    return "\n".join(parts)


def _layer_round_facts(facts: list[RoundExecutionFact], round_num: int) -> str:
    """Layer 3: Accumulated round execution facts.

    Shows last 10 rounds in full detail, and provides a path reference
    to the complete ``06-execution-facts.json`` file for earlier rounds.
    Yang can use ``read_file`` to access any round's full data on demand.
    """
    if not facts:
        return f"## 当前轮次: {round_num}\n\n这是第一轮，请开始执行任务。"

    lines = [f"## 执行历史 (当前轮次: {round_num})\n"]
    lines.append(f"完整执行事实文件: 06-execution-facts.json（包含所有 {len(facts)} 轮的完整数据）\n")

    # Always show last 10 in detail
    recent = facts[-min(len(facts), 10):]
    for f in recent:
        lines.append(_format_single_fact(f))

    if len(facts) > 10:
        lines.append(f"\n（前 {len(facts) - 10} 轮详情请通过 read_file 读取 06-execution-facts.json）")

    return "\n".join(lines)


def _format_single_fact(f: RoundExecutionFact) -> str:
    """Format a single execution fact into a markdown block."""
    tool_info = f"调用了 {f.tool_call_count} 个工具" if f.had_action_request else "纯思考"
    yin_str = f"审批: {f.yin_decision}"
    if f.yin_decision in ("rejected", "modified") and f.yin_reason:
        yin_str += f" — {f.yin_reason[:150]}"
    return (
        f"### 第{f.round}轮\n"
        f"- 意图: {f.yang_intent_summary[:200]}\n"
        f"- 动作: {tool_info}\n"
        f"- {yin_str}\n"
        f"- 执行: {f.execution_status} — {f.execution_result_summary}\n"
    )


# ---------------------------------------------------------------------------
# Loop detection
# ---------------------------------------------------------------------------


def _detect_loop(facts: list[RoundExecutionFact]) -> str | None:
    """Detect repeated identical tool-call patterns across rounds.

    Detects two patterns:
    1. Identical intent summaries across recent rounds (existing).
    2. Alternating success/failure cycle where list_dir succeeds but
       read_file repeatedly fails (indicating the agent is trying to
       read non-existent files).
    """
    if len(facts) < _MAX_IDENTICAL_ROUNDS:
        return None

    # Pattern 1: Identical intent summaries
    recent = facts[-_MAX_IDENTICAL_ROUNDS:]
    intents = [f.yang_intent_summary[:60] for f in recent]
    hashes = [hashlib.md5(i.encode()).hexdigest() for i in intents]

    if len(set(hashes)) <= 2:  # Very similar intents
        return (
            f"检测到最近 {_MAX_IDENTICAL_ROUNDS} 轮出现高度相似的思考模式。\n"
            "建议：尝试不同的方法，或加载新的认知网格来打破循环。"
        )

    # Pattern 2: Alternating success/failure cycle (stuck pattern)
    # e.g. list_dir success → read_file failure → list_dir success → read_file failure...
    if len(facts) >= 6:
        last_6 = facts[-6:]
        statuses = [f.execution_status for f in last_6]

        # Check for alternating pattern: success, failure, success, failure...
        if statuses in [
            ["success", "failure", "success", "failure", "success", "failure"],
            ["success", "partial_failure", "success", "partial_failure", "success", "partial_failure"],
            ["failure", "success", "failure", "success", "failure", "success"],
            ["partial_failure", "success", "partial_failure", "success", "partial_failure", "success"],
        ]:
            return (
                "检测到交替执行模式："
                "你正在反复执行 list_directory（成功）然后尝试读取不存在的文件（失败）。"
                "请停止这种循环。你需要的信息可能位于 目标目录 中（参考上面的目标目录文件清单），"
                "而不是当前任务目录。使用 read_file 读取目标目录中的实际文件（如 blueprint.md 等）来获取所需信息。"
            )

    # Pattern 3: Consecutive veto rounds — Yang keeps proposing read-only
    # calls that Action refuses to execute, forming a deadlock.
    if len(facts) >= 3:
        last_3 = facts[-3:]
        veto_count = sum(
            1 for f in last_3
            if f.execution_status == "skipped"
            and f.tool_call_count > 0
            and f.yin_decision in ("approved", "modified")
        )
        if veto_count >= 3:
            return (
                "## ⚠️ 死锁检测：连续 3 轮被行动节点拒绝执行\n\n"
                "你连续 3 轮提出了只读工具调用（read_file/list_directory），"
                "但行动节点每次都将它们拦截——因为纯读取不产生交付物，"
                "而系统已经进入需要产出的阶段。\n\n"
                "**注意：exec 中的诊断命令（ls/head/wc/cat/grep 等）也被视为只读。**"
                "如果你在用 exec 做文件内容检查，这些不会被视为产出，"
                "请改用 write_file 直接写入成果。\n\n"
                "**打破死锁的唯一方法：本轮不要提出任何 read_file 或 list_directory。**\n"
                "直接用 write_file 写入成果文件（分析报告、代码、文档等），"
                "或用 exec 执行真正的任务脚本（build/render/generate 等）。"
                "你已有的知识和第 1 轮的收集结果足够支撑第一步产出。\n\n"
                "如果你确实 100% 确定需要读取某个特定文件才能继续："
                "只提 1 个 read_file + 1 个 write_file 搭配使用，"
                "不要提纯读取批次。"
            )

    return None


# ---------------------------------------------------------------------------
# Cognitive profile decision
# ---------------------------------------------------------------------------


def _build_state_from_facts(
    facts: list[RoundExecutionFact],
    mingjue: MingjueOutput,
) -> MetaCognitionState:
    """Extract current MetaCognitionState from last round's fact or defaults."""
    if facts and facts[-1].yao > 0:
        last = facts[-1]
        return MetaCognitionState(
            current_yao=last.yao,
            current_gua=last.current_gua or mingjue.trigram or "乾",
            current_sixiang=last.sixiang or "少阳",
        )
    return MetaCognitionState(
        current_yao=mingjue.initial_yao,
        current_gua=mingjue.trigram or "乾",
        current_sixiang="少阳",
    )


def _load_meta_grids() -> tuple[SixiangGrid, LiuyaoGrid, BaguaGrid]:
    """Load the three meta-cognitive grids from the templates directory."""
    from vingobot.core.workspace import get_workspace_paths
    wp = get_workspace_paths()
    grids_dir = wp.grids
    sixiang = SixiangGrid()
    liuyao = LiuyaoGrid()
    bagua = BaguaGrid()
    try:
        sf = grids_dir / "太极-四象.json"
        if sf.is_file():
            sixiang = parse_sixiang_grid(json.loads(sf.read_text(encoding="utf-8")))
    except Exception:
        logger.warning("[编织] 加载太极-四象网格失败")
    try:
        lf = grids_dir / "太极-六爻.json"
        if lf.is_file():
            liuyao = parse_liuyao_grid(json.loads(lf.read_text(encoding="utf-8")))
    except Exception:
        logger.warning("[编织] 加载太极-六爻网格失败")
    try:
        bf = grids_dir / "太极-八卦.json"
        if bf.is_file():
            bagua = parse_bagua_grid(json.loads(bf.read_text(encoding="utf-8")))
    except Exception:
        logger.warning("[编织] 加载太极-八卦网格失败")
    return sixiang, liuyao, bagua


def _build_cognitive_profile_prompt(
    mingjue: MingjueOutput,
    state: MetaCognitionState,
    facts: list[RoundExecutionFact],
    round_num: int,
    max_rounds: int,
    sixiang_grid: SixiangGrid,
    liuyao_grid: LiuyaoGrid,
    bagua_grid: BaguaGrid,
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for the cognitive profile decision LLM call."""

    # Serialize grids as readable text
    sixiang_text = _serialize_sixiang_grid(sixiang_grid)
    liuyao_text = _serialize_liuyao_grid(liuyao_grid)
    bagua_text = _serialize_bagua_grid(bagua_grid)

    # Build recent facts summary
    recent_facts_text = _format_recent_facts_for_profile(facts[-5:]) if facts else "（尚无执行历史）"

    system = f"""你是二爻·编织器的认知决策引擎。基于三套元认知格栅和执行历史，决定下一轮的认知姿态。

## 元认知格栅
{liuyao_text}

{sixiang_text}

{bagua_text}

## 决策原则
- **六爻推进**: 执行成功则推进，被驳回则后退，连续停滞则强推。
  初爻(接收)→二爻(反思)→三爻(行动)→四爻(调整)→五爻(精通)→上爻(超越)
- **四象选择**: 少阳(聚焦,T≈0.7)/老阳(发散,T≈0.8)/少阴(精准,T≈0.3)/老阴(批判,T≈0.2)
- **八卦路由**: 乾(创造)/坤(积累)/震(启动)/巽(渗透)/坎(风险)/离(澄清)/艮(暂停)/兑(表达)
- **动态参数**: 严谨类卦象(坎/离/艮)收敛参数，发散类卦象(乾/震/兑)放宽参数

## 当前状态
- 爻位: {state.current_yao} | 卦象: {state.current_gua} | 四象: {state.current_sixiang}
- 第 {round_num}/{max_rounds} 轮

## 执行历史（最近5轮）
{recent_facts_text}

## 输出格式
输出以下 JSON，不要其他文字：
{{
  "current_yao": <1-6 整数>,
  "current_gua": "<乾|坤|震|巽|坎|离|艮|兑>",
  "sixiang_selected": "<少阳|老阳|少阴|老阴>",
  "temperature": <0.1-1.2 浮点数>,
  "top_p": <0.5-1.0 浮点数>,
  "top_k": <1-100 整数>,
  "repetition_penalty": <0.8-1.5 浮点数>,
  "yao_reasoning": "<爻位推进理由>",
  "sixiang_reasoning": "<四象选择理由>",
  "gua_reasoning": "<卦象路由理由>"
}}"""

    user = f"当前任务: {mingjue.concrete_goal or mingjue.summary}\n目标ID: {mingjue.goal_id}"

    return system, user


def _parse_cognitive_profile(raw: str) -> CognitiveProfile:
    """Parse Weaver LLM JSON output into CognitiveProfile with defaults."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        # Try extracting from markdown code block
        for fence in ("```json", "```"):
            if fence in raw:
                start = raw.index(fence) + len(fence)
                end = raw.rfind("```")
                if end > start:
                    try:
                        data = json.loads(raw[start:end].strip())
                    except (json.JSONDecodeError, TypeError):
                        return CognitiveProfile()
                    break
            else:
                continue
            break
        else:
            return CognitiveProfile()

    return CognitiveProfile(
        current_yao=max(1, min(6, int(data.get("current_yao", 1)))),
        current_gua=str(data.get("current_gua", "乾")),
        sixiang_selected=str(data.get("sixiang_selected", "少阳")),
        temperature=float(data.get("temperature", 0.7)),
        top_p=float(data.get("top_p", 0.85)),
        top_k=int(data.get("top_k", 40)),
        repetition_penalty=float(data.get("repetition_penalty", 1.1)),
        yao_reasoning=str(data.get("yao_reasoning", "")),
        sixiang_reasoning=str(data.get("sixiang_reasoning", "")),
        gua_reasoning=str(data.get("gua_reasoning", "")),
    )


async def _decide_cognitive_profile(
    mingjue: MingjueOutput,
    state: MetaCognitionState,
    facts: list[RoundExecutionFact],
    round_num: int,
    sixiang_grid: SixiangGrid,
    liuyao_grid: LiuyaoGrid,
    bagua_grid: BaguaGrid,
) -> CognitiveProfile:
    """Call Weaver LLM to decide the cognitive profile for this round."""
    max_rounds = 30
    provider = _get_provider()
    if provider is None:
        # Fallback: use current state as profile
        return CognitiveProfile(
            current_yao=state.current_yao,
            current_gua=state.current_gua,
            sixiang_selected=state.current_sixiang,
        )

    system, user = _build_cognitive_profile_prompt(
        mingjue, state, facts, round_num, max_rounds,
        sixiang_grid, liuyao_grid, bagua_grid,
    )

    try:
        response = await provider.chat_with_retry(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = (response.content or "").strip()
        if content:
            return _parse_cognitive_profile(content)
    except Exception:
        logger.exception("[编织] 认知画像 LLM 调用失败，使用默认值")

    return CognitiveProfile(
        current_yao=state.current_yao,
        current_gua=state.current_gua,
        sixiang_selected=state.current_sixiang,
    )


def _build_eternal_context(
    mingjue: MingjueOutput,
    profile: CognitiveProfile,
    liuyao_grid: LiuyaoGrid,
    bagua_grid: BaguaGrid,
    sixiang_grid: SixiangGrid,
    round_num: int,
    max_rounds: int = 30,
) -> str:
    """Build the eternal context for Yang, with cognitive posture injected."""
    parts: list[str] = []

    # ── Task (hardcoded at top) ──
    parts.append("## 任务")
    parts.append(f"你的任务是: {mingjue.concrete_goal or mingjue.summary}")
    parts.append(f"目标ID: {mingjue.goal_id}")
    parts.append(f"第 {round_num}/{max_rounds} 轮")

    # ── Cognitive posture ──
    parts.append("")
    parts.append("## 认知姿态")

    # Yao info
    yao_keys = ["初爻", "二爻", "三爻", "四爻", "五爻", "上爻"]
    yao_name = yao_keys[profile.current_yao - 1] if 1 <= profile.current_yao <= 6 else "初爻"
    yao_node = liuyao_grid.nodes.get(yao_name)
    if yao_node:
        parts.append(f"爻位: {yao_name}({yao_node.phase}) — {yao_node.rule}")
        parts.append(f"行动指引: {yao_node.execution_hint}")
    else:
        parts.append(f"爻位: {yao_name}")

    # Gua info
    bagua_node = bagua_grid.nodes.get(profile.current_gua)
    if bagua_node:
        parts.append(f"卦象: {profile.current_gua}卦 — {bagua_node.context}")
        parts.append(f"行动基调: {bagua_node.prompt_prefix}")

    # Sixiang info
    sixiang_mode = sixiang_grid.modes.get(profile.sixiang_selected)
    if sixiang_mode:
        parts.append(f"四象: {profile.sixiang_selected}({sixiang_mode.description}) — {sixiang_mode.role_prompt}")

    # ── Workspace boundaries ──
    ctx = mingjue.context
    if ctx and ctx.task_dir:
        parts.append("")
        parts.append("## 工作区")
        parts.append(f"写入范围: {ctx.task_dir}")
        if ctx.goal_dir:
            parts.append(f"只读范围: {ctx.goal_dir}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Grid serialization helpers (for LLM consumption)
# ---------------------------------------------------------------------------


def _serialize_sixiang_grid(grid: SixiangGrid) -> str:
    """Serialize the sixiang grid for LLM consumption."""
    lines = []
    for name, mode in grid.modes.items():
        lines.append(f"- {name}: temp={mode.temperature}, top_p={mode.top_p}")
        lines.append(f"  描述: {mode.description}")
        lines.append(f"  角色: {mode.role_prompt}")
        lines.append(f"  阶段: {mode.act_phase}")
    return "\n".join(lines)


def _serialize_liuyao_grid(grid: LiuyaoGrid) -> str:
    """Serialize the liuyao grid for LLM consumption."""
    lines = []
    for name, node in grid.nodes.items():
        lines.append(f"- {name}: {node.phase}")
        lines.append(f"  规则: {node.rule}")
        lines.append(f"  指引: {node.execution_hint}")
        if node.temperature_override is not None:
            lines.append(f"  温度覆盖: {node.temperature_override}")
    return "\n".join(lines)


def _serialize_bagua_grid(grid: BaguaGrid) -> str:
    """Serialize the bagua grid for LLM consumption."""
    lines = []
    for name, node in grid.nodes.items():
        lines.append(f"- {name}卦: {node.context}")
        lines.append(f"  基调: {node.prompt_prefix}")
        lines.append(f"  默认四象: {node.default_sixiang}")
        if node.preferred_actions:
            lines.append(f"  偏好行动: {', '.join(node.preferred_actions)}")
    return "\n".join(lines)


def _format_recent_facts_for_profile(facts: list[RoundExecutionFact]) -> str:
    """Format recent facts for the cognitive profile prompt."""
    if not facts:
        return "（无）"
    lines = []
    for f in facts:
        lines.append(
            f"第{f.round}轮: {f.yang_intent_summary[:80]} → "
            f"审批:{f.yin_decision} 执行:{f.execution_status} "
            f"爻:{f.yao} 四象:{f.sixiang} 卦:{f.current_gua}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_discovered_grid_domain(trigram: str) -> str:
    """Return the domain name of the grid discovered for this trigram."""
    cached = _grid_discovery_cache.get(trigram)
    if cached is None:
        return ""
    # Re-read the grid to get domain
    try:
        from vingobot.core.workspace import get_workspace_paths
        from vingobot.goal.cognition_tools import parse_grid

        wp = get_workspace_paths()
        for gf in wp.grids.glob("*.json"):
            try:
                grid = parse_grid(gf)
                if grid and grid.trigram == trigram:
                    return grid.domain
            except Exception:
                continue
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Provider lazy-loading
# ---------------------------------------------------------------------------

_agent_name = "weaver"
_provider: Any = None


def _get_provider() -> Any:
    """Lazily obtain the LLM provider for this sixiang agent.

    Uses the per-agent config (``agents.defaults.sixiang.agents.weaver``)
    when available, falling back to the global defaults.
    """
    global _provider
    if _provider is not None:
        return _provider

    try:
        from vingobot.providers.factory import build_sixiang_provider_snapshot
        from vingobot.config.loader import load_config, resolve_config_env_vars

        config = resolve_config_env_vars(load_config())
        snapshot = build_sixiang_provider_snapshot(config, _agent_name)
        _provider = snapshot.provider
    except Exception:
        from loguru import logger as _lg

        _lg.warning("[编织] 无法初始化 provider")
    return _provider


def set_provider(provider: Any) -> None:
    """Explicitly set the provider used by this sixiang module."""
    global _provider
    _provider = provider


# ---------------------------------------------------------------------------
# Termination directives and round limit constants
# ---------------------------------------------------------------------------

_TERMINATION_MAX_ROUNDS = 30
"""默认最大轮次（与 task_inner_loop 保持一致）。"""

_TERMINATION_WARN_COUNT = 6
"""纯读轮次达到此值，在统计信息中显示警告。"""

_TERMINATION_AUTO_FLOOR = 5
"""任务内循环自动终止的纯读轮次下限。需与 task_inner_loop._AUTO_TERMINATE_FLOOR 同步。"""

_TERMINATION_AUTO_THRESHOLD = 12
"""超过此轮次+纯读轮次超过下限时自动终止。需与 task_inner_loop._AUTO_TERMINATE_THRESHOLD 同步。"""

_DEFAULT_MAX_ROUNDS = _TERMINATION_MAX_ROUNDS
"""导出给 task_inner_loop 使用。"""


def _build_termination_directive(round_num: int, read_only_round_count: int, had_successful_write: bool = False) -> str:
    """Build a termination STATS block (informational, not imperative).

    Yang receives execution statistics and decides autonomously when to
    call task_complete — the framework no longer issues imperative
    "you must" directives.
    """
    lines: list[str] = []
    lines.append("## 执行统计")

    # Round budget
    remaining = _TERMINATION_MAX_ROUNDS - round_num
    lines.append(f"- 已用轮次: {round_num}/{_TERMINATION_MAX_ROUNDS}（剩余 {remaining} 轮）")

    # Read-only counter
    if read_only_round_count > 0:
        lines.append(f"- 连续纯读取: {read_only_round_count} 轮")
        if read_only_round_count >= _TERMINATION_WARN_COUNT:
            lines.append(
                f"  （连续纯读取已达 {read_only_round_count} 轮，"
                f"系统将在第 {_TERMINATION_AUTO_THRESHOLD} 轮触发自动终止）"
            )

    # Write status
    if had_successful_write:
        lines.append("- 上轮: 成功写入/执行了交付物")
        lines.append("  （如果任务目标已全部达成，可调用 task_complete）")

    # File count
    lines.append("- 已产生交付物: 见任务目录下的 outputs/ 子目录")

    return "\n".join(lines)
