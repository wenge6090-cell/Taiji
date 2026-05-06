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
    SixiangMode,
    TrigramNode,
    YaoNode,
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
# Trigram → tool set mapping (base tools only — skill tools are discovered)
# ---------------------------------------------------------------------------

_TRIGRAM_TOOLS: dict[str, list[str]] = {
    "qian": [
        "read_file",
        "list_directory",
        "write_file",
        "search_skills",
        "search_models",
        "load_grid",
        "web_search",
        "web_fetch",
        "query_capabilities",
        "task_complete",
    ],
    "kun": ["read_file", "list_directory", "write_file", "exec", "query_capabilities", "task_complete"],
    "zhen": ["read_file", "list_directory", "write_file", "exec", "query_capabilities", "task_complete"],
    "xun": [
        "read_file",
        "list_directory",
        "search_skills",
        "search_models",
        "load_grid",
        "web_search",
        "query_capabilities",
        "task_complete",
    ],
    "kan": [
        "read_file",
        "list_directory",
        "search_skills",
        "search_models",
        "load_grid",
        "exec",
        "query_capabilities",
        "task_complete",
    ],
    "li": ["read_file", "list_directory", "write_file", "search_skills", "query_capabilities", "task_complete"],
    "gen": ["read_file", "list_directory", "search_skills", "load_grid", "query_capabilities", "task_complete"],
    "dui": ["read_file", "list_directory", "write_file", "query_capabilities", "task_complete"],
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
    "search_skills": {
        "type": "function",
        "function": {
            "name": "search_skills",
            "description": "Search the L1 skill library for relevant reusable skills.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query for skills"},
                },
                "required": ["query"],
            },
        },
    },
    "search_models": {
        "type": "function",
        "function": {
            "name": "search_models",
            "description": "Search the L2 experience model library for relevant patterns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query for models"},
                },
                "required": ["query"],
            },
        },
    },
    "load_grid": {
        "type": "function",
        "function": {
            "name": "load_grid",
            "description": "Load an L3 cognitive grid by name (e.g. 'exploration', 'debugging').",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Cognitive grid name to load"},
                },
                "required": ["name"],
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
            "fully achieved the task goal. Provide a summary of what was done.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Summary of what was accomplished",
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

_grid_discovery_cache: dict[str, tuple[list[dict[str, Any]], str]] = {}


def _discover_skill_tools(trigram: str) -> tuple[list[dict[str, Any]], str]:
    """Discover L1 skill tools from the trigram's L3 cognitive grid.

    Reads the grid JSON at ``<workspace>/.taiji/cognition/grids/<trigram相关的grid>.json``,
    resolves referenced skills, and returns their tool definitions + workflow
    guidance text.

    Returns:
        (tool_definitions, workflow_guidance)
    """
    # Check cache first (per trigram, per session)
    cached = _grid_discovery_cache.get(trigram)
    if cached is not None:
        return cached

    discovered_tools: list[dict[str, Any]] = []
    workflow_guidance = ""

    try:
        from vingobot.core.workspace import get_workspace_paths
        from vingobot.goal.cognition_tools import parse_grid

        wp = get_workspace_paths()
        grids_dir = wp.grids

        if not grids_dir.is_dir():
            return [], ""

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
            _grid_discovery_cache[trigram] = ([], "")
            return [], ""

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

    except Exception as exc:
        logger.warning("[编织] 发现L3格栅技能工具失败: {}", exc)

    result = (discovered_tools, workflow_guidance)
    _grid_discovery_cache[trigram] = result
    return result


def _load_skill_tools(
    skill_name: str,
    skill_dir: Any,
) -> list[dict[str, Any]]:
    """Load tool definitions from a skill's SKILL.md using skill_parser.

    Parses the YAML frontmatter, converts each tool to OpenAI schema,
    and registers them in the global skill tool registry for Executor routing.
    """
    from vingobot.goal.skill_parser import parse_skill_md, register_skill_tools_from_meta

    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        return []

    meta = parse_skill_md(skill_md)
    if meta is None:
        return []

    # Register tools so Executor can route calls
    register_skill_tools_from_meta(meta)

    # Convert each tool to OpenAI schema
    return [t.to_openai_tool_def() for t in meta.tools]


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
_MAX_DETAIL_ROUNDS = 10  # early rounds beyond this count get summarised


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def weave(
    mingjue: MingjueOutput,
    facts: list[RoundExecutionFact],
    goal_context: Any,
    round_num: int = 1,
    previous_invoke_results: str = "",
    previous_yang_content: str = "",
    read_only_round_count: int = 0,
    had_successful_write: bool = False,
) -> WeaverOutput:
    """Weave context, tools, and cognitive profile for one round."""
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

    # 4. Discover L3 grid skill tools
    skill_tools, workflow_guidance = _discover_skill_tools(trigram)

    # 5. Build full system prompt
    system_prompt = _build_system_prompt_v2(
        eternal, mingjue, goal_context, facts, round_num,
        workflow_guidance, previous_invoke_results,
        previous_yang_content=previous_yang_content,
        read_only_round_count=read_only_round_count,
        had_successful_write=had_successful_write,
    )

    # 6. Build tool definitions
    tool_defs = [_BASE_TOOL_DEFS[name] for name in tool_names if name in _BASE_TOOL_DEFS]
    tool_defs.extend(skill_tools)

    # 7. LLM strategy weaving (inject CognitiveProfile)
    strategy_text = await _weave_with_llm(mingjue, facts, round_num, profile)
    if strategy_text:
        system_prompt += f"\n\n## 本轮策略\n{strategy_text}"

    # 8. Loop detection
    loop_warning = _detect_loop(facts)
    if loop_warning:
        system_prompt += f"\n\n## ⚠️ 死循环检测\n{loop_warning}"

    # 9. Extract discovered skill names
    grid_skills = []
    if skill_tools:
        for td in skill_tools:
            try:
                name = td["function"]["name"]
                if name.startswith("skill_"):
                    grid_skills.append(name[6:])
            except (KeyError, IndexError):
                pass

    return WeaverOutput(
        system_prompt=system_prompt,
        tool_definitions=tool_defs,
        cognitive_profile=profile,
        grid_domain=_get_discovered_grid_domain(trigram),
        grid_skills=grid_skills,
    )


# ---------------------------------------------------------------------------
# System prompt builder — three layers + grid workflow
# ---------------------------------------------------------------------------


def _build_system_prompt(
    mingjue: MingjueOutput,
    facts: list[RoundExecutionFact],
    goal_context: Any,
    round_num: int,
    workflow_guidance: str = "",
    previous_invoke_results: str = "",
) -> str:
    """Build the (now four-layer) system prompt for Yang.

    Layer order:
    1. Eternal context — identity, workspace boundaries, execution rules.
    2. Goal context — blueprint, meta, navigation hints.
    3. invoke 结果 — previous round's read-only outputs.
    4. Round facts — accumulated execution history.
    """

    # Layer 1: Eternal context (philosophical grounding)
    l1 = _layer_eternal(mingjue)

    # Layer 2: Goal context
    l2 = _layer_goal_context(mingjue, goal_context, workflow_guidance)

    # Layer 3: Previous invoke results — bridge between rounds
    if previous_invoke_results:
        l3 = f"## 上一轮只读查询结果\n{previous_invoke_results[:15000]}"
    else:
        l3 = ""

    # Layer 4: Round facts (accumulated execution history)
    l4 = _layer_round_facts(facts, round_num)

    parts = [l1, l2, l4]
    if l3:
        parts.insert(2, l3)  # inject between goal context and round facts
    return "\n\n---\n\n".join(p for p in parts if p)


def _build_system_prompt_v2(
    eternal: str,
    mingjue: MingjueOutput,
    goal_context: Any,
    facts: list[RoundExecutionFact],
    round_num: int,
    workflow_guidance: str = "",
    previous_invoke_results: str = "",
    previous_yang_content: str = "",
    read_only_round_count: int = 0,
    had_successful_write: bool = False,
) -> str:
    """Build the system prompt from pre-computed eternal context."""
    parts = [eternal]
    l2 = _layer_goal_context(mingjue, goal_context, workflow_guidance)
    if l2:
        parts.append(l2)
    if previous_invoke_results:
        parts.append(f"## 上一轮工具执行结果\n{previous_invoke_results[:15000]}")
    l4 = _layer_round_facts(facts, round_num)
    if l4:
        parts.append(l4)
    # Inject Yang's previous thinking for cross-round continuity
    if previous_yang_content:
        parts.append(
            "## 你上一轮的思考\n"
            + (previous_yang_content or "")[:3000]
            + "\n\n"
            "（以上是你上一轮的思考结论。无需重新读取相同文件验证，"
            "直接基于已有信息推进任务。）"
        )
    directive = _build_termination_directive(round_num, read_only_round_count, had_successful_write)
    if directive:
        parts.append(directive)
    return "\n\n---\n\n".join(p for p in parts if p)


def _layer_eternal(mingjue: MingjueOutput) -> str:
    """Layer 1: Minimal identity + workspace boundaries for Yang."""
    parts: list[str] = []
    parts.append("你是一个自主AI智能体，以六爻循环的方式推进目标任务。")
    parts.append("你是三爻·阳，保持空性——不预判、不预设，基于当前轮次的事实信息做出判断。")
    parts.append("你的唯一使命：推动当前任务走向完成。")

    ctx = mingjue.context
    if ctx and ctx.task_dir:
        parts.append("")
        parts.append("## 工作区")
        parts.append(f"当前任务目录: {ctx.task_dir}")

        # Derive project root for boundary hints
        try:
            task_p = Path(ctx.task_dir)
            ws_dir = task_p
            for _ in range(10):
                if (ws_dir / ".vingobot").is_dir():
                    break
                ws_dir = ws_dir.parent
            project_root = str(ws_dir.resolve())
        except Exception:
            project_root = ""

        parts.append("")
        parts.append("### 安全边界")
        parts.append(f"- **可读范围**: 项目根目录 `{project_root}` 及子目录内的所有文件（read_file/list_directory/exec 只读命令）")
        parts.append(f"- **写入限制**: write_file 仅允许在 `{ctx.task_dir}` 目录及其 outputs/ 子目录")
        parts.append("- **保护路径**: `.vingobot/.taiji/cognition/truths/` 目录及其内容受 L4 安全真理层保护，**禁止写入**")
        parts.append("- **禁止操作**: 不得尝试写入、删除或修改认知库文件 (skills/models/grids/truths)")
        parts.append("- **路径绝对化**: 使用绝对路径，不要使用相对路径（会被阴层拦截）")
        parts.append("")

        if ctx.goal_dir:
            parts.append(f"目标目录（可读）: {ctx.goal_dir}")
        if ctx.memory_dir:
            parts.append(f"目标记忆（可读）: {ctx.memory_dir}")

        if ctx.cognition_dirs:
            parts.append("认知库（可读）:")
            for kind, dir_path in ctx.cognition_dirs.items():
                parts.append(f"  {kind}: {dir_path}")
            parts.append("使用 load_grid / search_skills / search_models 按需读取认知库，也可用 read_file 直接读取。")

        parts.append("如需执行命令，请确保命令的工作目录在当前任务目录下。")

        # ── 效率提示 ───────────────────────────────────────────
        parts.append("")
        parts.append("## 效率提示")
        parts.append("你拥有 1M tokens 上下文窗口，无需过于保守。以下机制可大幅提升效率：")
        parts.append("")
        parts.append("1. **并发工具调用**：一轮最多同时调用 10 个工具，所有调用并发执行。")
        parts.append("   需要读取多个文件？一次性发送多个 read_file 调用，不要逐个串行。")
        parts.append("")
        parts.append("2. **跨轮信息传递**：上一轮的 read_file / list_directory / load_grid / search_skills 等")
        parts.append("   只读工具的返回结果会自动注入到下一轮的系统提示中，你无需反复重读相同内容。")
        parts.append("")
        parts.append("3. **广泛读权限**：除了当前任务目录，你还可以只读访问目标目录和认知库")
        parts.append("   (skills / models / grids) 中的文件。先用 list_directory 摸清目录结构，")
        parts.append("   再用 read_file 批量读取关键文件。")
        parts.append("")
        parts.append("4. **不确定能力边界？** 调用 query_capabilities 工具获取当前执行环境的")
        parts.append("   能力配额（并发数、读写权限、上下文预算等）。")

    return "\n".join(parts)


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
        except Exception:
            pass

    # Cognitive navigation — tools only, no content
    ctx = mingjue.context
    if ctx and ctx.suggested_grids:
        grids_str = ", ".join(ctx.suggested_grids)
        parts.append(f"\n## 认知导航\n建议加载的认知网格: {grids_str}")
        parts.append("使用 `load_grid` 工具按需加载认知网格内容。")
        parts.append("使用 `search_skills` 搜索可复用的技能。")
        parts.append("使用 `search_models` 搜索经验模型。")

    # L3 grid workflow guidance (if discovered)
    if workflow_guidance:
        parts.append(workflow_guidance)

    parts.append("\n## 工具使用原则")
    parts.append("- 按需调用工具，一次不要超过 3 个。")
    parts.append("- 完成任务后调用 `task_complete` 并给出总结。")
    parts.append("- 每个工具调用都有成本的，请谨慎使用。")

    return "\n".join(parts)


def _layer_round_facts(facts: list[RoundExecutionFact], round_num: int) -> str:
    """Layer 3: Accumulated round execution facts with layered summarization.

    Early rounds (beyond ``_MAX_DETAIL_ROUNDS``) are condensed into a
    one-line statistical summary to save prompt space; recent rounds are
    displayed in full detail for fine-grained context.
    """
    if not facts:
        return f"## 当前轮次: {round_num}\n\n这是第一轮，请开始执行任务。"

    lines = [f"## 执行历史 (当前轮次: {round_num})\n"]

    if len(facts) <= _MAX_DETAIL_ROUNDS:
        # ── All rounds in full detail ──────────────────────
        for f in facts:
            lines.append(_format_single_fact(f))
    else:
        # ── Early rounds → statistical summary ─────────────
        earlier = facts[:-_MAX_DETAIL_ROUNDS]
        summary = _summarize_earlier_rounds(earlier)
        if summary:
            lines.append(summary)

        # ── Recent rounds → full detail ────────────────────
        recent = facts[-_MAX_DETAIL_ROUNDS:]
        for f in recent:
            lines.append(_format_single_fact(f))

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


def _summarize_earlier_rounds(earlier: list[RoundExecutionFact]) -> str:
    """Condense a list of earlier rounds into a one-line statistical summary."""
    if not earlier:
        return ""

    total = len(earlier)
    success_count = sum(1 for f in earlier if f.execution_status == "success")
    failure_count = sum(1 for f in earlier if f.execution_status in ("failure", "partial_failure"))
    skipped_count = sum(1 for f in earlier if f.execution_status == "skipped")
    action_rounds = sum(1 for f in earlier if f.had_action_request)
    rejected_count = sum(1 for f in earlier if f.yin_decision == "rejected")

    first_round = earlier[0].round
    last_round = earlier[-1].round
    range_label = (
        f"第{first_round}轮" if first_round == last_round else f"第{first_round}-{last_round}轮"
    )

    parts = [f"（{range_label}概述：共 {total} 轮"]
    if action_rounds > 0:
        parts.append(f"，{action_rounds} 轮有工具调用")
    if success_count > 0:
        parts.append(f"，{success_count} 次执行成功")
    if failure_count > 0:
        parts.append(f"，{failure_count} 次执行失败")
    if rejected_count > 0:
        parts.append(f"，{rejected_count} 次被阴拒绝")
    if skipped_count > 0:
        parts.append(f"，{skipped_count} 轮纯思考")
    parts.append("）")

    return "".join(parts)


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
            f"检测到最近 {_MAX_IDENTICAL_ROUNDS} 轮出现高度相似的思考模式。"
            "请尝试不同的方法，或者加载新的认知网格来打破循环。"
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

    return None


# ---------------------------------------------------------------------------
# LLM strategy weaving
# ---------------------------------------------------------------------------


async def _weave_with_llm(
    mingjue: MingjueOutput,
    facts: list[RoundExecutionFact],
    round_num: int,
    profile: CognitiveProfile,
) -> str | None:
    """Call the LLM to generate a brief strategy text for this round.

    Produces 80-200 character Chinese strategy text that tells Yang
    what to focus on.  Returns ``None`` to fall back to the template
    strategy (no LLM available or generation failed).

    Injects the cognitive profile info so the strategy text is coherent
    with the cognitive posture for this round.
    """
    provider = _get_provider()
    if provider is None:
        return None

    trigram = mingjue.trigram or "kun"

    # Build cognitive posture summary
    yao_keys = ["初爻", "二爻", "三爻", "四爻", "五爻", "上爻"]
    yao_name = yao_keys[profile.current_yao - 1] if 1 <= profile.current_yao <= 6 else "初爻"

    posture_text = (
        f"当前认知姿态: {yao_name} | {profile.current_gua}卦 | {profile.sixiang_selected}\n"
        f"爻位理由: {profile.yao_reasoning}\n"
        f"四象理由: {profile.sixiang_reasoning}\n"
        f"卦象理由: {profile.gua_reasoning}"
    )

    system_prompt = (
        "你是二爻·编织器，为阳 Agent 生成本轮的策略部分。"
        "你的输出会拼接到阳的完整 system prompt 中。\n\n"
        f"八卦卦象: {trigram}\n\n"
        f"## 认知姿态\n{posture_text}\n\n"
        "请生成一段简短的中文策略（80-200字），告诉阳：\n"
        "1. 本轮的核心目标是什么（一句话）\n"
        "2. 最小可行的具体行动是什么（读取？写入？执行？）\n"
        "3. 上一轮进度的延续方式（如果有执行历史）\n"
        "4. 有什么需要避免的陷阱\n\n"
        "**重要**：\n"
        "- 直接输出策略文本，不要前缀，不要JSON，不要markdown\n"
        "- 控制在80-200字之间"
    )

    # Build user prompt
    user_prompt = f"当前任务: {mingjue.concrete_goal or mingjue.summary}"
    if facts:
        recent = facts[-3:]
        user_prompt += "\n\n执行历史（最近轮次）:"
        for f in recent:
            user_prompt += (
                f"\n  第{f.round}轮: {f.yang_intent_summary[:80]}"
                f" → 审批: {f.yin_decision}"
                f" → 执行: {f.execution_status}"
            )

    try:
        response = await provider.chat_with_retry(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = (response.content or "").strip()
        if content and 30 < len(content) < 500:
            return content
    except Exception:
        logger.debug("[编织] LLM 策略编织失败，使用模板")

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

    system = f"""你是二爻·编织器的认知决策引擎。你的任务是基于三套元认知格栅和执行历史，决定下一轮的认知姿态。

## 六爻认知阶段格栅 (太极-六爻.json)
{liuyao_text}

## 四象思维模式格栅 (太极-四象.json)
{sixiang_text}

## 八卦情境路由格栅 (太极-八卦.json)
{bagua_text}

## 决策原则
- **六爻推进**: 根据执行结果判断爻位进退。执行成功则推进，被驳回则后退，连续停滞则强推。
  初爻(接收)→二爻(反思)→三爻(行动)→四爻(调整)→五爻(精通)→上爻(超越)
- **四象选择**: 根据当前爻位和任务阶段选择匹配的思维模式。
  少阳(聚焦感知,T≈0.7)/老阳(发散探索,T≈0.8)/少阴(精准执行,T≈0.3)/老阴(批判反思,T≈0.2)
- **八卦路由**: 根据任务情境切换卦象。
  乾(创造)/坤(积累)/震(启动)/巽(渗透)/坎(风险)/离(澄清)/艮(暂停评估)/兑(表达)
- **动态参数**: temperature/top_p/top_k/repetition_penalty 需与四象模式匹配，根据卦象动态微调。
  严谨类卦象(坎/离/艮)收敛参数，发散类卦象(乾/震/兑)放宽参数。

## 当前状态
- 当前爻位: {state.current_yao}
- 当前卦象: {state.current_gua}
- 当前四象: {state.current_sixiang}
- 第 {round_num}/{max_rounds} 轮

## 执行历史（最近5轮）
{recent_facts_text}

请以 JSON 格式输出下一轮的认知画像。只输出 JSON，不要其他文字：
```json
{{
  "current_yao": <1-6 整数>,
  "current_gua": "<乾|坤|震|巽|坎|离|艮|兑>",
  "sixiang_selected": "<少阳|老阳|少阴|老阴>",
  "temperature": <0.1-1.2 浮点数>,
  "top_p": <0.5-1.0 浮点数>,
  "top_k": <1-100 整数>,
  "repetition_penalty": <0.8-1.5 浮点数>,
  "yao_reasoning": "<爻位推进理由，20字左右>",
  "sixiang_reasoning": "<四象选择理由，20字左右>",
  "gua_reasoning": "<卦象路由理由，20字左右>"
}}
```"""

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

_ENTIRE_READ_THRESHOLD = 3
"""连续N轮纯读取即注入警示。"""


_TERMINATION_MAX_ROUNDS = 30
"""默认最大轮次（与 task_inner_loop 保持一致）。"""


_TERMINATION_FORCE_CEILING = _TERMINATION_MAX_ROUNDS - 5
"""超过此轮次强制要求 task_complete。"""


_TERMINATION_CRITICAL_CEILING = _TERMINATION_MAX_ROUNDS - 3
"""超过此轮次注入极其强硬的终止指令。"""


_TERMINATION_EARLY_COUNT = 4
"""纯读轮次达到此值，注入早期警示。"""

_ENFORCE_WRITE_COUNT = 3
"""连续纯读轮次达到此值，注入强制写入指令。"""

_TERMINATION_WARN_COUNT = 6
"""纯读轮次达到此值，注入警告。"""


_TERMINATION_CRITICAL_COUNT = 8
"""纯读轮次达到此值，注入强制终止指令。"""


_TERMINATION_AUTO_FLOOR = 5
"""任务内循环自动终止的纯读轮次下限。需与 task_inner_loop._AUTO_TERMINATE_FLOOR 同步。"""


_TERMINATION_AUTO_THRESHOLD = 12
"""超过此轮次+纯读轮次超过下限时自动终止。需与 task_inner_loop._AUTO_TERMINATE_THRESHOLD 同步。"""


_DEFAULT_MAX_ROUNDS = _TERMINATION_MAX_ROUNDS
"""导出给 task_inner_loop 使用。"""


def _build_termination_directive(round_num: int, read_only_round_count: int, had_successful_write: bool = False) -> str:
    """Build a round-termination directive based on execution patterns.

    Injects increasingly forceful instructions into the system prompt as
    the read-only loop deepens, guiding Yang to either call task_complete
    or switch to write/exec tools.
    """
    lines: list[str] = []

    # ── Round-based urgency ────────────────────────────────────
    if round_num >= _TERMINATION_CRITICAL_CEILING:
        lines.append(
            "## 强制提示: 即将达到轮次上限\n"
            "本轮结束后系统将自动终止。请立即调用 `task_complete` "
            "提交你已经完成的所有工作。如果有未完成的成果，先写文件再结束。"
        )
    elif round_num >= _TERMINATION_FORCE_CEILING:
        remaining = _TERMINATION_MAX_ROUNDS - round_num
        lines.append(
            "## 剩余轮次警告\n"
            f"你还剩 {remaining} 轮。请加速推进，尽快调用 `task_complete` 完成本任务。"
        )

    # ── Force completion check after successful write ──────────
    if had_successful_write:
        lines.append(
            "## 上轮执行成功 — 强制完成判定\n"
            "上一轮你已成功通过 `write_file` 或 `exec` 写入/执行了成果。\n"
            "你必须立即对照任务目标，逐项检查所有交付物是否已创建且内容正确。\n"
            "如果已全部达成 → **本轮必须调用 `task_complete` 结束任务**，不得再执行任何操作。\n"
            "如果有遗漏 → 本轮使用 `write_file` 补充写入。不得反复读取已成功写入的文件做'验证'。\n"
            "**警告：read_file/list_directory 是只读工具会即时执行，无需再次验证。确认后立即 task_complete。**"
        )

    # ── No-action enforcement: push from read-only to writing ──
    if read_only_round_count >= _ENFORCE_WRITE_COUNT:
        lines.append(
            f"## ⚠️ 必须写入成果\n"
            f"你已经连续 {read_only_round_count} 轮只读取信息，尚未产生任何交付物。\n"
            "**本轮你必须使用 `write_file` 写入成果文件**，不得继续仅做读操作。\n"
            "任务若要完成，必须将收集到的信息通过 `write_file` 写成文件。\n"
            "你上一轮的思考和已读内容都已注入到本轮的上下文中——不要重新读取。"
        )

    # ── Read-only loop detection ───────────────────────────────
    if read_only_round_count >= _TERMINATION_CRITICAL_COUNT:
        remaining = _TERMINATION_AUTO_THRESHOLD - round_num
        if remaining < 0:
            remaining = 0
        lines.append(
            f"## ⛔ 自读循环强制终止警告\n"
            f"你已经连续 {read_only_round_count} 轮只读取信息，系统已检测到自读循环。\n"
            f"**再执行 {remaining} 轮将触发系统强制终止**，任务将被关闭且不计为完成。\n"
            "立即执行以下操作之一：\n"
            "1. 如果你已经收集了足够信息，立即调用 `task_complete` 结束本任务。\n"
            "2. 如果你还有工作需要推进，立即使用 `write_file` 写入成果文件。\n"
            "3. **绝不能再调用 `read_file` 或 `list_directory`**——你读过的内容已通过跨轮注入存在于上下文中。"
        )
    elif read_only_round_count >= _TERMINATION_WARN_COUNT:
        lines.append(
            f"## 执行效率警告\n"
            f"你已经连续 {read_only_round_count} 轮只读取信息。请立即切换行动：\n"
            "- 使用 `write_file` 写入成果文件\n"
            "- 使用 `exec` 执行命令\n"
            "- 如果无需更多操作，调用 `task_complete` 结束"
        )
    elif read_only_round_count >= _TERMINATION_EARLY_COUNT:
        lines.append(
            f"## 效率提示\n"
            f"注意：你已经连续 {read_only_round_count} 轮只有读取操作。"
            "如果信息已经足够，请直接推进执行或调用 `task_complete`。"
        )

    return "\n\n---\n\n".join(lines) if lines else ""
