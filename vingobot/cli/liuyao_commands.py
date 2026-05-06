"""
六爻 CLI 命令 — vingobot liuyao 子命令组。

Commands:
    vingobot liuyao goal list     列出所有目标
    vingobot liuyao goal show ID  查看目标详情
    vingobot liuyao goal new ID   创建新目标
    vingobot liuyao queue list    查看待办队列
    vingobot liuyao queue add DESC 添加任务到队列

Note: 六爻协程池的启动/停止由 Agent 内部 TPN 工具控制，
"""

from __future__ import annotations

import typer

liuyao_app = typer.Typer(help="六爻目标驱动循环", no_args_is_help=True)
goal_app = typer.Typer(help="目标管理", no_args_is_help=True)
queue_app = typer.Typer(help="任务队列管理", no_args_is_help=True)

liuyao_app.add_typer(goal_app, name="goal")
liuyao_app.add_typer(queue_app, name="queue")


# ===========================================================================
# goal list
# ===========================================================================


@goal_app.command("list")
def list_goals():
    """列出所有目标及其状态。"""
    from vingobot.core.goal_meta import get_all_goals

    goals = get_all_goals()
    if not goals:
        typer.echo("(无目标)")
        return

    typer.echo(f"{'ID':<24} {'名称':<20} {'状态':<12} {'优先级':<8}")
    typer.echo("-" * 64)
    for g in goals:
        name = (g.name or g.id)[:18]
        status = g.status or "unknown"
        priority = g.priority
        icon = {"active": "🔄", "completed": "✅", "failed": "❌", "paused": "⏸"}.get(status, "❓")
        typer.echo(f"{icon} {g.id:<22} {name:<20} {status:<12} {priority:<8}")


# ---------------------------------------------------------------------------
# goal show
# ---------------------------------------------------------------------------


@goal_app.command("show")
def show_goal(
    goal_id: str = typer.Argument(..., help="目标 ID"),
):
    """查看目标详细信息。"""
    from vingobot.core.goal_context import load_goal_context
    from vingobot.core.goal_meta import read_goal_meta

    meta = read_goal_meta(goal_id)
    if meta is None:
        typer.echo(f"❌ 目标 '{goal_id}' 不存在", err=True)
        raise typer.Exit(1)

    ctx = load_goal_context(goal_id)

    typer.echo(f"目标: {meta.name or goal_id}")
    typer.echo(f"ID: {meta.id}")
    typer.echo(f"状态: {meta.status}")
    typer.echo(f"优先级: {meta.priority}")
    typer.echo(f"创建时间: {meta.created_at}")
    typer.echo(f"最近活跃: {meta.last_active}")
    typer.echo()

    if ctx and ctx.blueprint_summary:
        typer.echo(f"蓝图:\n{ctx.blueprint_summary[:500]}")
        typer.echo()

    if ctx and ctx.recent_task_statuses:
        typer.echo("近期任务:")
        for t in ctx.recent_task_statuses:
            icon = "✅" if t.status == "completed" else "❌"
            typer.echo(f"  {icon} {t.task_id}: {t.status} — {t.summary_snippet[:80]}")


# ---------------------------------------------------------------------------
# goal new
# ---------------------------------------------------------------------------


@goal_app.command("new")
def new_goal(
    goal_id: str = typer.Argument(..., help="目标 ID"),
    description: str = typer.Option("", "--desc", "-d", help="目标描述"),
    priority: int = typer.Option(5, "--priority", "-p", help="优先级 (1-10)"),
):
    """创建新目标。"""
    from vingobot.core.workspace import ensure_goal_dir

    ensure_goal_dir(goal_id, priority=priority, description=description or None)
    typer.echo(f"✅ 目标 '{goal_id}' 已创建 (优先级: {priority})")


# ===========================================================================
# queue list
# ===========================================================================


@queue_app.command("list")
def list_queue():
    """查看待办任务队列。"""
    from vingobot.core.pending_queue import PendingQueue

    queue = PendingQueue()
    tasks = queue.scan_pending()

    if not tasks:
        typer.echo("(队列为空)")
        return

    typer.echo(f"{'目标':<24} {'描述':<60}")
    typer.echo("-" * 84)
    for t in tasks:
        typer.echo(f"{t.goal_id:<24} {t.description[:58]}")


# ---------------------------------------------------------------------------
# queue add
# ---------------------------------------------------------------------------


@queue_app.command("add")
def add_to_queue(
    description: str = typer.Argument(..., help="任务描述"),
    goal_id: str = typer.Option("default", "--goal-id", "-g", help="目标 ID"),
):
    """添加任务到待办队列。"""
    from vingobot.core.pending_queue import PendingQueue, PendingTask

    queue = PendingQueue()
    task = PendingTask(id="", goal_id=goal_id, description=description, source="cli")

    file_path = queue.enqueue(task)
    typer.echo(f"✅ 任务已加入队列: {file_path}")
