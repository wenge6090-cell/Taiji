"""
独立运行六爻协程池，观察任务变化。

用法: python run_sixiang_standalone.py [--goal GOAL_ID] [--description DESC] [--workers N]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from vingobot.config.loader import load_config, resolve_config_env_vars
from vingobot.core.workspace import init_workspace
from vingobot.core.pending_queue import PendingQueue, PendingTask
from vingobot.core.goal_meta import get_all_goals, read_goal_meta
from vingobot.goal.coroutine import WorkerPool
from vingobot.goal.sixiang_loop import execute_complete_sixiang_loop
from vingobot.goal.dialogue_target import _inject_provider, _inject_sixiang_providers
from vingobot.providers.factory import make_provider


def print_header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_goals() -> None:
    """打印所有目标状态"""
    goals = get_all_goals()
    if not goals:
        print("  (无目标)")
        return
    print(f"  {'ID':<28} {'名称':<20} {'状态':<10} {'优先级':<6} {'已完成轮次'}")
    print(f"  {'-'*76}")
    for g in goals:
        name = (g.name or g.id)[:18]
        status_icon = {"active": "🔄", "completed": "✅", "failed": "❌", "paused": "⏸️"}.get(g.status, "❓")
        print(f"  {status_icon} {g.id:<26} {name:<20} {g.status:<10} {g.priority:<6} {g.rounds_completed}")


def print_queue() -> None:
    """打印待处理队列"""
    queue = PendingQueue()
    tasks = queue.scan_pending()
    if not tasks:
        print("  (队列为空)")
        return
    print(f"  队列中有 {len(tasks)} 个待处理任务:")
    for t in tasks:
        print(f"    - [{t.goal_id}] {t.description[:80]}")


async def monitor_loop(pool: WorkerPool, interval: float = 5.0) -> None:
    """监控循环：目标变化时显示详情，空闲时静默，30秒心跳保活。"""
    print_header("监控运行中 — 状态变化时自动输出详情")

    last_goals = None
    last_active_goals: dict[str, int] = {}
    heartbeat_elapsed = 0

    while pool.running:
        await asyncio.sleep(interval)
        heartbeat_elapsed += interval

        current_goals = get_all_goals()
        current_state = {(g.id, g.status, g.rounds_completed) for g in current_goals}
        current_active = pool.get_active_goals()

        changed = current_state != last_goals
        active_changed = current_active != last_active_goals

        if changed:
            print(f"\n--- [{datetime.now().strftime('%H:%M:%S')}] 状态变化 ---")
            print_goals()
            print_queue()
            last_goals = current_state
            _print_latest_round(current_active)
            heartbeat_elapsed = 0
        elif active_changed:
            if current_active:
                goals_str = ", ".join(f"{gid}(w{wid})" for gid, wid in current_active.items())
                print(f"  [worker] 开始处理: {goals_str}")
            else:
                print(f"  [worker] 处理完成")
            last_active_goals = dict(current_active)
            heartbeat_elapsed = 0

        # 30秒心跳：无状态变化时提醒仍在运行
        if heartbeat_elapsed >= 30:
            if current_active:
                gid = next(iter(current_active))
                meta = read_goal_meta(gid)
                rounds = meta.rounds_completed if meta else "?"
                print(f"  [心跳] 仍在运行 — {gid} 已完成 {rounds} 轮")
            heartbeat_elapsed = 0

        # 检查完成条件
        all_done = all(g.status in ("completed", "failed") for g in current_goals if g.id != "default")
        if all_done and not PendingQueue().scan_pending():
            print("\n  所有非默认目标已完成，队列为空。")
            break

    print_header("监控结束")


def _print_latest_round(active_goals: dict[str, int]) -> None:
    """读取活跃目标的最新轮次输出，打印工具调用摘要。"""
    if not active_goals:
        return
    gid = next(iter(active_goals))
    taiji = os.path.expanduser("~/.vingobot/.taiji")
    tasks_dir = Path(taiji) / "goals" / gid / "tasks"
    if not tasks_dir.is_dir():
        return

    # 找最新的任务目录
    task_dirs = sorted(tasks_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for td in task_dirs[:3]:
        if not td.is_dir():
            continue
        outputs_dir = td / "outputs"
        if not outputs_dir.is_dir():
            continue
        round_files = sorted(outputs_dir.glob("*-round.json"),
                             key=lambda p: int(p.stem.split("-")[0]))
        if not round_files:
            continue

        latest = round_files[-1]
        try:
            data = json.loads(latest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        rn = data.get("round", "?")
        tc = data.get("tool_calls") or []
        called_tc = data.get("called_task_complete", False)
        yin = data.get("yin_decision", "")

        # 生成工具摘要：name(param=val) 格式
        tools_str = ", ".join(
            _short_tool(c) for c in tc[:4]
        )
        if len(tc) > 4:
            tools_str += f" +{len(tc)-4}"

        status = "✅ 完成" if called_tc else "⏳ 处理中"
        if "rejected" in yin:
            status = "❌ 被拒"

        task_name = td.name
        print(f"  ── {task_name} 第{rn}轮 {status}")
        if tools_str:
            print(f"     工具: {tools_str}")
        break


def _short_tool(call: dict) -> str:
    """工具调用摘要，如 write_file(path=...)"""
    name = call.get("name", "?")
    args = call.get("arguments") or {}
    # 选第一个有意义的参数
    for key in ("path", "command", "url", "query", "content"):
        val = args.get(key)
        if val:
            s = str(val)[:40]
            return f"{name}({key}={s})"
    return name


async def main() -> None:
    parser = argparse.ArgumentParser(description="独立运行六爻协程池")
    parser.add_argument("--goal", "-g", default="monthly-token-income", help="目标 ID")
    parser.add_argument("--description", "-d", default="继续执行月收入目标的下一个任务", help="任务描述")
    parser.add_argument("--workers", "-w", type=int, default=1, help="Worker 数量")
    parser.add_argument("--monitor-only", action="store_true", help="仅监控，不添加任务")
    args = parser.parse_args()

    # ── 1. 加载配置 ──────────────────────────────────────────
    print_header("加载配置")
    config = resolve_config_env_vars(load_config())
    print(f"  Provider: {config.agents.defaults.provider}")
    print(f"  Model: {config.agents.defaults.model}")

    # ── 2. 初始化工作区 ──────────────────────────────────────
    print_header("初始化六爻工作区")
    from vingobot.config.paths import get_workspace_path as _get_ws
    ws_path = _get_ws()
    init_workspace(ws_path / ".taiji")
    print(f"  工作区: {ws_path / '.taiji'}")

    # ── 3. 注入 Provider ─────────────────────────────────────
    print_header("注入 LLM Provider")
    # 优先使用 per-agent 六爻配置，回退到全局默认
    try:
        _inject_sixiang_providers(config)
        print("  使用 per-agent 六爻模型配置")
    except Exception:
        provider = make_provider(config)
        _inject_provider(provider)
        print(f"  使用全局默认 provider (model={config.agents.defaults.model})")

    # ── 4. 当前状态快照 ──────────────────────────────────────
    print_header("运行前状态")
    print_goals()
    print_queue()

    # ── 5. 添加任务到队列 ────────────────────────────────────
    if not args.monitor_only:
        print_header(f"添加任务: [{args.goal}] {args.description}")
        queue = PendingQueue()
        task = PendingTask(
            goal_id=args.goal,
            description=args.description,
            source="standalone_script",
        )
        file_path = queue.enqueue(task)
        print(f"  任务文件: {file_path}")
        print_queue()

    # ── 6. 启动六爻协程池 ────────────────────────────────────
    print_header(f"启动六爻协程池 (workers={args.workers})")
    
    pool = WorkerPool(
        max_workers=args.workers,
        run_task_fn=execute_complete_sixiang_loop,
    )

    # 启动监控协程
    monitor_task = asyncio.create_task(monitor_loop(pool, interval=5.0))
    
    try:
        await pool.start()
        await pool.wait_stopped()
    except KeyboardInterrupt:
        print("\n\n  收到中断信号，正在停止...")
    finally:
        await pool.stop()
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

    # ── 7. 最终状态 ──────────────────────────────────────────
    print_header("运行后最终状态")
    print_goals()
    print_queue()

    # 显示目标详情
    meta = read_goal_meta(args.goal)
    if meta:
        print(f"\n  目标 [{args.goal}] 详情:")
        print(f"    状态: {meta.status}")
        print(f"    完成轮次: {meta.rounds_completed}")
        print(f"    最后暗驱: {meta.last_anqu_at}")
        bp_file = Path(ws_path) / ".taiji" / "goals" / args.goal / "blueprint.md"
        if bp_file.is_file():
            print(f"    蓝图: {bp_file.read_text(encoding='utf-8')[:200]}...")


if __name__ == "__main__":
    asyncio.run(main())
