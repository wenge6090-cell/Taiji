"""
独立运行六爻协程池，观察任务变化。

用法: python run_sixiang_standalone.py [--goal GOAL_ID] [--description DESC] [--workers N]
"""

from __future__ import annotations

import argparse
import asyncio
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


async def monitor_loop(pool: WorkerPool, interval: float = 3.0) -> None:
    """监控循环：定时打印目标和队列状态"""
    print_header("监控开始 — 每 3 秒刷新状态")
    
    last_goals = None
    while pool.running:
        await asyncio.sleep(interval)
        
        # 检查目标变化
        current_goals = get_all_goals()
        current_state = {(g.id, g.status, g.rounds_completed) for g in current_goals}
        
        if current_state != last_goals:
            print(f"\n--- [{datetime.now().strftime('%H:%M:%S')}] 状态变化 ---")
            print_goals()
            print_queue()
            last_goals = current_state
        
        # 显示活跃 worker 数
        active = pool.active_count
        active_goals = pool.get_active_goals()
        if active_goals:
            goals_str = ", ".join(f"{gid}(w{wid})" for gid, wid in active_goals.items())
            print(f"  [workers active: {active}] 正在处理: {goals_str}")
        
        # 检查是否所有任务都完成了
        all_done = all(g.status in ("completed", "failed") for g in current_goals if g.id != "default")
        if all_done and not PendingQueue().scan_pending():
            print("\n  所有非默认目标已完成，队列为空。")
            break
    
    print_header("监控结束")


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
        if meta.blueprint:
            print(f"    蓝图: {meta.blueprint[:200]}...")


if __name__ == "__main__":
    asyncio.run(main())
