"""Core infrastructure for the sixiang (六爻) goal-driven architecture and shared tooling."""

from vingobot.core.workspace import (
    WorkspacePaths,
    init_workspace,
    get_workspace_paths,
    get_goal_dir,
    get_task_dir,
    create_task_folder,
    ensure_goal_dir,
)
from vingobot.core.goal_meta import (
    GoalMeta,
    read_goal_meta,
    write_goal_meta,
    update_goal_meta,
    scan_active_goals,
    get_all_goals,
)
from vingobot.core.manifest import (
    TaskManifest,
    create_manifest,
    read_manifest,
    update_manifest_status,
)
from vingobot.core.pending_queue import (
    PendingTask,
    PendingQueue,
)
from vingobot.core.goal_context import (
    GoalContext,
    load_goal_context,
    refresh_goal_context,
)
from vingobot.core.trajectory import (
    GoalProgress,
    TaskProgressEntry,
    read_goal_progress,
    update_goal_progress,
    read_progress_snapshot,
    read_trajectory_snapshot,
)
from vingobot.core.tool_base import Schema, Tool, tool_parameters
from vingobot.core.tool_registry import ToolRegistry
from vingobot.core.tool_executor import (
    check_exec_safety,
    check_path_safety,
    execute_builtin_tool,
)

__all__ = [
    "WorkspacePaths",
    "init_workspace",
    "get_workspace_paths",
    "get_goal_dir",
    "get_task_dir",
    "create_task_folder",
    "ensure_goal_dir",
    "GoalMeta",
    "read_goal_meta",
    "write_goal_meta",
    "update_goal_meta",
    "scan_active_goals",
    "get_all_goals",
    "TaskManifest",
    "create_manifest",
    "read_manifest",
    "update_manifest_status",
    "PendingTask",
    "PendingQueue",
    "GoalContext",
    "load_goal_context",
    "refresh_goal_context",
    "TrajectoryEntry",
    "GoalTrajectory",
    "append_trajectory_entry",
    "read_trajectory_snapshot",
    "Schema",
    "Tool",
    "tool_parameters",
    "ToolRegistry",
    "execute_builtin_tool",
    "check_path_safety",
    "check_exec_safety",
]
