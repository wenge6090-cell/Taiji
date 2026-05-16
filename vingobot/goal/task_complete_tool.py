"""
task_complete — Special tool that signals end of the inner loop.

This is a zero-side-effect tool whose sole purpose is to let Yang declare
that the current task is done.  The inner loop detects calls to this tool
and exits, passing control back to Anqu for goal-level evaluation.
"""

from __future__ import annotations

from typing import Any

TASK_COMPLETE_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "task_complete",
        "description": (
            "Mark the current task as complete. Call this when you have "
            "fully achieved the task's stated goal. Provide a summary of "
            "what was accomplished, any files created/modified, and the "
            "key outcomes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Detailed summary of what was accomplished in this task.",
                },
                "outcome": {
                    "type": "string",
                    "description": "Brief outcome: 'success' or 'partial'.",
                    "enum": ["success", "partial"],
                },
            },
            "required": ["summary"],
        },
    },
}


class TaskCompleteTool:
    """Lightweight tool class for the task_complete marker."""

    name: str = "task_complete"
    description: str = (
        "Mark the current task as complete. Call this when you have "
        "fully achieved the task's stated goal."
    )
    read_only: bool = True

    @staticmethod
    async def execute(summary: str = "", outcome: str = "success", **kwargs: Any) -> str:
        """Record task completion (no side effects)."""
        return f"任务已标记为完成: {summary[:200]}"


def to_schema() -> dict[str, Any]:
    """Return the OpenAI function schema for task_complete."""
    return TASK_COMPLETE_DEFINITION
