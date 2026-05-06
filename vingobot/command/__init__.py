"""Slash command routing and built-in handlers."""

from vingobot.command.builtin import register_builtin_commands
from vingobot.command.router import CommandContext, CommandRouter

__all__ = ["CommandContext", "CommandRouter", "register_builtin_commands"]
