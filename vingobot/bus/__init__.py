"""Message bus module for decoupled channel-agent communication."""

from vingobot.bus.events import InboundMessage, OutboundMessage
from vingobot.bus.queue import MessageBus

__all__ = ["MessageBus", "InboundMessage", "OutboundMessage"]
