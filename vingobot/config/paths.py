"""Runtime path helpers derived from the active config context."""

from __future__ import annotations

from pathlib import Path

from vingobot.config.loader import get_config_path
from vingobot.utils.helpers import ensure_dir


def get_data_dir() -> Path:
    """Return the instance-level runtime data directory."""
    return ensure_dir(get_config_path().parent)


def get_runtime_subdir(name: str) -> Path:
    """Return a named runtime subdirectory under the instance data dir."""
    return ensure_dir(get_data_dir() / name)


def get_media_dir(channel: str | None = None) -> Path:
    """Return the media directory, optionally namespaced per channel."""
    base = get_runtime_subdir("media")
    return ensure_dir(base / channel) if channel else base


def get_cron_dir() -> Path:
    """Return the cron storage directory."""
    return get_runtime_subdir("cron")


def get_logs_dir() -> Path:
    """Return the logs directory."""
    return get_runtime_subdir("logs")


def get_workspace_path(workspace: str | None = None) -> Path:
    """Resolve and ensure the agent workspace path.

    When *workspace* is provided it is used directly.
    Otherwise the authoritative workspace path is read from the active
    configuration file (``Config.agents.defaults.workspace``).

    Relative paths are resolved against the project root so that the
    default ``".vingobot"`` always points to ``<project>/.vingobot``
    regardless of CWD.
    """
    if workspace:
        path = Path(workspace).expanduser().resolve()
    else:
        from vingobot.config.loader import load_config

        config = load_config()
        raw = config.agents.defaults.workspace
        path = _resolve_workspace_path(raw)
    return ensure_dir(path)


def is_default_workspace(workspace: str | Path | None) -> bool:
    """Return whether a workspace resolves to vingobot's default workspace path."""
    if workspace is not None:
        current = _resolve_workspace_path(workspace)
    else:
        current = get_workspace_path()
    try:
        return current.name == ".vingobot"
    except OSError:
        return False


def _resolve_workspace_path(raw: str | Path) -> Path:
    """Resolve a raw workspace path string to an absolute path.

    Relative paths are resolved against the project root so that
    ".vingobot" always means ``<project>/.vingobot``, not ``<CWD>/.vingobot``.
    """
    path = Path(raw)
    if not path.is_absolute():
        # config.json lives at <project>/.vingobot/config.json,
        # so two parents up gives us the project root.
        config_dir = get_config_path().parent  # <project>/.vingobot
        project_root = config_dir.parent       # <project>
        path = (project_root / path).resolve()
    else:
        path = path.expanduser().resolve()
    return path


def _get_vingobot_dir() -> Path:
    """Return the vingobot data directory under the project root.

    Resolves relative to this file::

        vingobot/config/paths.py  →  project-root/.vingobot
    """
    return Path(__file__).parent.parent.parent / ".vingobot"


def get_cli_history_path() -> Path:
    """Return the shared CLI history file path."""
    return _get_vingobot_dir() / "history" / "cli_history"


def get_bridge_install_dir() -> Path:
    """Return the shared WhatsApp bridge installation directory."""
    return _get_vingobot_dir() / "bridge"


def get_legacy_sessions_dir() -> Path:
    """Return the legacy global session directory used for migration fallback."""
    return _get_vingobot_dir() / "sessions"
