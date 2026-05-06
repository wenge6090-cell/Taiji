from pathlib import Path

from vingobot.config.paths import (
    get_bridge_install_dir,
    get_cli_history_path,
    get_cron_dir,
    get_data_dir,
    get_legacy_sessions_dir,
    get_logs_dir,
    get_media_dir,
    get_runtime_subdir,
    get_workspace_path,
    is_default_workspace,
)


def test_runtime_dirs_follow_config_path(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "instance-a" / "config.json"
    monkeypatch.setattr("vingobot.config.paths.get_config_path", lambda: config_file)

    assert get_data_dir() == config_file.parent
    assert get_runtime_subdir("cron") == config_file.parent / "cron"
    assert get_cron_dir() == config_file.parent / "cron"
    assert get_logs_dir() == config_file.parent / "logs"


def test_media_dir_supports_channel_namespace(monkeypatch, tmp_path: Path) -> None:
    config_file = tmp_path / "instance-b" / "config.json"
    monkeypatch.setattr("vingobot.config.paths.get_config_path", lambda: config_file)

    assert get_media_dir() == config_file.parent / "media"
    assert get_media_dir("telegram") == config_file.parent / "media" / "telegram"


def test_shared_and_legacy_paths_resolve_to_project_root() -> None:
    """These paths now live under the project-root .vingobot directory."""
    project_root = Path(__file__).parent.parent.parent
    expected_vingobot = project_root / ".vingobot"
    assert get_cli_history_path() == expected_vingobot / "history" / "cli_history"
    assert get_bridge_install_dir() == expected_vingobot / "bridge"
    assert get_legacy_sessions_dir() == expected_vingobot / "sessions"


def test_workspace_path_is_explicitly_resolved() -> None:
    # With no argument, resolves ".vingobot" relative to CWD
    assert get_workspace_path().name == ".vingobot"
    assert get_workspace_path("~/custom-workspace") == Path.home() / "custom-workspace"


def test_is_default_workspace_distinguishes_default_and_custom_paths() -> None:
    # The default workspace name is ".vingobot"
    # We pass ".vingobot" as a string which gets resolved via expanduser().resolve()
    # Actually None resolves to Path(".vingobot").resolve() == CWD/.vingobot
    # So let's test with the relative path
    from pathlib import Path
    default = Path(".vingobot").resolve()
    assert is_default_workspace(str(default)) is True
    assert is_default_workspace("~/custom-workspace") is False
