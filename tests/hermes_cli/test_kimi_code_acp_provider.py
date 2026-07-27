"""Provider and credential contracts for the Kimi Code ACP backend."""

from __future__ import annotations

from argparse import Namespace
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _install_fake_kimi(home: Path) -> Path:
    binary = home / ".kimi-code" / "bin" / "kimi"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    return binary


def _clear_external_process_overrides(monkeypatch) -> None:
    for name in (
        "HERMES_REAL_HOME",
        "KIMI_CODE_CLI_PATH",
        "HERMES_COPILOT_ACP_COMMAND",
        "COPILOT_CLI_PATH",
        "HERMES_COPILOT_ACP_ARGS",
        "COPILOT_ACP_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_kimi_code_profile_and_provider_catalog_metadata():
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.provider_catalog import provider_catalog_by_slug
    from providers import get_provider_profile

    profile = get_provider_profile("kimi-code")
    assert profile is not None
    assert profile.name == "kimi-code"
    assert profile.auth_type == "external_process"
    assert profile.base_url == "acp://kimi"
    assert profile.default_aux_model == "k3"
    assert profile.fallback_models == ("k3", "k3-256k")
    assert profile.model_context_lengths == {"k3": 1_048_576, "k3-256k": 262_144}
    assert "OAuth" in profile.display_name
    assert profile.external_command_env_vars == ("KIMI_CODE_CLI_PATH",)
    assert profile.external_preferred_commands == ("~/.kimi-code/bin/kimi",)
    assert profile.external_default_command == "kimi"
    assert profile.external_default_args == ("acp",)
    assert "KIMI_CODE_HOME" in profile.external_process_env_vars
    assert "KIMI_DISABLE_TELEMETRY" in profile.external_process_env_vars
    assert "KIMI_CODE_NO_AUTO_UPDATE" in profile.external_process_env_vars
    assert "HTTPS_PROXY" in profile.external_process_env_vars
    assert "NODE_EXTRA_CA_CERTS" in profile.external_process_env_vars
    assert profile.external_data_root_env_var == "KIMI_CODE_HOME"
    assert profile.external_default_data_root == "~/.kimi-code"
    assert profile.external_login_args == ("login",)
    assert profile.external_login_markers == ("credentials/kimi-code.json",)
    assert profile.external_logout_removes_login_markers is True

    registry = PROVIDER_REGISTRY["kimi-code"]
    assert registry.auth_type == "external_process"
    assert registry.inference_base_url == "acp://kimi"

    descriptor = provider_catalog_by_slug()["kimi-code"]
    assert descriptor.auth_type == "external_process"
    assert descriptor.tab == "accounts"
    assert "OAuth" in descriptor.label


def test_kimi_profile_forwards_private_hermes_session_binding():
    from providers import get_provider_profile

    profile = get_provider_profile("kimi-code")
    extra_body, top_level = profile.build_api_kwargs_extras(
        reasoning_config={"enabled": True, "effort": "high"},
        session_id="hermes-session-123",
    )

    assert extra_body == {}
    assert top_level == {
        "reasoning_config": {"enabled": True, "effort": "high"},
        "_hermes_session_id": "hermes-session-123",
    }


def test_kimi_code_status_distinguishes_installed_cli_from_logged_in_marker(
    tmp_path, monkeypatch
):
    from hermes_cli.auth import (
        get_external_process_provider_status,
        resolve_external_process_provider_credentials,
    )

    home = tmp_path / "home"
    home.mkdir()
    binary = _install_fake_kimi(home)
    monkeypatch.setattr(Path, "home", lambda: home)
    _clear_external_process_overrides(monkeypatch)
    monkeypatch.setenv("PATH", os.defpath)

    status = get_external_process_provider_status("kimi-code")
    assert status["installed"] is True
    assert status["configured"] is False
    assert status["logged_in"] is False
    assert status["command"] == str(binary)
    assert status["resolved_command"] == str(binary)
    assert status["args"] == ["acp"]
    assert status["base_url"] == "acp://kimi"
    assert status["credential_owner"] == "kimi-code-cli"
    assert status["login_markers"] == [
        str(home / ".kimi-code" / "credentials" / "kimi-code.json")
    ]

    marker = home / ".kimi-code" / "credentials" / "kimi-code.json"
    marker.parent.mkdir(parents=True)
    marker.touch()

    status = get_external_process_provider_status("kimi-code")
    assert status["installed"] is True
    assert status["configured"] is True
    assert status["logged_in"] is True

    credentials = resolve_external_process_provider_credentials("kimi-code")
    assert credentials["provider"] == "kimi-code"
    assert credentials["api_key"] == "external-process"
    assert credentials["base_url"] == "acp://kimi"
    assert credentials["command"] == str(binary)
    assert credentials["args"] == ["acp"]
    assert not (
        {"access_token", "refresh_token", "oauth_token", "token"}
        & set(credentials)
    )


def test_kimi_code_status_resolves_marker_from_relocated_data_root(
    tmp_path, monkeypatch
):
    from hermes_cli.auth import get_external_process_provider_status

    home = tmp_path / "home"
    home.mkdir()
    binary = _install_fake_kimi(home)
    provider_home = tmp_path / "relocated-kimi"
    marker = provider_home / "credentials" / "kimi-code.json"
    marker.parent.mkdir(parents=True)
    marker.touch()
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("KIMI_CODE_HOME", str(provider_home))
    _clear_external_process_overrides(monkeypatch)
    monkeypatch.setenv("KIMI_CODE_CLI_PATH", str(binary))

    status = get_external_process_provider_status("kimi-code")

    assert status["installed"] is True
    assert status["logged_in"] is True
    assert status["configured"] is True
    assert status["login_markers"] == [str(marker)]


def test_kimi_code_logout_removes_cli_owned_marker_from_relocated_root(
    tmp_path, monkeypatch, capsys
):
    from hermes_cli.auth import get_external_process_provider_status, logout_command

    home = tmp_path / "home"
    home.mkdir()
    binary = _install_fake_kimi(home)
    provider_home = tmp_path / "relocated-kimi"
    marker = provider_home / "credentials" / "kimi-code.json"
    marker.parent.mkdir(parents=True)
    marker.touch()
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    auth_path = hermes_home / "auth.json"
    original_auth = '{"version": 1, "providers": {}, "credential_pool": {}}\n'
    auth_path.write_text(original_auth, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("KIMI_CODE_HOME", str(provider_home))
    _clear_external_process_overrides(monkeypatch)
    monkeypatch.setenv("KIMI_CODE_CLI_PATH", str(binary))

    assert get_external_process_provider_status("kimi-code")["logged_in"] is True
    logout_command(SimpleNamespace(provider="kimi-code"))

    assert marker.exists() is False
    assert get_external_process_provider_status("kimi-code")["logged_in"] is False
    assert auth_path.read_text(encoding="utf-8") == original_auth
    assert "Logged out of Kimi Code" in capsys.readouterr().out


def test_kimi_code_logout_removes_owned_marker_even_when_cli_is_missing(
    tmp_path, monkeypatch, capsys
):
    from hermes_cli.auth import logout_command

    home = tmp_path / "home"
    marker = home / ".kimi-code" / "credentials" / "kimi-code.json"
    marker.parent.mkdir(parents=True)
    marker.touch()
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HOME", str(home))
    _clear_external_process_overrides(monkeypatch)
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))

    logout_command(SimpleNamespace(provider="kimi-code"))

    assert marker.exists() is False
    assert "Logged out of Kimi Code" in capsys.readouterr().out


def test_auth_add_kimi_code_runs_visible_cli_login_without_prompting_or_pool_store(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    home.mkdir()
    binary = _install_fake_kimi(home)
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    auth_path = hermes_home / "auth.json"
    original_auth = '{"version": 1, "providers": {}, "credential_pool": {}}\n'
    auth_path.write_text(original_auth, encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GH_TOKEN", "must-not-leak")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv("KIMI_ACCESS_TOKEN", "must-not-leak")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "must-not-leak")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "must-not-leak")
    monkeypatch.setenv("AWS_PROFILE", "must-not-leak")
    monkeypatch.setenv("SSH_AUTH_SOCK", str(tmp_path / "agent.sock"))
    provider_home = tmp_path / "relocated-kimi"
    monkeypatch.setenv("KIMI_CODE_HOME", str(provider_home))
    _clear_external_process_overrides(monkeypatch)
    monkeypatch.setenv("KIMI_CODE_CLI_PATH", str(binary))
    monkeypatch.setattr(
        "hermes_cli.auth_commands.masked_secret_prompt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Kimi auth add must never prompt for an API key")
        ),
    )
    monkeypatch.setattr(
        "hermes_cli.auth_commands.load_pool",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Kimi auth add must not load or mutate a credential pool")
        ),
    )
    run = subprocess.CompletedProcess([str(binary), "login"], 0)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        marker = provider_home / "credentials" / "kimi-code.json"
        marker.parent.mkdir(parents=True)
        marker.touch()
        return run

    monkeypatch.setattr("hermes_cli.auth.subprocess.run", fake_run)

    from hermes_cli.auth_commands import auth_add_command

    # Even an explicitly requested API-key flow must be ignored for a
    # CLI-owned external-process provider.
    auth_add_command(
        SimpleNamespace(
            provider="kimi-code",
            auth_type="api-key",
            api_key="must-not-be-stored",
            label="must-not-be-stored",
        )
    )

    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv == [str(binary), "login"]
    assert kwargs["check"] is False
    allowed_from_parent = {
        "PATH",
        "TMPDIR",
        "TMP",
        "TEMP",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "USERPROFILE",
        "HOMEDRIVE",
        "HOMEPATH",
        "APPDATA",
        "LOCALAPPDATA",
        "KIMI_CODE_HOME",
    }
    expected_keys = {name for name in allowed_from_parent if name in os.environ}
    expected_keys.update({"HOME", "HERMES_REAL_HOME"})
    assert set(kwargs["env"]) == expected_keys
    assert kwargs["env"]["HOME"] == str(home)
    assert kwargs["env"]["HERMES_REAL_HOME"] == str(home)
    assert kwargs["env"]["KIMI_CODE_HOME"] == str(provider_home)
    assert auth_path.read_text(encoding="utf-8") == original_auth


def test_kimi_code_login_exit_zero_without_marker_fails_verification(
    tmp_path, monkeypatch, capsys
):
    from hermes_cli.auth import AuthError, run_external_process_provider_login

    home = tmp_path / "home"
    home.mkdir()
    binary = _install_fake_kimi(home)
    provider_home = tmp_path / "kimi-home"
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("KIMI_CODE_HOME", str(provider_home))
    monkeypatch.setenv("KIMI_CODE_CLI_PATH", str(binary))
    monkeypatch.setattr(
        "hermes_cli.auth.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([str(binary), "login"], 0),
    )

    with pytest.raises(AuthError, match="no safe, readable login marker"):
        run_external_process_provider_login("kimi-code")

    assert "login completed" not in capsys.readouterr().out.lower()


@pytest.mark.parametrize("symlink_kind", ["parent", "marker"])
def test_kimi_code_status_and_logout_reject_symlinked_login_marker(
    tmp_path, monkeypatch, symlink_kind
):
    from hermes_cli.auth import (
        get_external_process_provider_status,
        run_external_process_provider_logout,
    )

    home = tmp_path / "home"
    home.mkdir()
    binary = _install_fake_kimi(home)
    provider_home = tmp_path / "kimi-home"
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_marker = outside / "kimi-code.json"
    outside_marker.write_text("outside-must-survive", encoding="utf-8")
    provider_home.mkdir()
    if symlink_kind == "parent":
        (provider_home / "credentials").symlink_to(outside, target_is_directory=True)
    else:
        credentials = provider_home / "credentials"
        credentials.mkdir()
        (credentials / "kimi-code.json").symlink_to(outside_marker)

    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("KIMI_CODE_HOME", str(provider_home))
    monkeypatch.setenv("KIMI_CODE_CLI_PATH", str(binary))

    assert get_external_process_provider_status("kimi-code")["logged_in"] is False
    assert run_external_process_provider_logout("kimi-code") is False
    assert outside_marker.read_text(encoding="utf-8") == "outside-must-survive"


def test_console_auth_add_api_key_override_does_not_require_key_for_kimi():
    from hermes_cli.console_engine import _apply_confirmed_defaults

    args = Namespace(
        auth_action="add",
        auth_type="api-key",
        api_key=None,
        provider="kimi-code",
    )
    _apply_confirmed_defaults(args)


def test_kimi_code_cli_path_override_wins_over_default_location(tmp_path, monkeypatch):
    from hermes_cli.auth import resolve_external_process_provider_credentials

    home = tmp_path / "home"
    home.mkdir()
    _install_fake_kimi(home)
    override = tmp_path / "custom-kimi"
    override.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    override.chmod(0o755)
    monkeypatch.setattr(Path, "home", lambda: home)
    _clear_external_process_overrides(monkeypatch)
    monkeypatch.setenv("KIMI_CODE_CLI_PATH", str(override))

    credentials = resolve_external_process_provider_credentials("kimi-code")
    assert credentials["command"] == str(override)


def test_runtime_provider_rejects_kimi_code_as_model_route(tmp_path, monkeypatch):
    from hermes_cli.runtime_provider import resolve_runtime_provider

    home = tmp_path / "home"
    home.mkdir()
    _install_fake_kimi(home)
    monkeypatch.setattr(Path, "home", lambda: home)
    _clear_external_process_overrides(monkeypatch)
    monkeypatch.setenv("PATH", os.defpath)

    with pytest.raises(ValueError, match="not a Hermes model route"):
        resolve_runtime_provider(requested="kimi-code", target_model="k3")


def test_interactive_kimi_model_flow_fails_closed_without_auth_or_config_change(
    tmp_path, monkeypatch
):
    from hermes_cli.config import load_config, save_config
    from hermes_cli.model_setup_flows import _model_flow_kimi_code

    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    cfg = load_config()
    cfg["model"] = {"provider": "anthropic", "default": "prior-model"}
    save_config(cfg)
    config_path = hermes_home / "config.yaml"
    before = config_path.read_bytes()
    statuses = iter(
        [
            {
                "installed": True,
                "logged_in": False,
                "command": "kimi",
                "resolved_command": "/tmp/kimi",
                "base_url": "acp://kimi",
            },
            {
                "installed": True,
                "logged_in": True,
                "command": "kimi",
                "resolved_command": "/tmp/kimi",
                "base_url": "acp://kimi",
            },
        ]
    )
    login_calls = []
    monkeypatch.setattr(
        "hermes_cli.auth.get_external_process_provider_status",
        lambda _provider: next(statuses),
    )
    monkeypatch.setattr(
        "hermes_cli.auth.run_external_process_provider_login",
        lambda provider: login_calls.append(provider)
        or subprocess.CompletedProcess(["kimi", "login"], 0),
    )
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_external_process_provider_credentials",
        lambda _provider: {"base_url": "acp://kimi", "command": "/tmp/kimi"},
    )
    monkeypatch.setattr(
        "hermes_cli.auth._prompt_model_selection", lambda *_args, **_kwargs: "k3"
    )

    _model_flow_kimi_code({}, current_model="")

    assert login_calls == []
    assert config_path.read_bytes() == before
    assert load_config()["model"] == {
        "provider": "anthropic",
        "default": "prior-model",
    }


def test_interactive_kimi_login_failure_leaves_config_byte_identical(
    tmp_path, monkeypatch
):
    from hermes_cli.auth import AuthError
    from hermes_cli.config import load_config, save_config
    from hermes_cli.model_setup_flows import _model_flow_kimi_code

    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    cfg = load_config()
    cfg["model"] = {"provider": "anthropic", "default": "prior-model"}
    save_config(cfg)
    config_path = hermes_home / "config.yaml"
    before = config_path.read_bytes()
    monkeypatch.setattr(
        "hermes_cli.auth.get_external_process_provider_status",
        lambda _provider: {
            "installed": True,
            "logged_in": False,
            "command": "kimi",
            "resolved_command": "/tmp/kimi",
            "base_url": "acp://kimi",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.auth.run_external_process_provider_login",
        lambda _provider: (_ for _ in ()).throw(
            AuthError(
                "login failed sentinel",
                provider="kimi-code",
                code="external_process_login_failed",
            )
        ),
    )
    monkeypatch.setattr(
        "hermes_cli.auth._prompt_model_selection",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("model picker must not run after failed login")
        ),
    )

    _model_flow_kimi_code({}, current_model="")

    assert config_path.read_bytes() == before
    assert load_config()["model"] == {
        "provider": "anthropic",
        "default": "prior-model",
    }


def test_kimi_model_flow_uses_one_atomic_write_when_role_capability_exists(
    tmp_path, monkeypatch
):
    from hermes_cli.config import load_config, save_config
    from hermes_cli.model_setup_flows import _model_flow_kimi_code
    from providers import get_provider_profile

    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    cfg = load_config()
    cfg["model"] = {"provider": "anthropic", "default": "prior-model"}
    save_config(cfg)
    profile = get_provider_profile("kimi-code")
    monkeypatch.setattr(profile, "external_preserves_system_instructions", True)
    monkeypatch.setattr(
        "hermes_cli.auth.get_external_process_provider_status",
        lambda _provider: {
            "installed": True,
            "logged_in": True,
            "resolved_command": "/tmp/kimi",
            "base_url": "acp://kimi",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_external_process_provider_credentials",
        lambda _provider: {"base_url": "acp://kimi", "command": "/tmp/kimi"},
    )
    monkeypatch.setattr(
        "hermes_cli.auth._prompt_model_selection", lambda *_args, **_kwargs: "k3-256k"
    )

    with patch("hermes_cli.config.save_config", wraps=save_config) as save:
        _model_flow_kimi_code({}, current_model="prior-model")

    assert save.call_count == 1
    assert load_config()["model"] == {
        "provider": "kimi-code",
        "default": "k3-256k",
        "base_url": "acp://kimi",
        "api_mode": "chat_completions",
    }


def test_kimi_model_flow_atomic_write_failure_keeps_original_bytes(
    tmp_path, monkeypatch, capsys
):
    from hermes_cli.config import load_config, save_config
    from hermes_cli.model_setup_flows import _model_flow_kimi_code
    from providers import get_provider_profile

    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    cfg = load_config()
    cfg["model"] = {"provider": "anthropic", "default": "prior-model"}
    save_config(cfg)
    path = hermes_home / "config.yaml"
    before = path.read_bytes()
    profile = get_provider_profile("kimi-code")
    monkeypatch.setattr(profile, "external_preserves_system_instructions", True)
    monkeypatch.setattr(
        "hermes_cli.auth.get_external_process_provider_status",
        lambda _provider: {
            "installed": True,
            "logged_in": True,
            "resolved_command": "/tmp/kimi",
            "base_url": "acp://kimi",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_external_process_provider_credentials",
        lambda _provider: {"base_url": "acp://kimi", "command": "/tmp/kimi"},
    )
    monkeypatch.setattr(
        "hermes_cli.auth._prompt_model_selection", lambda *_args, **_kwargs: "k3"
    )
    monkeypatch.setattr(
        "hermes_cli.config.save_config",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk sentinel")),
    )

    _model_flow_kimi_code({}, current_model="prior-model")

    assert path.read_bytes() == before
    assert "Could not save Kimi model configuration atomically" in capsys.readouterr().out
