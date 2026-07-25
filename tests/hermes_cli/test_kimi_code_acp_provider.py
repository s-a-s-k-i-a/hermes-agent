"""Provider and credential contracts for the Kimi Code ACP backend."""

from __future__ import annotations

from argparse import Namespace
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace


def _install_fake_kimi(home: Path) -> Path:
    binary = home / ".kimi-code" / "bin" / "kimi"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    return binary


def _clear_external_process_overrides(monkeypatch) -> None:
    for name in (
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
    assert profile.fallback_models == ("k3",)
    assert "OAuth" in profile.display_name
    assert profile.external_command_env_vars == ("KIMI_CODE_CLI_PATH",)
    assert profile.external_preferred_commands == ("~/.kimi-code/bin/kimi",)
    assert profile.external_default_command == "kimi"
    assert profile.external_default_args == ("acp",)
    assert profile.external_login_args == ("login",)
    assert profile.external_login_markers == (
        "~/.kimi-code/credentials/kimi-code.json",
    )

    registry = PROVIDER_REGISTRY["kimi-code"]
    assert registry.auth_type == "external_process"
    assert registry.inference_base_url == "acp://kimi"

    descriptor = provider_catalog_by_slug()["kimi-code"]
    assert descriptor.auth_type == "external_process"
    assert descriptor.tab == "accounts"
    assert "OAuth" in descriptor.label


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
    assert kwargs["env"]["HOME"] == str(home)
    for secret_name in ("GH_TOKEN", "OPENAI_API_KEY", "KIMI_ACCESS_TOKEN"):
        assert secret_name not in kwargs["env"]
    assert auth_path.read_text(encoding="utf-8") == original_auth


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


def test_runtime_provider_resolves_kimi_code_without_api_key(tmp_path, monkeypatch):
    from hermes_cli.runtime_provider import resolve_runtime_provider

    home = tmp_path / "home"
    home.mkdir()
    binary = _install_fake_kimi(home)
    monkeypatch.setattr(Path, "home", lambda: home)
    _clear_external_process_overrides(monkeypatch)
    monkeypatch.setenv("PATH", os.defpath)

    runtime = resolve_runtime_provider(requested="kimi-code", target_model="k3")

    assert runtime == {
        "provider": "kimi-code",
        "api_mode": "chat_completions",
        "base_url": "acp://kimi",
        "api_key": "external-process",
        "command": str(binary),
        "args": ["acp"],
        "source": "process",
        "requested_provider": "kimi-code",
    }


def test_interactive_kimi_model_flow_runs_cli_owned_login_before_persisting(
    tmp_path, monkeypatch
):
    from hermes_cli.config import load_config
    from hermes_cli.model_setup_flows import _model_flow_kimi_code

    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
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

    assert login_calls == ["kimi-code"]
    model = load_config()["model"]
    assert model["provider"] == "kimi-code"
    assert model["default"] == "k3"
    assert model["base_url"] == "acp://kimi"


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
