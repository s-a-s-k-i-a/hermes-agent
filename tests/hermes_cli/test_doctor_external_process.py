"""Behavior tests for doctor's external-process provider diagnostics.

The doctor section is driven by ``PROVIDER_REGISTRY`` entries with
``auth_type == "external_process"``.  It must diagnose install/login state
purely structurally: executable resolution + login-marker *existence*.
It must never HTTP-probe the provider and never read marker contents.
"""

from __future__ import annotations

import os
from pathlib import Path


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


def _write_hermes_config(monkeypatch, tmp_path: Path, data: dict) -> Path:
    """Point HERMES_HOME at a temp dir containing the given config.yaml."""
    import yaml

    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir(exist_ok=True)
    (hermes_home / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


def test_doctor_silent_for_unreferenced_missing_provider(tmp_path, monkeypatch, capsys):
    """A registered external-process plugin that is neither configured
    (primary/fallback) nor installed/logged in must not produce warnings —
    an unused missing CLI is not a problem worth surfacing."""
    home = tmp_path / "home"
    home.mkdir()  # no binaries, no markers
    monkeypatch.setattr(Path, "home", lambda: home)
    _clear_external_process_overrides(monkeypatch)
    monkeypatch.setenv("PATH", os.defpath)
    _write_hermes_config(monkeypatch, tmp_path, {})  # nothing referenced

    from hermes_cli.doctor import external_process_provider_checks

    external_process_provider_checks()

    out = capsys.readouterr().out
    assert "Kimi Code" not in out
    assert "Copilot" not in out
    assert "not found" not in out


def test_doctor_reports_external_process_provider_ready(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    home.mkdir()
    _install_fake_kimi(home)
    marker = home / ".kimi-code" / "credentials" / "kimi-code.json"
    marker.parent.mkdir(parents=True)
    marker.write_text('{"sentinel": "MARKER-CONTENT-MUST-NOT-APPEAR"}', encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: home)
    _clear_external_process_overrides(monkeypatch)
    monkeypatch.setenv("PATH", os.defpath)
    monkeypatch.setattr(
        "agent.copilot_acp_client.probe_external_acp_initialize",
        lambda _provider: {
            "protocolVersion": 1,
            "agentInfo": {"name": "Kimi Code CLI", "version": "test"},
            "authMethods": [{"id": "login"}],
            "agentCapabilities": {
                "promptCapabilities": {"image": True, "embeddedContext": True},
                "sessionCapabilities": {"resume": {}},
            },
        },
    )

    from hermes_cli.doctor import external_process_provider_checks

    external_process_provider_checks()

    out = capsys.readouterr().out
    assert "Kimi Code" in out
    assert "✓" in out
    # Structural check only — no HTTP probing, no marker contents.
    assert "http" not in out.lower()
    assert "MARKER-CONTENT-MUST-NOT-APPEAR" not in out
    assert "read-only protocol and capabilities verified" in out


def test_doctor_reports_each_missing_kimi_acp_contract_without_starting_session(
    tmp_path, monkeypatch, capsys
):
    home = tmp_path / "home"
    home.mkdir()
    _install_fake_kimi(home)
    marker = home / ".kimi-code" / "credentials" / "kimi-code.json"
    marker.parent.mkdir(parents=True)
    marker.touch()
    monkeypatch.setattr(Path, "home", lambda: home)
    _clear_external_process_overrides(monkeypatch)
    monkeypatch.setenv("PATH", os.defpath)
    calls = []
    monkeypatch.setattr(
        "agent.copilot_acp_client.probe_external_acp_initialize",
        lambda provider: calls.append(provider)
        or {
            "protocolVersion": 99,
            "agentInfo": {"name": "wrong"},
            "authMethods": [],
            "agentCapabilities": {
                "promptCapabilities": {},
                "sessionCapabilities": {},
            },
        },
    )

    from hermes_cli.doctor import external_process_provider_checks

    external_process_provider_checks()

    out = capsys.readouterr().out
    assert calls == ["kimi-code"]
    assert "incompatible protocolVersion" in out
    assert "unexpected agent identity" in out
    assert "missing auth method: login" in out
    assert "missing prompt capability: image" in out
    assert "missing session capability: resume" in out


def test_doctor_reports_missing_cli_with_override_hint_and_no_secrets(
    tmp_path, monkeypatch, capsys
):
    home = tmp_path / "home"
    home.mkdir()
    # No binary anywhere — but kimi-code is referenced as the configured
    # primary, so doctor must warn.  A marker file with sentinel content
    # exists; doctor must name the env override var as remedy and must not
    # read (let alone print) marker contents.
    marker = home / ".kimi-code" / "credentials" / "kimi-code.json"
    marker.parent.mkdir(parents=True)
    marker.write_text('{"sentinel": "MARKER-CONTENT-MUST-NOT-APPEAR"}', encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: home)
    _clear_external_process_overrides(monkeypatch)
    monkeypatch.setenv("PATH", os.defpath)
    _write_hermes_config(
        monkeypatch,
        tmp_path,
        {"model": {"provider": "kimi-code", "default": "k3"}},
    )

    from hermes_cli.doctor import external_process_provider_checks

    external_process_provider_checks()

    out = capsys.readouterr().out
    assert "Kimi Code" in out
    assert "KIMI_CODE_CLI_PATH" in out
    assert "MARKER-CONTENT-MUST-NOT-APPEAR" not in out


def test_doctor_warns_for_missing_cli_referenced_as_fallback(
    tmp_path, monkeypatch, capsys
):
    home = tmp_path / "home"
    home.mkdir()  # no binary, no marker
    monkeypatch.setattr(Path, "home", lambda: home)
    _clear_external_process_overrides(monkeypatch)
    monkeypatch.setenv("PATH", os.defpath)
    _write_hermes_config(
        monkeypatch,
        tmp_path,
        {
            "model": {"provider": "openrouter", "default": "gpt-5.4"},
            "fallback_providers": [{"provider": "kimi-code", "model": "k3"}],
        },
    )

    from hermes_cli.doctor import external_process_provider_checks

    external_process_provider_checks()

    out = capsys.readouterr().out
    assert "Kimi Code" in out
    assert "not found" in out


def test_doctor_recognizes_provider_alias_referenced_as_primary(
    tmp_path, monkeypatch, capsys
):
    """`github-copilot-acp` is a supported alias for copilot-acp — a config
    referencing the alias must still surface the provider's diagnostics."""
    home = tmp_path / "home"
    home.mkdir()  # copilot CLI not installed
    monkeypatch.setattr(Path, "home", lambda: home)
    _clear_external_process_overrides(monkeypatch)
    monkeypatch.setenv("PATH", os.defpath)
    _write_hermes_config(
        monkeypatch,
        tmp_path,
        {"model": {"provider": "github-copilot-acp", "default": "copilot-acp"}},
    )

    from hermes_cli.doctor import external_process_provider_checks

    external_process_provider_checks()

    out = capsys.readouterr().out
    assert "Copilot" in out
    assert "not found" in out


def test_doctor_recognizes_provider_alias_referenced_as_fallback(
    tmp_path, monkeypatch, capsys
):
    """`copilot-acp-agent` (profile alias) in the fallback chain must count
    as a reference to copilot-acp — alias dedup must not swallow it."""
    home = tmp_path / "home"
    home.mkdir()  # copilot CLI not installed
    monkeypatch.setattr(Path, "home", lambda: home)
    _clear_external_process_overrides(monkeypatch)
    monkeypatch.setenv("PATH", os.defpath)
    _write_hermes_config(
        monkeypatch,
        tmp_path,
        {
            "model": {"provider": "openrouter", "default": "gpt-5.4"},
            "fallback_providers": [
                {"provider": "copilot-acp-agent", "model": "copilot-acp"}
            ],
        },
    )

    from hermes_cli.doctor import external_process_provider_checks

    external_process_provider_checks()

    out = capsys.readouterr().out
    assert "Copilot" in out
    assert "not found" in out


def test_doctor_reports_installed_but_logged_out_with_login_hint(
    tmp_path, monkeypatch, capsys
):
    home = tmp_path / "home"
    home.mkdir()
    _install_fake_kimi(home)  # binary present, no login marker
    monkeypatch.setattr(Path, "home", lambda: home)
    _clear_external_process_overrides(monkeypatch)
    monkeypatch.setenv("PATH", os.defpath)

    from hermes_cli.doctor import external_process_provider_checks

    external_process_provider_checks()

    out = capsys.readouterr().out
    assert "Kimi Code" in out
    assert "no login marker" in out
    assert "hermes auth add kimi-code" in out


def test_doctor_command_paths_are_redacted_before_display(
    tmp_path, monkeypatch, capsys
):
    """External command values are never user-visible in doctor output —
    the central redactor intentionally skips token-shaped path segments,
    so the whole resolved/configured command path must be omitted, not
    merely scrubbed."""
    token = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
    home = tmp_path / "home"
    home.mkdir()
    binary = tmp_path / f"kimi-{token}"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    marker = home / ".kimi-code" / "credentials" / "kimi-code.json"
    marker.parent.mkdir(parents=True)
    marker.touch()
    monkeypatch.setattr(Path, "home", lambda: home)
    _clear_external_process_overrides(monkeypatch)
    monkeypatch.setenv("PATH", os.defpath)
    monkeypatch.setenv("KIMI_CODE_CLI_PATH", str(binary))
    _write_hermes_config(
        monkeypatch,
        tmp_path,
        {"model": {"provider": "kimi-code", "default": "k3"}},
    )

    from hermes_cli.doctor import external_process_provider_checks

    external_process_provider_checks()

    out = capsys.readouterr().out
    assert "Kimi Code" in out
    assert "CLI found" in out
    # The full command path (including its token-shaped segment) must be
    # entirely absent from user-visible output.
    assert token not in out
    assert str(binary) not in out


def test_doctor_ready_provider_without_login_markers_does_not_claim_session_marker(
    tmp_path, monkeypatch, capsys
):
    """copilot-acp declares no login markers — its login state cannot be
    verified structurally, so doctor must not claim verified readiness
    (no ✓, no 'session marker present'), even though the legacy status
    helper reports logged_in for marker-less installed CLIs."""
    home = tmp_path / "home"
    home.mkdir()
    binary = tmp_path / "copilot"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setattr(Path, "home", lambda: home)
    _clear_external_process_overrides(monkeypatch)
    monkeypatch.setenv("PATH", os.defpath)
    monkeypatch.setenv("HERMES_COPILOT_ACP_COMMAND", str(binary))
    monkeypatch.setenv("COPILOT_CLI_PATH", str(binary))

    from hermes_cli.doctor import external_process_provider_checks

    external_process_provider_checks()

    out = capsys.readouterr().out
    assert "Copilot" in out
    assert "CLI found" in out
    assert "not verifiable" in out
    assert "✓" not in out
    assert "session marker present" not in out


def test_doctor_status_check_exception_text_is_redacted(tmp_path, monkeypatch, capsys):
    """Exception messages from status checks flow into user-visible doctor
    output — they must pass through redact_sensitive_text so a secret that
    leaked into an exception string is scrubbed before printing."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    _clear_external_process_overrides(monkeypatch)
    monkeypatch.setenv("PATH", os.defpath)
    _write_hermes_config(
        monkeypatch,
        tmp_path,
        {"model": {"provider": "kimi-code", "default": "k3"}},
    )

    secret = "sk-proj-SENTINEL1234567890abcdefSENTINEL1234567890"

    import hermes_cli.doctor as doctor_mod

    def _boom(provider_id):
        raise RuntimeError(f"launch failed with env OPENAI_API_KEY={secret}")

    import hermes_cli.auth as auth_mod

    monkeypatch.setattr(
        auth_mod, "get_external_process_provider_status", _boom
    )

    doctor_mod.external_process_provider_checks()  # must not raise

    out = capsys.readouterr().out
    assert "status check failed" in out or "check failed" in out
    assert secret not in out


def test_doctor_external_process_never_opens_network(tmp_path, monkeypatch, capsys):
    """The diagnostic must stay structural — any network attempt is a bug."""
    home = tmp_path / "home"
    home.mkdir()
    _install_fake_kimi(home)
    marker = home / ".kimi-code" / "credentials" / "kimi-code.json"
    marker.parent.mkdir(parents=True)
    marker.touch()
    monkeypatch.setattr(Path, "home", lambda: home)
    _clear_external_process_overrides(monkeypatch)
    monkeypatch.setenv("PATH", os.defpath)

    network_calls: list[tuple] = []

    def _no_network(*args, **kwargs):
        network_calls.append((args, kwargs))
        raise AssertionError("external-process doctor checks must not open the network")

    import socket
    import urllib.request

    monkeypatch.setattr(socket.socket, "connect", _no_network)
    monkeypatch.setattr(urllib.request, "urlopen", _no_network)
    try:
        import httpx

        monkeypatch.setattr(httpx, "get", _no_network)
        monkeypatch.setattr(httpx, "request", _no_network)
    except ImportError:
        pass

    from hermes_cli.doctor import external_process_provider_checks

    external_process_provider_checks()  # must not raise

    out = capsys.readouterr().out
    assert "Kimi Code" in out
    assert len(network_calls) == 0, (
        f"expected 0 network attempts, saw {len(network_calls)}: {network_calls!r}"
    )


def test_run_doctor_invokes_external_process_checks(tmp_path, monkeypatch):
    """run_doctor must include the external-process section (wiring contract)."""
    import contextlib
    import io
    import sys
    import types
    from argparse import Namespace

    import hermes_cli.doctor as doctor_mod

    home = tmp_path / ".hermes"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", tmp_path / "project")
    monkeypatch.setattr(doctor_mod, "_DHH", str(home))
    (tmp_path / "project").mkdir(exist_ok=True)

    fake_model_tools = types.SimpleNamespace(
        check_tool_availability=lambda *a, **kw: ([], []),
        TOOLSET_REQUIREMENTS={},
    )
    monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)
    from hermes_cli import auth as _auth_mod

    monkeypatch.setattr(_auth_mod, "get_nous_auth_status", lambda: {})
    monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
    monkeypatch.setattr(_auth_mod, "get_xai_oauth_auth_status", lambda: {})

    calls = []
    monkeypatch.setattr(
        doctor_mod,
        "external_process_provider_checks",
        lambda: calls.append(True),
    )

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False))

    assert calls, "run_doctor did not invoke external_process_provider_checks"


def _run_full_doctor(
    tmp_path,
    monkeypatch,
    *,
    provider: str,
    write_env: bool,
    setup_home,
):
    """Run full run_doctor with the given provider as configured primary.

    ``setup_home(home)`` prepares fake binaries/markers inside the mocked
    ``Path.home()``.  Suppression of API-key demands in the ``.env`` section
    must follow only from the provider *contract* (``auth_type ==
    "external_process"``), never from any login/readiness state — auth
    readiness is diagnosed separately by ``external_process_provider_checks``.
    """
    import contextlib
    import io
    import sys
    import types
    from argparse import Namespace

    import yaml

    import hermes_cli.doctor as doctor_mod

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    _clear_external_process_overrides(monkeypatch)
    monkeypatch.setenv("PATH", os.defpath)
    setup_home(home, monkeypatch)

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"model": {"provider": provider, "default": "m1"}}),
        encoding="utf-8",
    )
    if write_env:
        # .env exists but holds no provider API key — fine for an
        # external-process primary that authenticates via its own CLI.
        (hermes_home / ".env").write_text("# no keys\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(doctor_mod, "HERMES_HOME", hermes_home)
    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", tmp_path / "project")
    monkeypatch.setattr(doctor_mod, "_DHH", str(hermes_home))
    (tmp_path / "project").mkdir(exist_ok=True)

    fake_model_tools = types.SimpleNamespace(
        check_tool_availability=lambda *a, **kw: ([], []),
        TOOLSET_REQUIREMENTS={},
    )
    monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)
    from hermes_cli import auth as _auth_mod

    monkeypatch.setattr(_auth_mod, "get_nous_auth_status", lambda: {})
    monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
    monkeypatch.setattr(_auth_mod, "get_xai_oauth_auth_status", lambda: {})

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False))
    return buf.getvalue()


def _setup_kimi_with_marker(home: Path, monkeypatch) -> None:
    _install_fake_kimi(home)
    marker = home / ".kimi-code" / "credentials" / "kimi-code.json"
    marker.parent.mkdir(parents=True)
    marker.touch()


def _run_doctor_with_kimi_external_process_primary(
    tmp_path, monkeypatch, *, write_env: bool
):
    """Run run_doctor with kimi-code as external-process primary."""
    return _run_full_doctor(
        tmp_path,
        monkeypatch,
        provider="kimi-code",
        write_env=write_env,
        setup_home=_setup_kimi_with_marker,
    )


def test_env_without_api_key_is_not_a_blocker_for_external_process_primary(
    tmp_path, monkeypatch
):
    """An external-process primary does not use Hermes API keys — doctor's
    .env section must not warn or demand key configuration."""
    out = _run_doctor_with_kimi_external_process_primary(
        tmp_path, monkeypatch, write_env=True
    )
    assert "No API key found" not in out
    assert "configure API keys" not in out


def test_missing_env_is_optional_for_external_process_primary(
    tmp_path, monkeypatch
):
    """With an external-process primary a missing .env is optional, not a
    failure demanding setup."""
    out = _run_doctor_with_kimi_external_process_primary(
        tmp_path, monkeypatch, write_env=False
    )
    assert ".env file missing" not in out
    assert "Run 'hermes setup' to create .env" not in out


def test_run_doctor_markerless_copilot_primary_is_neutral_about_session(
    tmp_path, monkeypatch
):
    """Full run_doctor with a marker-less copilot-acp primary (fake binary
    installed, no login markers declared, no .env):

    - the .env section must stay neutral — no API-key demand, but also no
      claim that a verified OAuth session exists (the legacy status helper
      reports logged_in == installed for marker-less CLIs, which verifies
      nothing);
    - the provider section must separately surface that the login state is
      not verifiable.
    """

    def _setup_copilot(home: Path, monkeypatch) -> None:
        binary = tmp_path / "copilot"
        binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        binary.chmod(0o755)
        monkeypatch.setenv("HERMES_COPILOT_ACP_COMMAND", str(binary))
        monkeypatch.setenv("COPILOT_CLI_PATH", str(binary))

    out = _run_full_doctor(
        tmp_path,
        monkeypatch,
        provider="github-copilot-acp",
        write_env=False,
        setup_home=_setup_copilot,
    )

    # No verified-session / readiness claim anywhere in doctor output.
    assert "OAuth session" not in out
    assert "session marker present" not in out
    # .env stays neutral: not demanded, not framed as a blocker.
    assert "No API key found" not in out
    assert "configure API keys" not in out
    assert ".env file missing" not in out
    assert "Run 'hermes setup' to create .env" not in out
    assert "optional" in out
    # Auth readiness is reported separately — and honestly.
    assert "login state not verifiable" in out
