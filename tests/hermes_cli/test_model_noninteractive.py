"""Non-interactive provider/model selection contracts (B2/B3)."""

from __future__ import annotations


def test_resolve_noninteractive_selection_kimi_code_defaults_model_k3():
    from hermes_cli.model_noninteractive import resolve_noninteractive_selection

    selection = resolve_noninteractive_selection("kimi-code", None)

    assert selection["provider"] == "kimi-code"
    assert selection["model"] == "k3"
    assert selection["base_url"] == "acp://kimi"
    assert selection["api_mode"] == "chat_completions"
    assert selection["needs_api_key"] is False


def test_resolve_noninteractive_selection_rejects_unknown_provider():
    import pytest

    from hermes_cli.model_noninteractive import (
        NonInteractiveSelectionError,
        resolve_noninteractive_selection,
    )

    with pytest.raises(NonInteractiveSelectionError) as excinfo:
        resolve_noninteractive_selection("definitely-not-a-provider", "k3")

    assert excinfo.value.exit_code == 2
    assert "definitely-not-a-provider" in str(excinfo.value)


def test_resolve_noninteractive_selection_rejects_unknown_model_for_provider():
    import pytest

    from hermes_cli.model_noninteractive import (
        NonInteractiveSelectionError,
        resolve_noninteractive_selection,
    )

    with pytest.raises(NonInteractiveSelectionError) as excinfo:
        resolve_noninteractive_selection("kimi-code", "gpt-4o")

    assert excinfo.value.exit_code == 2
    message = str(excinfo.value)
    assert "gpt-4o" in message
    assert "k3" in message


def _install_fake_kimi_home(tmp_path, monkeypatch, *, logged_in=True):
    """Fake CLI binary + optional login marker under an isolated home."""
    import os
    from pathlib import Path

    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    binary = home / ".kimi-code" / "bin" / "kimi"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    if logged_in:
        marker = home / ".kimi-code" / "credentials" / "kimi-code.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
    monkeypatch.setattr(Path, "home", lambda: home)
    for name in ("KIMI_CODE_CLI_PATH", "COPILOT_CLI_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PATH", os.defpath)
    return home


def test_apply_noninteractive_model_selection_writes_config_without_credentials(
    tmp_path, monkeypatch
):
    import json
    import os
    from pathlib import Path

    from hermes_cli.config import load_config
    from hermes_cli.model_noninteractive import (
        apply_noninteractive_model_selection,
        resolve_noninteractive_selection,
    )

    _install_fake_kimi_home(tmp_path, monkeypatch)
    auth_path = Path(os.environ["HERMES_HOME"]) / "auth.json"
    original_auth = json.dumps(
        {"version": 1, "providers": {}, "credential_pool": {}}
    )
    auth_path.write_text(original_auth, encoding="utf-8")

    selection = resolve_noninteractive_selection("kimi-code", None)
    apply_noninteractive_model_selection(selection)

    config = load_config()
    model = config["model"]
    assert model["default"] == "k3"
    assert model["provider"] == "kimi-code"
    assert model["base_url"] == "acp://kimi"
    assert model["api_mode"] == "chat_completions"
    assert "api_key" not in model
    assert "api" not in model
    stored_auth = json.loads(auth_path.read_text(encoding="utf-8"))
    assert stored_auth["providers"] == {}
    assert stored_auth["credential_pool"] == {}


def test_apply_noninteractive_external_process_missing_cli_exits_3_with_hint(
    tmp_path, monkeypatch
):
    import pytest

    from hermes_cli.model_noninteractive import (
        NonInteractiveSelectionError,
        apply_noninteractive_model_selection,
        resolve_noninteractive_selection,
    )

    home = _install_fake_kimi_home(tmp_path, monkeypatch)
    (home / ".kimi-code" / "bin" / "kimi").unlink()

    selection = resolve_noninteractive_selection("kimi-code", "k3")
    with pytest.raises(NonInteractiveSelectionError) as excinfo:
        apply_noninteractive_model_selection(selection)

    assert excinfo.value.exit_code == 3
    assert "KIMI_CODE_CLI_PATH" in str(excinfo.value)


def test_apply_noninteractive_external_process_logged_out_exits_3_with_auth_hint(
    tmp_path, monkeypatch
):
    import pytest

    from hermes_cli.model_noninteractive import (
        NonInteractiveSelectionError,
        apply_noninteractive_model_selection,
        resolve_noninteractive_selection,
    )

    _install_fake_kimi_home(tmp_path, monkeypatch, logged_in=False)

    selection = resolve_noninteractive_selection("kimi-code", "k3")
    with pytest.raises(NonInteractiveSelectionError) as excinfo:
        apply_noninteractive_model_selection(selection)

    assert excinfo.value.exit_code == 3
    assert "hermes auth add kimi-code" in str(excinfo.value)


def _build_cli_parser_with_model():
    import argparse

    from hermes_cli.main import cmd_model
    from hermes_cli.subcommands.model import build_model_parser

    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    build_model_parser(subparsers, cmd_model=cmd_model)
    return parser


def test_model_parser_accepts_provider_and_model_flags():
    parser = _build_cli_parser_with_model()

    args = parser.parse_args(
        ["model", "--provider", "kimi-code", "--model", "k3"]
    )

    assert args.provider == "kimi-code"
    assert args.model == "k3"


def test_cmd_model_noninteractive_kimi_sets_default_without_tty(
    tmp_path, monkeypatch, capsys
):
    import builtins
    import sys

    from hermes_cli.config import load_config
    from hermes_cli.main import cmd_model

    _install_fake_kimi_home(tmp_path, monkeypatch)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(
        builtins,
        "input",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("non-interactive model must never prompt")
        ),
    )

    parser = _build_cli_parser_with_model()
    args = parser.parse_args(
        ["model", "--provider", "kimi-code", "--model", "k3"]
    )
    cmd_model(args)

    out = capsys.readouterr().out
    assert "k3" in out
    assert "kimi-code" in out
    config = load_config()
    assert config["model"]["default"] == "k3"
    assert config["model"]["provider"] == "kimi-code"
    assert "api_key" not in config["model"]


def test_cmd_model_noninteractive_provider_only_uses_default_model(
    tmp_path, monkeypatch
):
    import sys

    from hermes_cli.config import load_config
    from hermes_cli.main import cmd_model

    _install_fake_kimi_home(tmp_path, monkeypatch)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    parser = _build_cli_parser_with_model()
    args = parser.parse_args(["model", "--provider", "kimi-code"])
    cmd_model(args)

    config = load_config()
    assert config["model"]["default"] == "k3"
    assert config["model"]["provider"] == "kimi-code"


def test_cmd_model_noninteractive_unknown_model_exits_2(
    tmp_path, monkeypatch, capsys
):
    import sys

    import pytest

    from hermes_cli.main import cmd_model

    _install_fake_kimi_home(tmp_path, monkeypatch)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    parser = _build_cli_parser_with_model()
    args = parser.parse_args(
        ["model", "--provider", "kimi-code", "--model", "bogus"]
    )
    with pytest.raises(SystemExit) as excinfo:
        cmd_model(args)

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "bogus" in err
    assert "k3" in err


def test_cmd_model_noninteractive_logged_out_exits_3(tmp_path, monkeypatch, capsys):
    import sys

    import pytest

    from hermes_cli.main import cmd_model

    _install_fake_kimi_home(tmp_path, monkeypatch, logged_in=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    parser = _build_cli_parser_with_model()
    args = parser.parse_args(
        ["model", "--provider", "kimi-code", "--model", "k3"]
    )
    with pytest.raises(SystemExit) as excinfo:
        cmd_model(args)

    assert excinfo.value.code == 3
    assert "hermes auth add kimi-code" in capsys.readouterr().err


def test_apply_noninteractive_model_selection_keeps_auth_json_byte_identical(
    tmp_path, monkeypatch
):
    """B2: switching models must never rewrite auth.json in any way.

    ``config.model.provider`` is authoritative; external OAuth credentials
    (and ``active_provider``, unknown future fields, and exact formatting)
    belong to auth.json's owners, not to the model switch. Byte-compare the
    whole file, not just selected keys.
    """
    import json
    import os
    from pathlib import Path

    from hermes_cli.model_noninteractive import (
        apply_noninteractive_model_selection,
        resolve_noninteractive_selection,
    )

    _install_fake_kimi_home(tmp_path, monkeypatch)
    auth_path = Path(os.environ["HERMES_HOME"]) / "auth.json"
    sentinel_store = {
        "version": 1,
        "active_provider": "anthropic",
        "providers": {
            "anthropic": {
                "type": "oauth",
                "refresh_token": "sentinel-refresh-token",
                "access_token": "sentinel-access-token",
            }
        },
        "credential_pool": {"anthropic": ["sentinel-pool-entry"]},
        "unknown_future_field": {"keep": ["me", "intact"]},
    }
    # Deliberate non-default formatting: indent=4 + trailing newline. A
    # load→dump round-trip would normalize this away, so byte equality also
    # proves the file was never rewritten.
    auth_path.write_text(
        json.dumps(sentinel_store, indent=4) + "\n", encoding="utf-8"
    )
    before = auth_path.read_bytes()

    selection = resolve_noninteractive_selection("kimi-code", None)
    apply_noninteractive_model_selection(selection)

    assert auth_path.read_bytes() == before


def test_missing_cli_error_never_echoes_resolved_command_path(
    tmp_path, monkeypatch
):
    """B2: the exit-3 message may name the provider id and env VAR NAMES,
    never the resolved ``status.command``/override value (a user-controlled,
    token-like path)."""
    import pytest

    from hermes_cli.model_noninteractive import (
        NonInteractiveSelectionError,
        apply_noninteractive_model_selection,
        resolve_noninteractive_selection,
    )

    home = _install_fake_kimi_home(tmp_path, monkeypatch)
    (home / ".kimi-code" / "bin" / "kimi").unlink()
    secret_path = str(
        tmp_path / "secret-token-dir-c8f1" / "kimi-sentinel-bin-7ab2"
    )
    monkeypatch.setenv("KIMI_CODE_CLI_PATH", secret_path)

    selection = resolve_noninteractive_selection("kimi-code", "k3")
    with pytest.raises(NonInteractiveSelectionError) as excinfo:
        apply_noninteractive_model_selection(selection)

    assert excinfo.value.exit_code == 3
    message = str(excinfo.value)
    assert secret_path not in message
    assert "secret-token-dir-c8f1" not in message
    assert "kimi-sentinel-bin-7ab2" not in message
    assert "kimi-code" in message
    assert "KIMI_CODE_CLI_PATH" in message


def test_omitted_model_uses_canonical_cost_safe_resolver(monkeypatch):
    """B3: an omitted --model must resolve through
    ``get_default_model_for_provider`` (cost-safe), not blind
    ``_PROVIDER_MODELS[0]``. external_process ``default_aux_model`` stays
    contractual and takes priority over the resolver."""
    import hermes_cli.models as models_mod
    from hermes_cli.model_noninteractive import resolve_noninteractive_selection

    seen = []

    def fake_resolver(provider_id):
        seen.append(provider_id)
        return "sentinel-cost-safe-default"

    monkeypatch.setattr(
        models_mod, "get_default_model_for_provider", fake_resolver
    )

    selection = resolve_noninteractive_selection("anthropic", None)
    assert selection["model"] == "sentinel-cost-safe-default"
    assert seen == ["anthropic"]

    # external_process contract: profile.default_aux_model wins, the
    # resolver is not consulted for kimi-code.
    seen.clear()
    kimi = resolve_noninteractive_selection("kimi-code", None)
    assert kimi["model"] == "k3"
    assert seen == []


def test_omitted_model_without_safe_default_exits_2(monkeypatch):
    """B3: when no cost-safe default exists, require an explicit --model
    (exit 2) — never fall back to the flagship-first catalog entry."""
    import pytest

    import hermes_cli.models as models_mod
    from hermes_cli.model_noninteractive import (
        NonInteractiveSelectionError,
        resolve_noninteractive_selection,
    )

    monkeypatch.setattr(
        models_mod, "get_default_model_for_provider", lambda _p: ""
    )

    with pytest.raises(NonInteractiveSelectionError) as excinfo:
        resolve_noninteractive_selection("anthropic", None)

    assert excinfo.value.exit_code == 2
    assert "--model" in str(excinfo.value)


def test_cmd_model_without_flags_still_requires_tty(monkeypatch):
    import sys

    import pytest

    from hermes_cli.main import cmd_model

    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    parser = _build_cli_parser_with_model()
    args = parser.parse_args(["model"])
    with pytest.raises(SystemExit) as excinfo:
        cmd_model(args)

    assert excinfo.value.code == 1


# -- A1: alias matrix must go through the canonical auth-aware resolver -------


def test_resolver_accepts_canonical_alias_matrix(monkeypatch):
    """A1: aliases valid in the auth-aware canonical resolver must not be
    rejected by a second (models.dev-shaped) normalization pass.

    Behavior contract: whatever ``hermes_cli.auth.resolve_provider`` accepts
    as an explicit provider, ``resolve_noninteractive_selection`` accepts too
    and yields the same canonical id.
    """
    import hermes_cli.models as models_mod
    from hermes_cli.model_noninteractive import resolve_noninteractive_selection

    # Deterministic, offline default-model resolution — the contract under
    # test is provider canonicalization, not model choice.
    monkeypatch.setattr(
        models_mod,
        "get_default_model_for_provider",
        lambda _p: "sentinel-default-model",
    )

    matrix = {
        "google": "gemini",
        "github": "copilot",
        "copilot-acp-agent": "copilot-acp",
        "codex": "openai-codex",
        "openai_codex": "openai-codex",
        "opencode": "opencode-zen",
        # Canonical ids must resolve to themselves (reviewer matrix).
        "copilot": "copilot",
        "kilocode": "kilocode",
        "kimi-coding": "kimi-coding",
        "kimi-coding-cn": "kimi-coding-cn",
        "opencode-zen": "opencode-zen",
    }
    for alias, canonical in matrix.items():
        selection = resolve_noninteractive_selection(alias, None)
        assert selection["provider"] == canonical, (
            f"alias {alias!r} resolved to {selection['provider']!r}, "
            f"expected {canonical!r}"
        )


def test_resolver_rejects_empty_and_whitespace_provider_exit_2():
    """A1: empty/whitespace provider is a usage error (exit 2) — it must
    never silently fall through to 'auto' credential detection."""
    import pytest

    from hermes_cli.model_noninteractive import (
        NonInteractiveSelectionError,
        resolve_noninteractive_selection,
    )

    for raw in ("", "   ", "\t", None):
        with pytest.raises(NonInteractiveSelectionError) as excinfo:
            resolve_noninteractive_selection(raw, "k3")
        assert excinfo.value.exit_code == 2, repr(raw)


def test_resolver_rejects_explicit_auto_exit_2():
    """A1: non-interactive selection requires a concrete provider —
    'auto' would resolve from ambient credentials, which is prompt-free
    but not deterministic."""
    import pytest

    from hermes_cli.model_noninteractive import (
        NonInteractiveSelectionError,
        resolve_noninteractive_selection,
    )

    with pytest.raises(NonInteractiveSelectionError) as excinfo:
        resolve_noninteractive_selection("auto", None)
    assert excinfo.value.exit_code == 2


# -- A2: flag PRESENCE (not truthiness) selects the non-interactive path -----


def _run_cmd_model_with_tty_spy(monkeypatch, argv):
    """Run cmd_model with a spy on _require_tty and a poisoned input()."""
    import builtins
    import sys

    import pytest

    from hermes_cli import main as main_mod

    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    called = []
    real_require_tty = main_mod._require_tty

    def spy(command_name):
        called.append(command_name)
        real_require_tty(command_name)

    monkeypatch.setattr(main_mod, "_require_tty", spy)
    monkeypatch.setattr(
        builtins,
        "input",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("non-interactive model must never prompt")
        ),
    )

    parser = _build_cli_parser_with_model()
    args = parser.parse_args(argv)
    with pytest.raises(SystemExit) as excinfo:
        main_mod.cmd_model(args)
    return excinfo.value.code, called


def test_cmd_model_empty_provider_flag_exits_2_never_tty(monkeypatch, capsys):
    code, tty_calls = _run_cmd_model_with_tty_spy(
        monkeypatch, ["model", "--provider", ""]
    )

    assert code == 2
    assert tty_calls == []
    assert "--provider" in capsys.readouterr().err


def test_cmd_model_empty_model_flag_exits_2_never_tty(monkeypatch, capsys):
    code, tty_calls = _run_cmd_model_with_tty_spy(
        monkeypatch, ["model", "--model", ""]
    )

    assert code == 2
    assert tty_calls == []
    assert "--model" in capsys.readouterr().err


# -- A3: bare custom/ollama/vllm rejected before ANY mutation ----------------


def test_cmd_model_bare_custom_rejected_exit_2_config_untouched(
    tmp_path, monkeypatch, capsys
):
    """A3: until a credential-preserving contract exists, non-interactive
    selection of the custom provider (and its local aliases) must exit 2
    with a pointer to the interactive/custom-config flow — and must not
    touch config.yaml at all (bytes + semantics)."""
    import os
    import sys
    from pathlib import Path

    import pytest

    from hermes_cli.config import load_config, save_config
    from hermes_cli.main import cmd_model

    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    cfg = load_config()
    cfg["model"] = {
        "default": "llama3.3-70b",
        "provider": "custom",
        "base_url": "http://localhost:11434/v1",
        "api_key": "sk-local-sentinel",
        "api_mode": "chat_completions",
    }
    save_config(cfg)
    config_path = Path(os.environ["HERMES_HOME"]) / "config.yaml"
    before = config_path.read_bytes()

    parser = _build_cli_parser_with_model()
    for prov in ("custom", "ollama", "vllm"):
        args = parser.parse_args(["model", "--provider", prov])
        with pytest.raises(SystemExit) as excinfo:
            cmd_model(args)
        assert excinfo.value.code == 2, prov
        err = capsys.readouterr().err
        assert "hermes model" in err, prov

    assert config_path.read_bytes() == before
    after_model = load_config()["model"]
    assert after_model["base_url"] == "http://localhost:11434/v1"
    assert after_model["api_key"] == "sk-local-sentinel"
    assert after_model["default"] == "llama3.3-70b"


# -- A4: exactly ONE atomic save_config, no partial writes -------------------


def test_apply_noninteractive_model_selection_single_save_config_call(
    tmp_path, monkeypatch
):
    """A4: apply must assemble the whole model-section update in memory and
    persist it with exactly one save_config call — no separate
    _save_model_choice write."""
    import hermes_cli.config as config_mod
    from hermes_cli.config import load_config, save_config
    from hermes_cli.model_noninteractive import (
        apply_noninteractive_model_selection,
        resolve_noninteractive_selection,
    )

    _install_fake_kimi_home(tmp_path, monkeypatch)
    # Prime config on disk so lazy first-write/migration saves don't count.
    save_config(load_config())

    calls = []
    real_save = config_mod.save_config

    def spy(cfg, *args, **kwargs):
        calls.append(cfg)
        return real_save(cfg, *args, **kwargs)

    monkeypatch.setattr(config_mod, "save_config", spy)

    selection = resolve_noninteractive_selection("kimi-code", None)
    apply_noninteractive_model_selection(selection)

    assert len(calls) == 1
    model = load_config()["model"]
    assert model["default"] == "k3"
    assert model["provider"] == "kimi-code"
    assert model["base_url"] == "acp://kimi"
    assert model["api_mode"] == "chat_completions"


def test_apply_noninteractive_model_selection_failed_save_leaves_disk_intact(
    tmp_path, monkeypatch
):
    """A4: when the single save_config raises, the on-disk config must be
    byte-identical — no partial model.default-only write may precede it."""
    import os
    from pathlib import Path

    import pytest

    import hermes_cli.config as config_mod
    from hermes_cli.config import load_config, save_config
    from hermes_cli.model_noninteractive import (
        apply_noninteractive_model_selection,
        resolve_noninteractive_selection,
    )

    _install_fake_kimi_home(tmp_path, monkeypatch)
    cfg = load_config()
    cfg["model"] = {"default": "prior-model", "provider": "anthropic"}
    save_config(cfg)
    config_path = Path(os.environ["HERMES_HOME"]) / "config.yaml"
    before = config_path.read_bytes()

    def exploding_save(*_args, **_kwargs):
        raise RuntimeError("disk full (sentinel)")

    monkeypatch.setattr(config_mod, "save_config", exploding_save)

    selection = resolve_noninteractive_selection("kimi-code", None)
    with pytest.raises(RuntimeError, match="disk full"):
        apply_noninteractive_model_selection(selection)

    assert config_path.read_bytes() == before
    prior = load_config()["model"]
    assert prior["default"] == "prior-model"
    assert prior["provider"] == "anthropic"


# -- B: prerequisite validation must mirror runtime credential contracts ------


def test_apply_noninteractive_openrouter_without_credentials_exits_3_untouched(
    tmp_path, monkeypatch
):
    """Missing OpenRouter auth is exit 3 and leaves config byte-identical."""
    from pathlib import Path

    import pytest

    from hermes_cli.config import load_config, save_config
    from hermes_cli.model_noninteractive import (
        NonInteractiveSelectionError,
        apply_noninteractive_model_selection,
        resolve_noninteractive_selection,
    )

    for name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    hermes_home.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    cfg["model"] = {"provider": "anthropic", "default": "prior-model"}
    save_config(cfg)
    config_path = Path(hermes_home) / "config.yaml"
    before = config_path.read_bytes()

    selection = resolve_noninteractive_selection("openrouter", "openai/gpt-5.4")
    with pytest.raises(NonInteractiveSelectionError) as excinfo:
        apply_noninteractive_model_selection(selection)

    assert excinfo.value.exit_code == 3
    message = str(excinfo.value)
    assert "openrouter" in message.lower()
    assert "OPENROUTER_API_KEY" in message
    assert config_path.read_bytes() == before


def test_validate_noninteractive_rejects_placeholder_api_key(monkeypatch):
    """The selector must reject the same placeholder values as runtime auth."""
    import pytest

    from hermes_cli.model_noninteractive import (
        NonInteractiveSelectionError,
        resolve_noninteractive_selection,
        validate_noninteractive_prerequisites,
    )

    for name in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "changeme")

    selection = resolve_noninteractive_selection(
        "anthropic", "claude-sonnet-4-20250514"
    )
    with pytest.raises(NonInteractiveSelectionError) as excinfo:
        validate_noninteractive_prerequisites(selection)

    assert excinfo.value.exit_code == 3
    message = str(excinfo.value)
    assert "changeme" not in message
    assert "ANTHROPIC_API_KEY" in message


def test_validate_noninteractive_accepts_openrouter_pool_credential(
    tmp_path, monkeypatch
):
    """A manual OpenRouter pool entry is a first-class runtime credential."""
    import uuid

    from agent.credential_pool import (
        AUTH_TYPE_API_KEY,
        SOURCE_MANUAL,
        PooledCredential,
        load_pool,
    )
    from hermes_cli.model_noninteractive import (
        resolve_noninteractive_selection,
        validate_noninteractive_prerequisites,
    )

    for name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    (tmp_path / "hermes").mkdir(parents=True, exist_ok=True)
    load_pool("openrouter").add_entry(
        PooledCredential(
            provider="openrouter",
            id=uuid.uuid4().hex[:6],
            label="api-key-1",
            auth_type=AUTH_TYPE_API_KEY,
            priority=0,
            source=SOURCE_MANUAL,
            access_token="«redacted:sk-…»",
            base_url="https://openrouter.ai/api/v1",
        )
    )

    selection = resolve_noninteractive_selection("openrouter", "openai/gpt-5.4")
    validate_noninteractive_prerequisites(selection)


def test_validate_noninteractive_accepts_zai_credential_pool(tmp_path, monkeypatch):
    """Provider pools, not only env vars, must satisfy API-key prerequisites."""
    import uuid

    from agent.credential_pool import (
        AUTH_TYPE_API_KEY,
        SOURCE_MANUAL,
        PooledCredential,
        load_pool,
    )
    from hermes_cli.model_noninteractive import (
        resolve_noninteractive_selection,
        validate_noninteractive_prerequisites,
    )

    for name in ("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GLM_BASE_URL", "https://api.z.ai/api/paas/v4")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    (tmp_path / "hermes").mkdir(parents=True, exist_ok=True)
    load_pool("zai").add_entry(
        PooledCredential(
            provider="zai",
            id=uuid.uuid4().hex[:6],
            label="api-key-1",
            auth_type=AUTH_TYPE_API_KEY,
            priority=0,
            source=SOURCE_MANUAL,
            access_token="zai-pool-key-sentinel",
            base_url="https://api.z.ai/api/paas/v4",
        )
    )

    selection = resolve_noninteractive_selection("zai", "glm-5")
    validate_noninteractive_prerequisites(selection)


def test_validate_noninteractive_accepts_lmstudio_no_auth(monkeypatch):
    """LM Studio's canonical no-auth sentinel must remain selectable."""
    from hermes_cli.model_noninteractive import validate_noninteractive_prerequisites

    monkeypatch.delenv("LM_API_KEY", raising=False)
    validate_noninteractive_prerequisites(
        {"provider": "lmstudio", "auth_type": "api_key"}
    )


def test_validate_noninteractive_markerless_external_provider_fails_closed(
    monkeypatch,
):
    """Installed is not authenticated when the CLI exposes no login marker."""
    import pytest

    import hermes_cli.auth as auth_mod
    from hermes_cli.model_noninteractive import (
        NonInteractiveSelectionError,
        validate_noninteractive_prerequisites,
    )

    monkeypatch.setattr(
        auth_mod,
        "get_external_process_provider_status",
        lambda _provider: {
            "configured": True,
            "installed": True,
            "logged_in": True,
            "login_markers": [],
        },
    )

    with pytest.raises(NonInteractiveSelectionError) as excinfo:
        validate_noninteractive_prerequisites(
            {"provider": "copilot-acp", "auth_type": "external_process"}
        )

    assert excinfo.value.exit_code == 3
    message = str(excinfo.value).lower()
    assert "cannot be verified" in message
    assert "token" not in message
