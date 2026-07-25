"""Non-interactive provider/model selection shared by CLI subcommands.

Small, promptless counterpart to the interactive picker flows in
``hermes_cli/model_setup_flows.py``.  Both ``hermes model --provider --model``
and (later) ``hermes fallback add --provider --model`` validate through here
so a mis-typed provider/model fails fast with an actionable error instead of
silently writing broken config.

Deliberately no prompting, no OAuth, no API-key ingestion: non-interactive
means non-interactive.  Exit-code contract (carried via
``NonInteractiveSelectionError.exit_code``):

* 2 — validation/usage error (unknown provider, unknown model)
* 3 — missing credential prerequisite (CLI not installed / not logged in /
      missing API key env var)
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class NonInteractiveSelectionError(Exception):
    """Raised when a non-interactive provider/model selection is invalid."""

    def __init__(self, message: str, *, exit_code: int = 2):
        super().__init__(message)
        self.exit_code = exit_code


def _valid_models_for(provider_id: str) -> list:
    """Model universe for a provider: picker catalog first, profile fallback."""
    from hermes_cli.models import _PROVIDER_MODELS

    models = list(_PROVIDER_MODELS.get(provider_id) or [])
    if models:
        return models
    try:
        from providers import get_provider_profile

        profile = get_provider_profile(provider_id)
        if profile is not None:
            return list(profile.fallback_models or ())
    except Exception:
        pass
    return []


def _resolve_canonical_provider(provider: Optional[str]) -> str:
    """Canonicalize a provider through the auth-aware resolver.

    Uses ``hermes_cli.auth.resolve_provider`` — the same alias matrix the
    rest of the CLI honours (including plugin-declared aliases like
    ``codex``/``openai_codex``) — instead of a second, models.dev-shaped
    normalization that rejects valid canonical ids.

    Non-interactive hard rules:

    * empty/whitespace/None provider → exit 2 (never falls through to
      ``auto`` credential auto-detection)
    * explicit ``auto`` → exit 2 (ambient-credential resolution is not
      deterministic; a concrete provider is required)
    * ``custom`` (and its local aliases ollama/vllm/llama.cpp) → exit 2
      until a credential-preserving non-interactive contract exists —
      the interactive ``hermes model`` custom flow owns base_url/api_key
      configuration.
    """
    provider_raw = str(provider or "").strip()
    if not provider_raw:
        raise NonInteractiveSelectionError(
            "--provider requires a non-empty provider id.",
            exit_code=2,
        )
    if provider_raw.lower() == "auto":
        raise NonInteractiveSelectionError(
            "Provider 'auto' is not valid for non-interactive selection — "
            "pass a concrete provider id.",
            exit_code=2,
        )

    from hermes_cli.auth import AuthError, resolve_provider

    try:
        provider_id = resolve_provider(provider_raw)
    except AuthError as exc:
        raise NonInteractiveSelectionError(str(exc), exit_code=2) from exc

    if provider_id == "custom":
        raise NonInteractiveSelectionError(
            f"Provider '{provider_raw}' uses the custom/local endpoint flow "
            f"(base_url + optional api_key), which cannot be configured "
            f"non-interactively yet. Run interactive 'hermes model' and pick "
            f"the custom provider instead.",
            exit_code=2,
        )
    return provider_id


def resolve_noninteractive_selection(
    provider: Optional[str], model: Optional[str]
) -> Dict[str, Any]:
    """Validate and normalize a provider/model pair without any prompting.

    Returns a selection dict with ``provider``, ``model``, ``base_url``,
    ``api_mode``, ``auth_type`` and ``needs_api_key``.  Raises
    :class:`NonInteractiveSelectionError` (exit_code 2) for unknown
    providers/models.
    """
    from hermes_cli.auth import PROVIDER_REGISTRY

    provider_id = _resolve_canonical_provider(provider)

    try:
        from providers import get_provider_profile

        profile = get_provider_profile(provider_id)
    except Exception:
        profile = None

    pconfig = PROVIDER_REGISTRY.get(provider_id)
    auth_type = (
        getattr(profile, "auth_type", None)
        or getattr(pconfig, "auth_type", None)
        or "api_key"
    )

    valid_models = _valid_models_for(provider_id)
    selected_model = (model or "").strip() if model else ""
    if not selected_model:
        # external_process contract: the profile's default_aux_model is the
        # provider-declared default and takes priority over the resolver.
        if auth_type == "external_process" and profile is not None:
            selected_model = str(
                getattr(profile, "default_aux_model", "") or ""
            ).strip()
        if not selected_model:
            # Canonical cost-safe resolver — never blindly take the catalog's
            # first (often flagship/most-expensive) entry.
            from hermes_cli.models import get_default_model_for_provider

            selected_model = (
                get_default_model_for_provider(provider_id) or ""
            ).strip()
        if not selected_model:
            raise NonInteractiveSelectionError(
                f"Provider '{provider_id}' has no safe default model — pass "
                f"--model explicitly.",
                exit_code=2,
            )
    elif valid_models and selected_model not in valid_models:
        raise NonInteractiveSelectionError(
            f"Unknown model '{selected_model}' for provider '{provider_id}'. "
            f"Valid models: {', '.join(valid_models)}",
            exit_code=2,
        )

    base_url = (
        getattr(profile, "base_url", None)
        or getattr(pconfig, "inference_base_url", "")
        or ""
    )
    api_mode = getattr(profile, "api_mode", None) or "chat_completions"

    return {
        "provider": provider_id,
        "model": selected_model,
        "base_url": base_url,
        "api_mode": api_mode,
        "auth_type": auth_type,
        "needs_api_key": auth_type == "api_key",
    }


def _check_external_process_prerequisites(provider_id: str) -> None:
    """Require installed + logged-in for external-process providers.

    Non-interactive means non-interactive: no OAuth flow, no API-key prompt.
    Missing prerequisites raise exit_code 3 with an executable remedy.
    """
    from hermes_cli.auth import get_external_process_provider_status

    status = get_external_process_provider_status(provider_id)
    if not status.get("installed"):
        # Never echo resolved command paths or env-var override VALUES here —
        # they are user-controlled and may embed token-like path segments.
        # Only the provider id and the allowed env VAR NAMES may appear.
        env_hint = ""
        try:
            from providers import get_provider_profile

            profile = get_provider_profile(provider_id)
            env_vars = tuple(
                getattr(profile, "external_command_env_vars", ()) or ()
            )
            if env_vars:
                env_hint = (
                    f", or point {' / '.join(env_vars)} at the executable"
                )
        except Exception:
            pass
        raise NonInteractiveSelectionError(
            f"Provider '{provider_id}' requires its CLI, but no executable "
            f"was found. Install the provider's CLI{env_hint} and retry.",
            exit_code=3,
        )
    login_markers = tuple(status.get("login_markers") or ())
    if not login_markers:
        raise NonInteractiveSelectionError(
            f"Provider '{provider_id}' authentication cannot be verified "
            "non-interactively because its CLI declares no login marker. "
            "Use the interactive 'hermes model' flow instead.",
            exit_code=3,
        )
    if not status.get("logged_in"):
        raise NonInteractiveSelectionError(
            f"Provider '{provider_id}' CLI is installed but not logged in. "
            f"Run 'hermes auth add {provider_id}' (which runs the CLI's own "
            f"login) and retry.",
            exit_code=3,
        )


def _check_api_key_prerequisites(provider_id: str) -> None:
    """Require the same usable credential the runtime resolver accepts.

    This deliberately delegates env/``.env``/credential-pool and no-auth-local
    handling to the canonical auth resolver instead of maintaining a weaker
    selector-only copy.  Errors are translated to a bounded, secret-free exit-3
    message.
    """
    from hermes_cli.auth import (
        PROVIDER_REGISTRY,
        has_usable_secret,
        resolve_api_key_provider_credentials,
    )

    pconfig = PROVIDER_REGISTRY.get(provider_id)
    env_vars = tuple(getattr(pconfig, "api_key_env_vars", ()) or ())
    env_hint = env_vars[0] if env_vars else "the provider's API-key variable"
    try:
        credentials = resolve_api_key_provider_credentials(provider_id)
        secret = credentials.get("api_key", "")
    except Exception:
        secret = ""
    if has_usable_secret(secret):
        return

    raise NonInteractiveSelectionError(
        f"Provider '{provider_id}' needs a usable API key but none is "
        f"configured. Set {env_hint} (e.g. via 'hermes auth add "
        f"{provider_id}') and retry.",
        exit_code=3,
    )


def _check_openrouter_prerequisites() -> None:
    """Require a usable OpenRouter env/pool credential without prompting.

    OpenRouter intentionally lives outside ``PROVIDER_REGISTRY``.  Mirror the
    runtime's two supported credential classes explicitly: ``.env``/environment
    first, then the OpenRouter credential pool.
    """
    from hermes_cli.auth import has_usable_secret
    from hermes_cli.config import get_env_value_prefer_dotenv

    for env_var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        if has_usable_secret(get_env_value_prefer_dotenv(env_var) or ""):
            return

    try:
        from agent.credential_pool import load_pool

        pool = load_pool("openrouter")
        entry = pool.peek() if pool and pool.has_credentials() else None
        secret = (
            getattr(entry, "runtime_api_key", "")
            or getattr(entry, "access_token", "")
            if entry is not None
            else ""
        )
        if has_usable_secret(secret):
            return
    except Exception:
        pass

    raise NonInteractiveSelectionError(
        "Provider 'openrouter' needs a usable credential. Set "
        "OPENROUTER_API_KEY (or add an OpenRouter credential with "
        "'hermes auth add openrouter') and retry.",
        exit_code=3,
    )


def validate_noninteractive_prerequisites(selection: Dict[str, Any]) -> None:
    """Validate selection readiness without prompting or config mutation."""
    provider_id = selection["provider"]
    auth_type = selection.get("auth_type") or "api_key"
    if auth_type == "external_process":
        _check_external_process_prerequisites(provider_id)
    elif auth_type == "api_key":
        if provider_id == "openrouter":
            _check_openrouter_prerequisites()
        else:
            _check_api_key_prerequisites(provider_id)


def apply_noninteractive_model_selection(selection: Dict[str, Any]) -> None:
    """Persist a validated selection to config.yaml without prompting.

    Mirrors the tail of the interactive picker flows: writes
    ``model.{default,provider,base_url,api_mode}`` and strips inline endpoint
    credentials.  Never writes ``model.api_key`` and never touches auth.json
    in any way — no credential writes, no credential-pool writes, and no
    ``active_provider`` mutation.  ``config.model.provider`` is authoritative
    for which provider runs; external OAuth credentials belong to their
    owning CLIs, not to Hermes, so a model switch must leave auth.json
    byte-identical.

    Atomicity contract: the whole model-section update (default, provider,
    base_url, api_mode, credential cleanup) is assembled in memory on one
    loaded config and persisted with exactly ONE ``save_config`` call.  A
    failed save must leave the on-disk config byte-identical — no separate
    ``_save_model_choice``-style partial write may precede it.
    """
    from hermes_cli import config as config_mod

    validate_noninteractive_prerequisites(selection)
    provider_id = selection["provider"]

    cfg = config_mod.load_config()
    model = cfg.get("model")
    if not isinstance(model, dict):
        model = {"default": model} if model else {}
        cfg["model"] = model
    model["default"] = selection["model"]
    model["provider"] = provider_id
    if selection.get("base_url"):
        model["base_url"] = selection["base_url"]
    model["api_mode"] = selection.get("api_mode") or "chat_completions"
    config_mod.clear_model_endpoint_credentials(model, clear_api_mode=False)
    config_mod.save_config(cfg)
