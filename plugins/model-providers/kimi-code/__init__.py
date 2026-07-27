"""Kimi Code CLI ACP provider profile.

Authentication stays wholly inside the Kimi Code CLI. Hermes launches the
CLI's ACP server and deliberately does not read or persist its OAuth session.
"""

from providers import register_provider
from providers.base import ProviderProfile


class KimiCodeProfile(ProviderProfile):
    """Kimi Code OAuth/subscription through the local ACP subprocess."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """The ACP subprocess owns model discovery and authentication."""
        return None

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        **context,
    ) -> tuple[dict, dict]:
        """Pass Hermes reasoning intent to the ACP transport unchanged."""
        top_level: dict = {}
        if reasoning_config:
            top_level["reasoning_config"] = reasoning_config
        session_id = str(context.get("session_id") or "").strip()
        if session_id:
            # Private transport metadata: ExternalACPClient consumes this to
            # keep exactly one ACP session bound to one Hermes session. It is
            # never serialized onto the ACP wire.
            top_level["_hermes_session_id"] = session_id
        return {}, top_level


kimi_code = KimiCodeProfile(
    name="kimi-code",
    display_name="Kimi Code ACP (OAuth / subscription; agent backend)",
    description="Kimi Code CLI ACP account integration (not a Hermes model route)",
    api_mode="chat_completions",
    env_vars=(),
    base_url="acp://kimi",
    auth_type="external_process",
    supports_health_check=False,
    supports_vision=True,
    fallback_models=("k3", "k3-256k"),
    model_context_lengths={"k3": 1_048_576, "k3-256k": 262_144},
    default_aux_model="k3",
    external_command_env_vars=("KIMI_CODE_CLI_PATH",),
    external_preferred_commands=("~/.kimi-code/bin/kimi",),
    external_default_command="kimi",
    external_default_args=("acp",),
    external_process_env_vars=(
        "KIMI_CODE_HOME",
        "KIMI_DISABLE_TELEMETRY",
        "KIMI_CODE_NO_AUTO_UPDATE",
        "KIMI_CLI_NO_AUTO_UPDATE",
        "KIMI_DISABLE_CRON",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "NODE_EXTRA_CA_CERTS",
    ),
    external_data_root_env_var="KIMI_CODE_HOME",
    external_default_data_root="~/.kimi-code",
    external_login_args=("login",),
    external_login_markers=("credentials/kimi-code.json",),
    external_logout_removes_login_markers=True,
    external_doctor_probe_initialize=True,
    external_expected_agent_name="Kimi Code CLI",
    external_required_auth_methods=("login",),
    external_required_prompt_capabilities=("image", "embeddedContext"),
    external_required_session_capabilities=("resume",),
    external_preserves_system_instructions=False,
)

register_provider(kimi_code)
