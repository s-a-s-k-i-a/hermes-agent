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


kimi_code = KimiCodeProfile(
    name="kimi-code",
    display_name="Kimi Code (OAuth / subscription)",
    description="Kimi Code CLI subscription via its existing OAuth session",
    api_mode="chat_completions",
    env_vars=(),
    base_url="acp://kimi",
    auth_type="external_process",
    supports_health_check=False,
    fallback_models=("k3",),
    default_aux_model="k3",
    external_command_env_vars=("KIMI_CODE_CLI_PATH",),
    external_preferred_commands=("~/.kimi-code/bin/kimi",),
    external_default_command="kimi",
    external_default_args=("acp",),
    external_login_args=("login",),
    external_login_markers=("~/.kimi-code/credentials/kimi-code.json",),
)

register_provider(kimi_code)
