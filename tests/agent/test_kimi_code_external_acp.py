"""Runtime and safety contracts for generic external-process ACP providers."""

from __future__ import annotations

import asyncio
import io
import json
import os
import queue
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _install_fake_kimi(home: Path) -> Path:
    binary = home / ".kimi-code" / "bin" / "kimi"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    return binary


def _prepare_kimi(monkeypatch, tmp_path) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    binary = _install_fake_kimi(home)
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("KIMI_CODE_CLI_PATH", str(binary))
    for name in (
        "HERMES_COPILOT_ACP_COMMAND",
        "COPILOT_CLI_PATH",
        "HERMES_COPILOT_ACP_ARGS",
        "COPILOT_ACP_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    return binary


def test_aiagent_constructs_external_acp_client_for_kimi(monkeypatch, tmp_path):
    binary = _prepare_kimi(monkeypatch, tmp_path)

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI") as openai_cls,
        patch("agent.copilot_acp_client.ExternalACPClient") as external_cls,
    ):
        external_client = MagicMock()
        external_cls.return_value = external_client

        from run_agent import AIAgent

        agent = AIAgent(
            api_key="external-process",
            base_url="acp://kimi",
            provider="kimi-code",
            model="k3",
            acp_command=str(binary),
            acp_args=["acp"],
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    assert agent.client is external_client
    openai_cls.assert_not_called()
    external_cls.assert_called_once()
    kwargs = external_cls.call_args.kwargs
    assert kwargs["provider"] == "kimi-code"
    assert kwargs["base_url"] == "acp://kimi"
    assert kwargs["command"] == str(binary)
    assert kwargs["args"] == ["acp"]


def test_generic_acp_runtime_is_non_streaming_and_never_upgrades_to_responses(
    monkeypatch, tmp_path
):
    _prepare_kimi(monkeypatch, tmp_path)

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("agent.copilot_acp_client.ExternalACPClient", return_value=MagicMock()),
    ):
        from agent.copilot_acp_client import is_external_acp_runtime
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="external-process",
            base_url="acp://kimi",
            provider="kimi-code",
            model="gpt-5.4",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    assert is_external_acp_runtime(agent.provider, agent.base_url) is True
    assert agent.api_mode == "chat_completions"
    assert agent._uses_non_streaming_runtime() is True


def test_auxiliary_client_resolves_kimi_external_process_without_openai(
    monkeypatch, tmp_path
):
    _prepare_kimi(monkeypatch, tmp_path)

    with (
        patch("agent.auxiliary_client.OpenAI") as openai_cls,
        patch("agent.copilot_acp_client.ExternalACPClient") as external_cls,
    ):
        external_client = MagicMock()
        external_client.api_key = "external-process"
        external_client.base_url = "acp://kimi"
        external_cls.return_value = external_client

        from agent.auxiliary_client import resolve_provider_client

        client, model = resolve_provider_client("kimi-code", model="k3")

    assert client is external_client
    assert model == "k3"
    openai_cls.assert_not_called()


def test_main_fallback_chain_activates_kimi_without_api_key(monkeypatch, tmp_path):
    _prepare_kimi(monkeypatch, tmp_path)

    primary_client = MagicMock()
    primary_client.api_key = "primary-key"
    primary_client.base_url = "https://openrouter.ai/api/v1"

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI", return_value=primary_client),
    ):
        from agent.copilot_acp_client import ExternalACPClient
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="primary-key",
            base_url="https://openrouter.ai/api/v1",
            provider="openrouter",
            model="primary/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent._fallback_chain = [{"provider": "kimi-code", "model": "k3"}]
        agent._fallback_index = 0
        agent._fallback_activated = False

        assert agent._try_activate_fallback() is True

    assert agent.provider == "kimi-code"
    assert agent.model == "k3"
    assert agent.api_mode == "chat_completions"
    assert agent.api_key == "external-process"
    assert isinstance(agent.client, ExternalACPClient)


class _FakeProcess:
    def __init__(self) -> None:
        self.stdin = io.StringIO()


class _ProtocolStdout:
    def __init__(self):
        self._lines = queue.Queue()

    def put(self, payload):
        self._lines.put(json.dumps(payload) + "\n")

    def close(self):
        self._lines.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        line = self._lines.get(timeout=2)
        if line is None:
            raise StopIteration
        return line


class _ProtocolStdin:
    def __init__(self, process, *, model_error=False):
        self._process = process
        self._model_error = model_error

    def write(self, line):
        request = json.loads(line)
        self._process.requests.append(request)
        request_id = request["id"]
        method = request["method"]
        if method == "initialize":
            result = {"authMethods": [{"id": "login", "name": "Log in"}]}
        elif method == "authenticate":
            result = {}
        elif method == "session/new":
            result = {
                "sessionId": "kimi-session",
                "configOptions": [
                    {
                        "id": "model",
                        "category": "model",
                        "type": "select",
                        "currentValue": "kimi-code/k3",
                        "options": [
                            {"value": "kimi-code/k3", "name": "K3"},
                            {"value": "kimi-code/k3-256k", "name": "K3 256K"},
                        ],
                    }
                ],
            }
        elif method == "session/set_config_option" and self._model_error:
            self._process.stdout.put(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32602, "message": "model rejected"},
                }
            )
            return len(line)
        elif method == "session/prompt":
            self._process.stdout.put(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": "kimi-session",
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": "ok"},
                        },
                    },
                }
            )
            result = {"stopReason": "end_turn"}
        else:
            result = {}
        self._process.stdout.put(
            {"jsonrpc": "2.0", "id": request_id, "result": result}
        )
        return len(line)

    def flush(self):
        return None


class _ProtocolProcess:
    def __init__(self, *, model_error=False):
        self.requests = []
        self.stdout = _ProtocolStdout()
        self.stderr = io.StringIO()
        self.stdin = _ProtocolStdin(self, model_error=model_error)
        self._returncode = None

    def poll(self):
        return self._returncode

    def terminate(self):
        self._returncode = 0
        self.stdout.close()

    def wait(self, timeout=None):
        return self._returncode

    def kill(self):
        self.terminate()


@pytest.mark.parametrize(
    ("requested_model", "acp_model"),
    [
        ("k3", "kimi-code/k3"),
        ("kimi-code/k3-256k", "kimi-code/k3-256k"),
    ],
)
def test_kimi_acp_authenticates_and_sets_selected_model_before_prompt(
    tmp_path, requested_model, acp_model
):
    from agent.copilot_acp_client import ExternalACPClient

    process = _ProtocolProcess()
    client = ExternalACPClient(
        provider="kimi-code",
        command="kimi",
        args=["acp"],
        acp_cwd=str(tmp_path),
    )

    with patch("agent.copilot_acp_client.subprocess.Popen", return_value=process):
        response = client._create_chat_completion(
            model=requested_model,
            messages=[{"role": "user", "content": "hello"}],
        )

    assert response.choices[0].message.content == "ok"
    assert [request["method"] for request in process.requests] == [
        "initialize",
        "authenticate",
        "session/new",
        "session/set_config_option",
        "session/prompt",
    ]
    assert process.requests[1]["params"] == {"methodId": "login"}
    assert process.requests[3]["params"] == {
        "sessionId": "kimi-session",
        "configId": "model",
        "value": acp_model,
    }


def test_kimi_acp_model_selection_error_fails_closed_before_prompt(tmp_path):
    from agent.copilot_acp_client import ExternalACPClient

    process = _ProtocolProcess(model_error=True)
    client = ExternalACPClient(
        provider="kimi-code",
        command="kimi",
        args=["acp"],
        acp_cwd=str(tmp_path),
    )

    with patch("agent.copilot_acp_client.subprocess.Popen", return_value=process):
        with pytest.raises(
            RuntimeError,
            match="session/set_config_option failed: model rejected",
        ):
            client._create_chat_completion(
                model="k3",
                messages=[{"role": "user", "content": "hello"}],
            )

    assert [request["method"] for request in process.requests] == [
        "initialize",
        "authenticate",
        "session/new",
        "session/set_config_option",
    ]


def test_copilot_acp_does_not_receive_kimi_model_configuration(tmp_path):
    from agent.copilot_acp_client import CopilotACPClient

    process = _ProtocolProcess()
    client = CopilotACPClient(
        command="copilot",
        args=["--acp", "--stdio"],
        acp_cwd=str(tmp_path),
    )

    with patch("agent.copilot_acp_client.subprocess.Popen", return_value=process):
        client._create_chat_completion(
            model="gpt-5.4",
            messages=[{"role": "user", "content": "hello"}],
        )

    methods = [request["method"] for request in process.requests]
    assert "authenticate" not in methods
    assert "session/set_config_option" not in methods
    assert "session/set_model" not in methods
    assert methods == ["initialize", "session/new", "session/prompt"]


def test_kimi_acp_denies_permission_requests_and_direct_file_writes(tmp_path):
    from agent.copilot_acp_client import ExternalACPClient

    client = ExternalACPClient(
        provider="kimi-code",
        base_url="acp://kimi",
        command="kimi",
        args=["acp"],
        acp_cwd=str(tmp_path),
    )

    permission_process = _FakeProcess()
    assert client._handle_server_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "session/request_permission",
            "params": {},
        },
        process=permission_process,
        cwd=str(tmp_path),
        text_parts=[],
        reasoning_parts=[],
    )
    permission = json.loads(permission_process.stdin.getvalue())
    assert permission["result"]["outcome"]["outcome"] == "cancelled"

    target = tmp_path / "must-not-exist.txt"
    write_process = _FakeProcess()
    assert client._handle_server_message(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "fs/write_text_file",
            "params": {"path": str(target), "content": "side effect"},
        },
        process=write_process,
        cwd=str(tmp_path),
        text_parts=[],
        reasoning_parts=[],
    )
    write_response = json.loads(write_process.stdin.getvalue())
    assert "error" in write_response
    assert target.exists() is False


def test_kimi_subprocess_preserves_home_but_strips_oauth_and_tier_one_secrets(
    monkeypatch, tmp_path
):
    binary = _prepare_kimi(monkeypatch, tmp_path)
    home = Path(os.environ["HOME"])
    monkeypatch.setenv("KIMI_ACCESS_TOKEN", "access-secret")
    monkeypatch.setenv("KIMI_REFRESH_TOKEN", "refresh-secret")
    monkeypatch.setenv("GH_TOKEN", "tier-one-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-provider-secret")

    captured = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        raise FileNotFoundError(str(binary))

    from agent.copilot_acp_client import ExternalACPClient

    client = ExternalACPClient(
        provider="kimi-code",
        base_url="acp://kimi",
        command=str(binary),
        args=["acp"],
        acp_cwd=str(tmp_path),
    )
    with patch(
        "agent.copilot_acp_client.subprocess.Popen", side_effect=fake_popen
    ):
        with pytest.raises(RuntimeError, match="Could not start Kimi Code"):
            client._run_prompt("hello", timeout_seconds=1)

    assert captured["command"] == [str(binary), "acp"]
    assert captured["env"]["HOME"] == str(home)
    assert captured["env"]["HERMES_REAL_HOME"] == str(home)
    assert "KIMI_ACCESS_TOKEN" not in captured["env"]
    assert "KIMI_REFRESH_TOKEN" not in captured["env"]
    assert "GH_TOKEN" not in captured["env"]
    assert "OPENAI_API_KEY" not in captured["env"]


def test_external_acp_async_facade_offloads_blocking_completion():
    from agent.auxiliary_client import _to_async_client
    from agent.copilot_acp_client import AsyncExternalACPClient, ExternalACPClient

    sync_client = ExternalACPClient(
        provider="kimi-code",
        api_key="external-process",
        base_url="acp://kimi",
        command="/tmp/kimi",
        args=["acp"],
    )
    caller_thread = threading.get_ident()
    worker_threads: list[int] = []
    response = object()

    def fake_create(**kwargs):
        worker_threads.append(threading.get_ident())
        assert kwargs["model"] == "k3"
        return response

    sync_client.chat.completions.create = fake_create
    async_client, model = _to_async_client(sync_client, "k3")

    assert isinstance(async_client, AsyncExternalACPClient)
    assert model == "k3"
    actual = asyncio.run(
        async_client.chat.completions.create(model="k3", messages=[])
    )
    assert actual is response
    assert worker_threads and worker_threads[0] != caller_thread
