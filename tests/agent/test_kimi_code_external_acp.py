"""Runtime and safety contracts for generic external-process ACP providers."""

from __future__ import annotations

import asyncio
import io
import json
import os
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
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
    monkeypatch.delenv("HERMES_REAL_HOME", raising=False)
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

        agent: Any = AIAgent(
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

        agent: Any = AIAgent(
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

        agent: Any = AIAgent(
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


def test_kimi_primary_restores_command_and_args_after_api_fallback(
    monkeypatch, tmp_path
):
    binary = _prepare_kimi(monkeypatch, tmp_path)

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
    ):
        from agent.copilot_acp_client import ExternalACPClient
        from run_agent import AIAgent

        agent: Any = AIAgent(
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

    fallback_client = SimpleNamespace(
        api_key="test-fallback-key",
        base_url="https://openrouter.ai/api/v1",
        _custom_headers={},
    )
    agent._fallback_chain = [
        {"provider": "openrouter", "model": "openai/gpt-4o-mini"}
    ]
    agent._fallback_index = 0
    agent._fallback_activated = False

    with patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(fallback_client, "openai/gpt-4o-mini"),
    ):
        assert agent._try_activate_fallback() is True

    assert agent.provider == "openrouter"
    assert agent.client is fallback_client
    assert agent._restore_primary_runtime() is True
    assert agent.provider == "kimi-code"
    assert isinstance(agent.client, ExternalACPClient)
    assert agent.client._acp_command == str(binary)
    assert agent.client._acp_args == ["acp"]


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
    def __init__(self, process, *, model_error=False, advertise_login=True):
        self._process = process
        self._model_error = model_error
        self._advertise_login = advertise_login

    def write(self, line):
        request = json.loads(line)
        self._process.requests.append(request)
        request_id = request["id"]
        method = request["method"]
        if method == "initialize":
            result = {
                "authMethods": (
                    [{"id": "login", "name": "Log in"}]
                    if self._advertise_login
                    else []
                )
            }
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
    def __init__(self, *, model_error=False, advertise_login=True):
        self.requests = []
        self.stdout = _ProtocolStdout()
        self.stderr = io.StringIO()
        self.stdin = _ProtocolStdin(
            self,
            model_error=model_error,
            advertise_login=advertise_login,
        )
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
    assert process.requests[0]["params"]["clientCapabilities"]["fs"] == {
        "readTextFile": False,
        "writeTextFile": False,
    }
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


def test_kimi_acp_without_login_method_reports_required_capability(tmp_path):
    from agent.copilot_acp_client import ExternalACPClient

    process = _ProtocolProcess(advertise_login=False)
    client = ExternalACPClient(
        provider="kimi-code",
        command="kimi",
        args=["acp"],
        acp_cwd=str(tmp_path),
    )

    with patch("agent.copilot_acp_client.subprocess.Popen", return_value=process):
        with pytest.raises(RuntimeError) as excinfo:
            client._create_chat_completion(
                model="k3",
                messages=[{"role": "user", "content": "hello"}],
            )

    message = str(excinfo.value)
    assert "did not advertise its CLI-owned login method" in message
    assert "0.29.1" not in message
    assert [request["method"] for request in process.requests] == ["initialize"]


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


def test_kimi_acp_selects_offered_reject_once_for_realistic_permission_request(
    tmp_path,
):
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
            "params": {
                "sessionId": "redacted-session",
                "toolCall": {
                    "toolCallId": "redacted-tool-call",
                    "title": "Bash",
                },
                "options": [
                    {
                        "optionId": "approve_once",
                        "name": "Approve once",
                        "kind": "allow_once",
                    },
                    {
                        "optionId": "approve_always",
                        "name": "Approve for this session",
                        "kind": "allow_always",
                    },
                    {
                        "optionId": "reject",
                        "name": "Reject",
                        "kind": "reject_once",
                    },
                ],
            },
        },
        process=cast(Any, permission_process),
        cwd=str(tmp_path),
        text_parts=[],
        reasoning_parts=[],
    )
    permission = json.loads(permission_process.stdin.getvalue())
    assert permission["result"]["outcome"] == {
        "outcome": "selected",
        "optionId": "reject",
    }


def test_kimi_acp_rejects_direct_reads_of_representative_secret_file(tmp_path):
    from agent.copilot_acp_client import ExternalACPClient

    client = ExternalACPClient(
        provider="kimi-code",
        base_url="acp://kimi",
        command="kimi",
        args=["acp"],
        acp_cwd=str(tmp_path),
    )
    secret = "https://user:representative-password@example.invalid"
    target = tmp_path / ".git-credentials"
    target.write_text(secret, encoding="utf-8")
    read_process = _FakeProcess()

    assert client._handle_server_message(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "fs/read_text_file",
            "params": {"path": str(target)},
        },
        process=cast(Any, read_process),
        cwd=str(tmp_path),
        text_parts=[],
        reasoning_parts=[],
    )

    read_response = read_process.stdin.getvalue()
    assert "error" in json.loads(read_response)
    assert secret not in read_response


def test_kimi_acp_denies_direct_file_writes(tmp_path):
    from agent.copilot_acp_client import ExternalACPClient

    client = ExternalACPClient(
        provider="kimi-code",
        base_url="acp://kimi",
        command="kimi",
        args=["acp"],
        acp_cwd=str(tmp_path),
    )

    target = tmp_path / "must-not-exist.txt"
    write_process = _FakeProcess()
    assert client._handle_server_message(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "fs/write_text_file",
            "params": {"path": str(target), "content": "side effect"},
        },
        process=cast(Any, write_process),
        cwd=str(tmp_path),
        text_parts=[],
        reasoning_parts=[],
    )
    write_response = json.loads(write_process.stdin.getvalue())
    assert "error" in write_response
    assert target.exists() is False


def test_kimi_inference_subprocess_receives_only_minimal_declared_environment(
    monkeypatch, tmp_path
):
    binary = _prepare_kimi(monkeypatch, tmp_path)
    home = Path(os.environ["HOME"])
    monkeypatch.setenv("KIMI_ACCESS_TOKEN", "access-secret")
    monkeypatch.setenv("KIMI_REFRESH_TOKEN", "refresh-secret")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "unrelated-oauth-secret")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "unrelated-cloud-secret")
    monkeypatch.setenv("AWS_PROFILE", "unrelated-cloud-profile")
    monkeypatch.setenv("SSH_AUTH_SOCK", str(tmp_path / "agent.sock"))
    monkeypatch.setenv("GH_TOKEN", "tier-one-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-provider-secret")
    provider_home = tmp_path / "relocated-kimi"
    monkeypatch.setenv("KIMI_CODE_HOME", str(provider_home))

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
    assert set(captured["env"]) == expected_keys
    assert captured["env"]["HOME"] == str(home)
    assert captured["env"]["HERMES_REAL_HOME"] == str(home)
    assert captured["env"]["KIMI_CODE_HOME"] == str(provider_home)


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

    setattr(sync_client.chat.completions, "create", fake_create)
    async_client, model = _to_async_client(sync_client, "k3")

    assert isinstance(async_client, AsyncExternalACPClient)
    assert model == "k3"
    actual = asyncio.run(
        async_client.chat.completions.create(model="k3", messages=[])
    )
    assert actual is response
    assert worker_threads and worker_threads[0] != caller_thread


def test_external_acp_serializes_overlapping_completions():
    """One client must never let a completion close another request's child."""
    from agent.copilot_acp_client import ExternalACPClient

    client = ExternalACPClient(
        provider="kimi-code",
        command="/tmp/kimi",
        args=["acp"],
    )
    active = 0
    max_active = 0
    state_lock = threading.Lock()
    rendezvous = threading.Barrier(2, timeout=0.3)

    def fake_run_prompt(_prompt, **_kwargs):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        try:
            try:
                rendezvous.wait()
            except threading.BrokenBarrierError:
                pass
            time.sleep(0.02)
            return "ok", ""
        finally:
            with state_lock:
                active -= 1

    setattr(client, "_run_prompt", fake_run_prompt)
    start = threading.Barrier(3)
    results = []
    errors = []

    def complete():
        start.wait()
        try:
            results.append(
                client.chat.completions.create(
                    model="k3", messages=[{"role": "user", "content": "hi"}]
                )
            )
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=complete) for _ in range(2)]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(timeout=2)

    assert not errors
    assert len(results) == 2
    assert max_active == 1


def test_async_cancellation_terminates_its_active_acp_child(tmp_path):
    """Cancelling an await must not strand its blocking worker subprocess."""
    from agent.copilot_acp_client import AsyncExternalACPClient, ExternalACPClient

    class BlockingStdin(_ProtocolStdin):
        def write(self, line):
            request = json.loads(line)
            if request["method"] == "session/prompt":
                self._process.requests.append(request)
                self._process.prompt_started.set()
                return len(line)
            return super().write(line)

    class BlockingProcess(_ProtocolProcess):
        def __init__(self):
            super().__init__()
            self.prompt_started = threading.Event()
            self.terminated = threading.Event()
            self.stdin = BlockingStdin(self)

        def terminate(self):
            self.terminated.set()
            super().terminate()

    process = BlockingProcess()
    sync_client = ExternalACPClient(
        provider="kimi-code",
        command="kimi",
        args=["acp"],
        acp_cwd=str(tmp_path),
    )
    async_client = AsyncExternalACPClient(sync_client)

    async def scenario():
        with patch(
            "agent.copilot_acp_client.subprocess.Popen", return_value=process
        ):
            task = asyncio.create_task(
                async_client.chat.completions.create(
                    model="k3", messages=[{"role": "user", "content": "wait"}]
                )
            )
            started = await asyncio.to_thread(process.prompt_started.wait, 1)
            assert started
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            stopped = await asyncio.to_thread(process.terminated.wait, 1)
            if not stopped:
                sync_client.close()
            assert stopped

    asyncio.run(scenario())


@pytest.mark.parametrize("reason", ["interrupt_abort", "stale_call_kill"])
def test_primary_abort_path_terminates_and_reaps_external_acp_child(
    tmp_path, reason
):
    from agent.copilot_acp_client import ExternalACPClient
    from run_agent import AIAgent

    class TrackingProcess(_ProtocolProcess):
        def __init__(self):
            super().__init__()
            self.terminated = False
            self.waited = False

        def terminate(self):
            self.terminated = True
            super().terminate()

        def wait(self, timeout=None):
            self.waited = True
            return super().wait(timeout=timeout)

    process = TrackingProcess()
    client = ExternalACPClient(
        provider="kimi-code",
        command="kimi",
        args=["acp"],
        acp_cwd=str(tmp_path),
    )
    with client._active_process_lock:
        client._active_process = cast(Any, process)
        client._active_cancel_event = threading.Event()

    agent = cast(Any, object.__new__(AIAgent))
    agent.provider = "kimi-code"
    agent.model = "k3"
    agent.base_url = "acp://kimi"

    AIAgent._abort_request_openai_client(agent, client, reason=reason)

    assert process.terminated is True
    assert process.waited is True
    assert client._active_process is None


def test_close_during_spawn_registration_gap_terminates_and_reaps_child(tmp_path):
    from agent.copilot_acp_client import ExternalACPClient

    spawn_started = threading.Event()
    release_spawn = threading.Event()

    class GapProcess(_ProtocolProcess):
        def __init__(self):
            super().__init__()
            self.terminated = threading.Event()
            self.waited = threading.Event()
            # Never answer initialize unless cancellation terminates the child.
            self.stdin = io.StringIO()

        def terminate(self):
            self.terminated.set()
            super().terminate()

        def wait(self, timeout=None):
            self.waited.set()
            return super().wait(timeout=timeout)

    process = GapProcess()

    def fake_popen(*_args, **_kwargs):
        spawn_started.set()
        assert release_spawn.wait(timeout=1)
        return process

    client = ExternalACPClient(
        provider="kimi-code",
        command="kimi",
        args=["acp"],
        acp_cwd=str(tmp_path),
    )
    result = {}

    def run_prompt():
        try:
            client._run_prompt("wait", timeout_seconds=5)
        except Exception as exc:
            result["error"] = exc

    with patch("agent.copilot_acp_client.subprocess.Popen", side_effect=fake_popen):
        worker = threading.Thread(target=run_prompt)
        worker.start()
        assert spawn_started.wait(timeout=1)
        client.close()
        release_spawn.set()
        worker.join(timeout=1)
        if worker.is_alive():
            client.close()
            worker.join(timeout=1)

    assert worker.is_alive() is False
    assert process.terminated.is_set()
    assert process.waited.is_set()
    assert "cancel" in str(result.get("error", "")).lower()


def test_close_before_prompt_start_prevents_child_spawn(tmp_path):
    from agent.copilot_acp_client import ExternalACPClient

    client = ExternalACPClient(
        provider="kimi-code",
        command="kimi",
        args=["acp"],
        acp_cwd=str(tmp_path),
    )
    client.close()

    with patch("agent.copilot_acp_client.subprocess.Popen") as popen:
        with pytest.raises(RuntimeError, match="closed|cancelled"):
            client._run_prompt("must not spawn", timeout_seconds=1)

    popen.assert_not_called()


def test_external_acp_client_supports_sequential_requests(tmp_path):
    from agent.copilot_acp_client import ExternalACPClient

    processes = [_ProtocolProcess(), _ProtocolProcess()]
    client = ExternalACPClient(
        provider="kimi-code",
        command="kimi",
        args=["acp"],
        acp_cwd=str(tmp_path),
    )

    with patch(
        "agent.copilot_acp_client.subprocess.Popen", side_effect=processes
    ):
        first = client._create_chat_completion(
            model="k3", messages=[{"role": "user", "content": "first"}]
        )
        second = client._create_chat_completion(
            model="k3", messages=[{"role": "user", "content": "second"}]
        )

    assert first.choices[0].message.content == "ok"
    assert second.choices[0].message.content == "ok"
    assert client.is_closed is False
    assert all(process.poll() == 0 for process in processes)


def test_external_acp_redacts_and_bounds_child_stderr(tmp_path):
    """CLI-owned stderr is untrusted and must never echo secrets unbounded."""
    from agent.copilot_acp_client import ExternalACPClient

    secret = "sk-proj-abcdefghijklmnopqrstuvwxyz123456"

    class ExitingProcess:
        def __init__(self):
            self.stdin = io.StringIO()
            self.stdout = io.StringIO("")
            self.stderr = io.StringIO(
                f"OPENAI_API_KEY={secret}\n" + ("diagnostic-x" * 1000)
            )
            self.started = time.monotonic()
            self._returncode = None

        def poll(self):
            if time.monotonic() - self.started > 0.02:
                self._returncode = 1
            return self._returncode

        def terminate(self):
            self._returncode = 1

        def wait(self, timeout=None):
            return self._returncode

        def kill(self):
            self._returncode = 1

    client = ExternalACPClient(
        provider="kimi-code",
        command="kimi",
        args=["acp"],
        acp_cwd=str(tmp_path),
    )
    with patch(
        "agent.copilot_acp_client.subprocess.Popen", return_value=ExitingProcess()
    ):
        with pytest.raises(RuntimeError) as excinfo:
            client._run_prompt("hello", timeout_seconds=0.2)

    message = str(excinfo.value)
    assert secret not in message
    assert "abcdefghijklmnopqrstuvwxyz123456" not in message
    assert len(message) < 5000
