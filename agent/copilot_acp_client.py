"""OpenAI-compatible shim for external-process ACP model providers.

Each request starts a short-lived ACP session, sends the formatted conversation
as a single prompt, collects text chunks, and converts the result back into the
minimal shape Hermes expects from an OpenAI client. ``CopilotACPClient`` stays
as the backward-compatible public facade for the original Copilot transport.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import shlex
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from agent.file_safety import get_read_block_error, get_write_denied_error
from agent.redact import redact_sensitive_text
from tools.environments.local import hermes_subprocess_env

ACP_MARKER_BASE_URL = "acp://copilot"
_DEFAULT_TIMEOUT_SECONDS = 900.0
_MAX_CHILD_DIAGNOSTIC_CHARS = 4096


def _safe_child_diagnostic(text: Any) -> str:
    """Bound and forcibly redact untrusted child-process diagnostics."""
    cleaned = str(text or "").strip()
    if len(cleaned) > _MAX_CHILD_DIAGNOSTIC_CHARS:
        half = _MAX_CHILD_DIAGNOSTIC_CHARS // 2
        cleaned = (
            cleaned[:half]
            + "\n...[child diagnostic truncated]...\n"
            + cleaned[-half:]
        )
    return redact_sensitive_text(cleaned, force=True) or ""

_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_TOOL_CALL_JSON_RE = re.compile(r"\{\s*\"id\"\s*:\s*\"[^\"]+\"\s*,\s*\"type\"\s*:\s*\"function\"\s*,\s*\"function\"\s*:\s*\{.*?\}\s*\}", re.DOTALL)

# Stderr fingerprint of the deprecated `gh copilot` CLI extension
# (https://github.blog/changelog/2025-09-25-upcoming-deprecation-of-gh-copilot-cli-extension).
# We require BOTH the literal product name ("gh-copilot") AND a deprecation
# marker, so generic stderr from the NEW `@github/copilot` CLI — whose repo
# is github.com/github/copilot-cli and which legitimately mentions "copilot-cli"
# in its own banners and error messages — doesn't get misclassified as the
# deprecated extension.
_DEPRECATION_REQUIRED = ("gh-copilot",)
_DEPRECATION_MARKERS = (
    "has been deprecated",
    "no commands will be executed",
)


def _is_gh_copilot_deprecation_message(stderr_text: str) -> bool:
    """True iff stderr looks like the deprecated gh-copilot extension's banner."""

    lower = stderr_text.lower()
    if not any(req in lower for req in _DEPRECATION_REQUIRED):
        return False
    return any(marker in lower for marker in _DEPRECATION_MARKERS)


def _resolve_command() -> str:
    return (
        os.getenv("HERMES_COPILOT_ACP_COMMAND", "").strip()
        or os.getenv("COPILOT_CLI_PATH", "").strip()
        or "copilot"
    )


def _resolve_args() -> list[str]:
    raw = os.getenv("HERMES_COPILOT_ACP_ARGS", "").strip()
    if not raw:
        return ["--acp", "--stdio"]
    return shlex.split(raw)


def is_external_acp_runtime(provider: str | None, base_url: str | None) -> bool:
    """Return whether a runtime must use the local ACP client, not HTTP."""
    normalized_url = str(base_url or "").strip().lower()
    if normalized_url.startswith(("acp://", "acp+tcp://")):
        return True
    normalized_provider = str(provider or "").strip().lower()
    if not normalized_provider:
        return False
    try:
        from providers import get_provider_profile

        profile = get_provider_profile(normalized_provider)
        return bool(profile and profile.auth_type == "external_process")
    except Exception:
        return normalized_provider == "copilot-acp"


def _resolve_external_defaults(provider: str) -> tuple[str, list[str]]:
    """Resolve declarative launch defaults without touching credentials."""
    try:
        from hermes_cli.auth import _resolve_external_process_launch

        return _resolve_external_process_launch(provider)
    except Exception:
        if provider == "copilot-acp":
            return _resolve_command(), _resolve_args()
        return "", []


def _resolve_home_dir() -> str:
    """Return a stable HOME for child ACP processes."""
    home = os.environ.get("HOME", "").strip()
    if home:
        return home

    expanded = os.path.expanduser("~")
    if expanded and expanded != "~":
        return expanded

    try:
        import pwd

        resolved = pwd.getpwuid(os.getuid()).pw_dir.strip()  # windows-footgun: ok — POSIX fallback inside try/except (pwd import fails on Windows)
        if resolved:
            return resolved
    except Exception:
        pass

    # Last resort: /tmp (writable on any POSIX system). Avoids crashing the
    # subprocess with no HOME; callers can set HERMES_HOME explicitly if they
    # need a different writable dir.
    return "/tmp"


def _build_subprocess_env(provider: str = "copilot-acp") -> dict[str, str]:
    # Preserve Copilot's established provider-key inheritance. Other external
    # ACP providers authenticate through their own HOME-scoped session and do
    # not receive Hermes' provider credentials. Tier-1 secrets are always
    # stripped by the central helper (#29157).
    if provider == "kimi-code":
        from hermes_cli.auth import build_external_process_subprocess_env

        return build_external_process_subprocess_env(provider)
    env = hermes_subprocess_env(inherit_credentials=provider == "copilot-acp")
    home = _resolve_home_dir()
    env["HOME"] = home
    from hermes_constants import apply_subprocess_home_env
    apply_subprocess_home_env(env)
    # Keep both home variables coherent for the child.  The central helper
    # may inherit a parent HERMES_REAL_HOME that no longer matches HOME (for
    # example in tests, profiles or supervised launches).
    env["HERMES_REAL_HOME"] = home
    return env


def _jsonrpc_error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "error": {
            "code": code,
            "message": message,
        },
    }


def _permission_denied(
    message_id: Any, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Select an offered one-shot rejection, or cancel when none is safe."""
    options = (params or {}).get("options") or []
    reject_option_id = next(
        (
            str(option.get("optionId") or "").strip()
            for option in options
            if isinstance(option, dict)
            and str(option.get("kind") or "").strip() == "reject_once"
            and str(option.get("optionId") or "").strip()
        ),
        "",
    )
    outcome = (
        {"outcome": "selected", "optionId": reject_option_id}
        if reject_option_id
        else {"outcome": "cancelled"}
    )
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "result": {"outcome": outcome},
    }


def _format_messages_as_prompt(
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
) -> str:
    sections: list[str] = [
        "You are being used as the active ACP agent backend for Hermes.",
        "Use ACP capabilities to complete tasks.",
        "IMPORTANT: If you take an action with a tool, you MUST output tool calls using <tool_call>{...}</tool_call> blocks with JSON exactly in OpenAI function-call shape.",
        "If no tool is needed, answer normally.",
    ]
    if model:
        sections.append(f"Hermes requested model hint: {model}")

    if isinstance(tools, list) and tools:
        tool_specs: list[dict[str, Any]] = []
        for t in tools:
            if not isinstance(t, dict):
                continue
            fn = t.get("function") or {}
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            tool_specs.append(
                {
                    "name": name.strip(),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                }
            )
        if tool_specs:
            sections.append(
                "Available tools (OpenAI function schema). "
                "When using a tool, emit ONLY <tool_call>{...}</tool_call> with one JSON object "
                "containing id/type/function{name,arguments}. arguments must be a JSON string.\n"
                + json.dumps(tool_specs, ensure_ascii=False)
            )

    if tool_choice is not None:
        sections.append(f"Tool choice hint: {json.dumps(tool_choice, ensure_ascii=False)}")

    transcript: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown").strip().lower()
        if role == "tool":
            role = "tool"
        elif role not in {"system", "user", "assistant"}:
            role = "context"

        content = message.get("content")
        rendered = _render_message_content(content)
        if not rendered:
            continue

        label = {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
            "tool": "Tool",
            "context": "Context",
        }.get(role, role.title())
        transcript.append(f"{label}:\n{rendered}")

    if transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))

    sections.append("Continue the conversation from the latest user request.")
    return "\n\n".join(section.strip() for section in sections if section and section.strip())


def _render_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text") or "").strip()
        if "content" in content and isinstance(content.get("content"), str):
            return str(content.get("content") or "").strip()
        return json.dumps(content, ensure_ascii=True)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()
    return str(content).strip()


def _build_openai_tool_call(
    *,
    call_id: str,
    name: str,
    arguments: str,
) -> ChatCompletionMessageToolCall:
    """Build an OpenAI-compatible tool-call object for downstream handling."""
    return ChatCompletionMessageToolCall(
        id=call_id,
        call_id=call_id,
        response_item_id=None,
        type="function",
        function=Function(name=name, arguments=arguments),
    )


def _completion_to_stream_chunks(completion: SimpleNamespace) -> list[SimpleNamespace]:
    """Convert a one-shot ACP response into OpenAI-style stream chunks."""
    choice = completion.choices[0]
    message = choice.message
    tool_call_deltas = None
    if message.tool_calls:
        tool_call_deltas = []
        for index, tool_call in enumerate(message.tool_calls):
            tool_call_deltas.append(
                SimpleNamespace(
                    index=index,
                    id=getattr(tool_call, "id", None),
                    type=getattr(tool_call, "type", "function"),
                    function=SimpleNamespace(
                        name=getattr(tool_call.function, "name", None),
                        arguments=getattr(tool_call.function, "arguments", None),
                    ),
                )
            )

    delta = SimpleNamespace(
        role="assistant",
        content=message.content or None,
        tool_calls=tool_call_deltas,
        reasoning_content=message.reasoning_content,
        reasoning=message.reasoning,
    )
    data_chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                index=0,
                delta=delta,
                finish_reason=choice.finish_reason,
            )
        ],
        model=completion.model,
        usage=None,
    )
    usage_chunk = SimpleNamespace(
        choices=[],
        model=completion.model,
        usage=completion.usage,
    )
    return [data_chunk, usage_chunk]


def _extract_tool_calls_from_text(text: str) -> tuple[list[ChatCompletionMessageToolCall], str]:
    if not isinstance(text, str) or not text.strip():
        return [], ""

    extracted: list[ChatCompletionMessageToolCall] = []
    consumed_spans: list[tuple[int, int]] = []

    def _try_add_tool_call(raw_json: str) -> None:
        try:
            obj = json.loads(raw_json)
        except Exception:
            return
        if not isinstance(obj, dict):
            return
        fn = obj.get("function")
        if not isinstance(fn, dict):
            return
        fn_name = fn.get("name")
        if not isinstance(fn_name, str) or not fn_name.strip():
            return
        fn_args = fn.get("arguments", "{}")
        if not isinstance(fn_args, str):
            fn_args = json.dumps(fn_args, ensure_ascii=False)
        call_id = obj.get("id")
        if not isinstance(call_id, str) or not call_id.strip():
            call_id = f"acp_call_{len(extracted)+1}"

        extracted.append(
            _build_openai_tool_call(
                call_id=call_id,
                name=fn_name.strip(),
                arguments=fn_args,
            )
        )

    for m in _TOOL_CALL_BLOCK_RE.finditer(text):
        raw = m.group(1)
        _try_add_tool_call(raw)
        consumed_spans.append((m.start(), m.end()))

    # Only try bare-JSON fallback when no XML blocks were found.
    if not extracted:
        for m in _TOOL_CALL_JSON_RE.finditer(text):
            raw = m.group(0)
            _try_add_tool_call(raw)
            consumed_spans.append((m.start(), m.end()))

    if not consumed_spans:
        return extracted, text.strip()

    consumed_spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in consumed_spans:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))

    parts: list[str] = []
    cursor = 0
    for start, end in merged:
        if cursor < start:
            parts.append(text[cursor:start])
        cursor = max(cursor, end)
    if cursor < len(text):
        parts.append(text[cursor:])

    cleaned = "\n".join(p.strip() for p in parts if p and p.strip()).strip()
    return extracted, cleaned



def _ensure_path_within_cwd(path_text: str, cwd: str) -> Path:
    candidate = Path(path_text)
    if not candidate.is_absolute():
        raise PermissionError("ACP file-system paths must be absolute.")
    resolved = candidate.resolve()
    root = Path(cwd).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PermissionError(f"Path '{resolved}' is outside the session cwd '{root}'.") from exc
    return resolved


class _ACPChatCompletions:
    def __init__(self, client: "ExternalACPClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _ACPChatNamespace:
    def __init__(self, client: "ExternalACPClient"):
        self.completions = _ACPChatCompletions(client)


class ExternalACPClient:
    """Minimal OpenAI-client-compatible facade for an ACP subprocess."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        acp_command: str | None = None,
        acp_args: list[str] | None = None,
        acp_cwd: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        provider: str = "external-acp",
        allow_file_reads: bool | None = None,
        allow_file_writes: bool = False,
        **_: Any,
    ):
        self.provider = provider
        self._acp_label = (
            "Copilot ACP"
            if provider == "copilot-acp"
            else "Kimi Code ACP"
            if provider == "kimi-code"
            else f"{provider} ACP"
        )
        self.api_key = api_key or (
            "copilot-acp" if provider == "copilot-acp" else "external-process"
        )
        self.base_url = base_url or (
            ACP_MARKER_BASE_URL if provider == "copilot-acp" else f"acp://{provider}"
        )
        self._default_headers = dict(default_headers or {})
        default_command, default_args = _resolve_external_defaults(provider)
        self._acp_command = acp_command or command or default_command
        self._acp_args = list(acp_args or args or default_args)
        self._acp_cwd = str(Path(acp_cwd or os.getcwd()).resolve())
        self._allow_file_reads = (
            provider == "copilot-acp"
            if allow_file_reads is None
            else allow_file_reads
        )
        self._allow_file_writes = allow_file_writes
        self.chat = _ACPChatNamespace(self)
        self.is_closed = False
        self._request_lock = threading.Lock()
        self._active_process: subprocess.Popen[str] | None = None
        self._active_cancel_event: threading.Event | None = None
        self._active_process_lock = threading.Lock()

    def close(self) -> None:
        proc: subprocess.Popen[str] | None
        with self._active_process_lock:
            self.is_closed = True
            cancel_event = self._active_cancel_event
            if cancel_event is not None:
                cancel_event.set()
            proc = self._active_process
            self._active_process = None
            self._active_cancel_event = None
        self._terminate_process(proc)

    @staticmethod
    def _terminate_process(proc: subprocess.Popen[str] | None) -> None:
        if proc is None:
            return
        try:
            if proc.poll() is not None:
                proc.wait(timeout=0)
                return
        except Exception:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=2)
            return
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            proc.wait(timeout=2)
        except Exception:
            pass

    def _finish_request(
        self,
        proc: subprocess.Popen[str],
        cancel_event: threading.Event,
    ) -> None:
        """Release one completed request without permanently closing the client."""
        with self._active_process_lock:
            if self._active_cancel_event is cancel_event:
                if self._active_process is proc:
                    self._active_process = None
                self._active_cancel_event = None
        self._terminate_process(proc)

    def _cancel_request(self, cancel_event: threading.Event) -> None:
        """Cancel only the subprocess owned by one async request."""
        cancel_event.set()
        proc: subprocess.Popen[str] | None = None
        with self._active_process_lock:
            if self._active_cancel_event is cancel_event:
                proc = self._active_process
                self._active_process = None
                self._active_cancel_event = None
        self._terminate_process(proc)

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        stream: bool = False,
        _cancel_event: threading.Event | None = None,
        **kwargs: Any,
    ) -> Any:
        """Serialize one full ACP subprocess lifecycle per client instance."""
        with self._request_lock:
            with self._active_process_lock:
                if self.is_closed:
                    raise RuntimeError(f"{self._acp_label} client is closed.")
            if _cancel_event is not None and _cancel_event.is_set():
                raise RuntimeError(f"{self._acp_label} request was cancelled.")
            return self._create_chat_completion_locked(
                model=model,
                messages=messages,
                timeout=timeout,
                tools=tools,
                tool_choice=tool_choice,
                stream=stream,
                _cancel_event=_cancel_event,
                **kwargs,
            )

    def _create_chat_completion_locked(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        stream: bool = False,
        _cancel_event: threading.Event | None = None,
        **_: Any,
    ) -> Any:
        prompt_text = _format_messages_as_prompt(
            messages or [],
            model=model,
            tools=tools,
            tool_choice=tool_choice,
        )
        if not self._allow_file_writes:
            prompt_text = (
                "ACP SAFETY: Do not execute side-effecting tools directly. "
                "Request every side effect by emitting the required Hermes "
                "<tool_call>{...}</tool_call> block; Hermes will apply its "
                "own permission policy. If that is not possible, do not "
                "perform the action.\n\n"
                + prompt_text
            )
        # Normalise timeout: run_agent.py may pass an httpx.Timeout object
        # (used natively by the OpenAI SDK) rather than a plain float.
        if timeout is None:
            _effective_timeout = _DEFAULT_TIMEOUT_SECONDS
        elif isinstance(timeout, (int, float)):
            _effective_timeout = float(timeout)
        else:
            # httpx.Timeout or similar — pick the largest component so the
            # subprocess has enough wall-clock time for the full response.
            _candidates = [
                getattr(timeout, attr, None)
                for attr in ("read", "write", "connect", "pool", "timeout")
            ]
            _numeric = [float(v) for v in _candidates if isinstance(v, (int, float))]
            _effective_timeout = max(_numeric) if _numeric else _DEFAULT_TIMEOUT_SECONDS

        run_kwargs: dict[str, Any] = {"timeout_seconds": _effective_timeout}
        if self.provider == "kimi-code":
            run_kwargs["model"] = model
        if _cancel_event is not None:
            run_kwargs["cancel_event"] = _cancel_event
        response_text, reasoning_text = self._run_prompt(prompt_text, **run_kwargs)

        tool_calls, cleaned_text = _extract_tool_calls_from_text(response_text)

        usage = SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        assistant_message = SimpleNamespace(
            content=cleaned_text,
            tool_calls=tool_calls,
            reasoning=reasoning_text or None,
            reasoning_content=reasoning_text or None,
            reasoning_details=None,
        )
        finish_reason = "tool_calls" if tool_calls else "stop"
        choice = SimpleNamespace(message=assistant_message, finish_reason=finish_reason)
        completion = SimpleNamespace(
            choices=[choice],
            usage=usage,
            model=model or self.provider,
        )
        if stream:
            return _completion_to_stream_chunks(completion)
        return completion

    def _run_prompt(
        self,
        prompt_text: str,
        *,
        timeout_seconds: float,
        model: str | None = None,
        cancel_event: threading.Event | None = None,
    ) -> tuple[str, str]:
        request_cancel_event = cancel_event or threading.Event()
        # Publish cancellation ownership before Popen. A close arriving while
        # the constructor is blocked must be visible immediately after spawn.
        with self._active_process_lock:
            if self.is_closed:
                raise RuntimeError(f"{self._acp_label} client is closed.")
            self._active_process = None
            self._active_cancel_event = request_cancel_event
        try:
            # Hide the console the CLI child would otherwise flash on Windows
            # (#56747). Hide-only — stdio pipes stay intact for the ACP wire.
            from hermes_cli._subprocess_compat import windows_hide_flags

            proc = subprocess.Popen(
                [self._acp_command] + self._acp_args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True, encoding='utf-8', errors='replace',
                bufsize=1,
                cwd=self._acp_cwd,
                env=_build_subprocess_env(self.provider),
                creationflags=windows_hide_flags(),
            )
        except FileNotFoundError as exc:
            with self._active_process_lock:
                if self._active_cancel_event is request_cancel_event:
                    self._active_cancel_event = None
            raise RuntimeError(
                f"Could not start {self._acp_label} command '{self._acp_command}'."
            ) from exc

        if proc.stdin is None or proc.stdout is None or proc.stderr is None:
            self._terminate_process(proc)
            raise RuntimeError(
                f"{self._acp_label} process did not expose stdin/stdout pipes."
            )
        proc_stdin = proc.stdin

        cancelled_before_start = False
        with self._active_process_lock:
            if (
                request_cancel_event.is_set()
                or self._active_cancel_event is not request_cancel_event
            ):
                cancelled_before_start = True
            else:
                self._active_process = proc
                self._active_cancel_event = request_cancel_event
        if cancelled_before_start:
            self._terminate_process(proc)
            raise RuntimeError(f"{self._acp_label} request was cancelled.")

        inbox: queue.Queue[dict[str, Any]] = queue.Queue()
        stderr_tail: deque[str] = deque(maxlen=40)

        def _stdout_reader() -> None:
            if proc.stdout is None:
                return
            for line in proc.stdout:
                try:
                    inbox.put(json.loads(line))
                except Exception:
                    inbox.put({"raw": line.rstrip("\n")})

        def _stderr_reader() -> None:
            if proc.stderr is None:
                return
            for line in proc.stderr:
                stderr_tail.append(line.rstrip("\n"))

        out_thread = threading.Thread(target=_stdout_reader, daemon=True)
        err_thread = threading.Thread(target=_stderr_reader, daemon=True)
        out_thread.start()
        err_thread.start()

        next_id = 0

        def _request(method: str, params: dict[str, Any], *, text_parts: list[str] | None = None, reasoning_parts: list[str] | None = None) -> Any:
            nonlocal next_id
            next_id += 1
            request_id = next_id
            payload = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
            if request_cancel_event.is_set():
                raise RuntimeError(f"{self._acp_label} request was cancelled.")
            proc_stdin.write(json.dumps(payload) + "\n")
            proc_stdin.flush()

            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
                if request_cancel_event.is_set():
                    raise RuntimeError(f"{self._acp_label} request was cancelled.")
                if proc.poll() is not None:
                    break
                try:
                    msg = inbox.get(timeout=0.1)
                except queue.Empty:
                    continue

                if self._handle_server_message(
                    msg,
                    process=proc,
                    cwd=self._acp_cwd,
                    text_parts=text_parts,
                    reasoning_parts=reasoning_parts,
                ):
                    continue

                if msg.get("id") != request_id:
                    continue
                if "error" in msg:
                    err = msg.get("error") or {}
                    detail = _safe_child_diagnostic(
                        err.get("message") if isinstance(err, dict) else err
                    )
                    raise RuntimeError(
                        f"{self._acp_label} {method} failed: {detail}"
                    )
                return msg.get("result")

            stderr_text = _safe_child_diagnostic("\n".join(stderr_tail))
            if proc.poll() is not None and stderr_text:
                if (
                    self.provider == "copilot-acp"
                    and _is_gh_copilot_deprecation_message(stderr_text)
                ):
                    raise RuntimeError(
                        "Hermes ACP mode requires the NEW GitHub Copilot CLI "
                        "(github.com/github/copilot-cli), but the binary it just "
                        "spawned is the deprecated `gh copilot` extension.\n\n"
                        "Install the new CLI:\n"
                        "  npm install -g @github/copilot\n"
                        "  # then verify with: copilot --help\n\n"
                        "If `copilot` already resolves to the new CLI but you still see this,\n"
                        "point Hermes at it explicitly:\n"
                        "  export HERMES_COPILOT_ACP_COMMAND=/path/to/new/copilot\n\n"
                        "Alternative: use the `copilot` provider (no ACP, hits the Copilot API\n"
                        "directly with a Copilot subscription token) via `hermes setup`.\n\n"
                        f"Original error:\n{stderr_text}"
                    )
                raise RuntimeError(
                    f"{self._acp_label} process exited early: {stderr_text}"
                )
            raise TimeoutError(
                f"Timed out waiting for {self._acp_label} response to {method}."
            )

        try:
            initialize_result = _request(
                "initialize",
                {
                    "protocolVersion": 1,
                    "clientCapabilities": {
                        "fs": {
                            "readTextFile": self._allow_file_reads,
                            "writeTextFile": self._allow_file_writes,
                        }
                    },
                    "clientInfo": {
                        "name": "hermes-agent",
                        "title": "Hermes Agent",
                        "version": "0.0.0",
                    },
                },
            ) or {}

            # Kimi owns its OAuth session.  Authenticate through the ACP
            # method it advertises; no token ever crosses into Hermes.
            if self.provider == "kimi-code":
                auth_methods = initialize_result.get("authMethods") or []
                login_method = next(
                    (
                        str(item.get("id") or "").strip()
                        for item in auth_methods
                        if isinstance(item, dict)
                        and str(item.get("id") or "").strip() == "login"
                    ),
                    "",
                )
                if not login_method:
                    raise RuntimeError(
                        "This Kimi Code ACP server did not advertise its "
                        "CLI-owned login method. Upgrade with `kimi upgrade`, "
                        "then retry."
                    )
                _request("authenticate", {"methodId": login_method})

            session = _request(
                "session/new",
                {
                    "cwd": self._acp_cwd,
                    "mcpServers": [],
                },
            ) or {}
            session_id = str(session.get("sessionId") or "").strip()
            if not session_id:
                raise RuntimeError(
                    f"{self._acp_label} did not return a sessionId."
                )

            # A model name in the prompt is only advisory.  Bind Kimi's real
            # ACP session to the selected model before any user content is
            # submitted, and fail closed if the server rejects it.
            if self.provider == "kimi-code" and model:
                requested_model = str(model).strip()
                kimi_model = {
                    "k3": "kimi-code/k3",
                    "kimi-k3": "kimi-code/k3",
                    "k3-256k": "kimi-code/k3-256k",
                    "kimi-k3-256k": "kimi-code/k3-256k",
                }.get(requested_model, requested_model)
                _request(
                    "session/set_config_option",
                    {
                        "sessionId": session_id,
                        "configId": "model",
                        "value": kimi_model,
                    },
                )

            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            _request(
                "session/prompt",
                {
                    "sessionId": session_id,
                    "prompt": [
                        {
                            "type": "text",
                            "text": prompt_text,
                        }
                    ],
                },
                text_parts=text_parts,
                reasoning_parts=reasoning_parts,
            )
            return "".join(text_parts), "".join(reasoning_parts)
        finally:
            self._finish_request(proc, request_cancel_event)

    def _handle_server_message(
        self,
        msg: dict[str, Any],
        *,
        process: subprocess.Popen[str],
        cwd: str,
        text_parts: list[str] | None,
        reasoning_parts: list[str] | None,
    ) -> bool:
        method = msg.get("method")
        if not isinstance(method, str):
            return False

        if method == "session/update":
            params = msg.get("params") or {}
            update = params.get("update") or {}
            kind = str(update.get("sessionUpdate") or "").strip()
            content = update.get("content") or {}
            chunk_text = ""
            if isinstance(content, dict):
                chunk_text = str(content.get("text") or "")
            if kind == "agent_message_chunk" and chunk_text and text_parts is not None:
                text_parts.append(chunk_text)
            elif kind == "agent_thought_chunk" and chunk_text and reasoning_parts is not None:
                reasoning_parts.append(chunk_text)
            return True

        if process.stdin is None:
            return True

        message_id = msg.get("id")
        params = msg.get("params") or {}

        if method == "session/request_permission":
            response = _permission_denied(message_id, params)
        elif method == "fs/read_text_file":
            try:
                if not self._allow_file_reads:
                    raise PermissionError(
                        "Direct ACP file reads are disabled. Emit a Hermes "
                        "tool-call block so Hermes can apply its file-safety policy."
                    )
                path = _ensure_path_within_cwd(str(params.get("path") or ""), cwd)
                block_error = get_read_block_error(str(path))
                if block_error:
                    raise PermissionError(block_error)
                try:
                    content = path.read_text(encoding="utf-8")
                except FileNotFoundError:
                    content = ""
                line = params.get("line")
                limit = params.get("limit")
                if isinstance(line, int) and line > 1:
                    lines = content.splitlines(keepends=True)
                    start = line - 1
                    end = start + limit if isinstance(limit, int) and limit > 0 else None
                    content = "".join(lines[start:end])
                if content:
                    content = redact_sensitive_text(content, force=True)
                response = {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "result": {
                        "content": content,
                    },
                }
            except Exception as exc:
                response = _jsonrpc_error(message_id, -32602, str(exc))
        elif method == "fs/write_text_file":
            try:
                if not self._allow_file_writes:
                    raise PermissionError(
                        "Direct ACP file writes are disabled. Emit a Hermes "
                        "tool-call block so Hermes can apply its approval policy."
                    )
                path = _ensure_path_within_cwd(str(params.get("path") or ""), cwd)
                denied = get_write_denied_error(str(path))
                if denied:
                    raise PermissionError(denied)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(str(params.get("content") or ""), encoding="utf-8")
                response = {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "result": None,
                }
            except Exception as exc:
                response = _jsonrpc_error(message_id, -32602, str(exc))
        else:
            response = _jsonrpc_error(
                message_id,
                -32601,
                f"ACP client method '{method}' is not supported by Hermes yet.",
            )

        process.stdin.write(json.dumps(response) + "\n")
        process.stdin.flush()
        return True


class _AsyncExternalACPCompletions:
    """Awaitable facade that keeps blocking ACP subprocess I/O off the loop."""

    def __init__(self, sync_client: ExternalACPClient):
        self._sync_client = sync_client

    async def create(self, **kwargs: Any):
        cancel_event = threading.Event()
        try:
            return await asyncio.to_thread(
                self._sync_client.chat.completions.create,
                _cancel_event=cancel_event,
                **kwargs,
            )
        except asyncio.CancelledError:
            self._sync_client._cancel_request(cancel_event)
            raise


class AsyncExternalACPClient:
    """Minimal async OpenAI-style facade around ``ExternalACPClient``."""

    def __init__(self, sync_client: ExternalACPClient):
        self._sync_client = sync_client
        self.api_key = sync_client.api_key
        self.base_url = sync_client.base_url
        self.chat = SimpleNamespace(
            completions=_AsyncExternalACPCompletions(sync_client)
        )

    @property
    def is_closed(self) -> bool:
        return self._sync_client.is_closed

    def close(self) -> None:
        self._sync_client.close()

    async def aclose(self) -> None:
        await asyncio.to_thread(self._sync_client.close)

    async def __aenter__(self) -> "AsyncExternalACPClient":
        return self

    async def __aexit__(self, *_exc_info: Any) -> None:
        await self.aclose()


class CopilotACPClient(ExternalACPClient):
    """Backward-compatible GitHub Copilot ACP client."""

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("provider", "copilot-acp")
        # Preserve the original Copilot ACP filesystem behavior. Generic ACP
        # providers default to no direct writes.
        kwargs.setdefault("allow_file_writes", True)
        super().__init__(**kwargs)
