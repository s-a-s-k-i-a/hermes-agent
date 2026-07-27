#!/usr/bin/env python3
"""Run a live Kimi ACP command against isolated copies of provider state.

The production Kimi credential store and Hermes auth.json are hashed before
and after the command.  Credential contents are copied into a temporary data
root but are never read into Python strings or printed.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def _file_hash(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(128 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_kimi_state(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise RuntimeError("Kimi Code data root was not found; run `kimi login` first.")
    shutil.copytree(
        source,
        destination,
        symlinks=True,
        ignore=shutil.ignore_patterns("sessions", "logs", "cache", "*.log"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a live Kimi/Hermes E2E command without mutating real auth state."
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command after --, for example: -- hermes doctor",
    )
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("provide a command after --")

    real_home = Path(os.environ.get("HOME") or Path.home()).expanduser()
    real_kimi = Path(
        os.environ.get("KIMI_CODE_HOME") or real_home / ".kimi-code"
    ).expanduser()
    real_hermes = Path(
        os.environ.get("HERMES_HOME") or real_home / ".hermes"
    ).expanduser()
    watched = {
        "Hermes auth store": real_hermes / "auth.json",
        "Kimi login marker": real_kimi / "credentials" / "kimi-code.json",
    }
    before = {label: _file_hash(path) for label, path in watched.items()}

    with tempfile.TemporaryDirectory(prefix="hermes-kimi-e2e-") as temp_text:
        temp_root = Path(temp_text)
        isolated_kimi = temp_root / "kimi-code"
        isolated_hermes = temp_root / "hermes"
        _copy_kimi_state(real_kimi, isolated_kimi)
        isolated_hermes.mkdir(mode=0o700)

        env = dict(os.environ)
        env["KIMI_CODE_HOME"] = str(isolated_kimi)
        env["HERMES_HOME"] = str(isolated_hermes)
        result = subprocess.run(command, env=env, check=False)

    after = {label: _file_hash(path) for label, path in watched.items()}
    changed = [label for label in watched if before[label] != after[label]]
    if changed:
        print(
            "E2E isolation failure: production state changed: " + ", ".join(changed),
            file=sys.stderr,
        )
        return 3
    print("E2E isolation verified: production auth files are hash-identical.")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
