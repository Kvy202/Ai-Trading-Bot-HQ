#!/usr/bin/env python
"""
tools/gen_env_example.py
========================
Generate .env.example from the live .env file by replacing secret values
with placeholders.  All non-secret lines (comments, blank lines, config
values) are preserved exactly.

Secret keys (values are masked):
    API_KEY, API_SECRET, API_PASSWORD,
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    SUPERVISOR_JWT_SECRET, SUPERVISOR_HMAC_SECRET,
    EX_KEYS_JSON

Usage:
    python tools/gen_env_example.py               # writes .env.example
    python tools/gen_env_example.py --dry-run     # print to stdout only
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

_ROOT   = Path(__file__).resolve().parents[1]
_ENV    = _ROOT / ".env"
_EXAMPLE = _ROOT / ".env.example"

SECRET_KEYS = {
    "API_KEY",
    "API_SECRET",
    "API_PASSWORD",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CONTROLLER_TOKEN",
    "TELEGRAM_NOTIFIER_TOKEN",
    "TELEGRAM_CHAT_ID",
    "SUPERVISOR_JWT_SECRET",
    "SUPERVISOR_HMAC_SECRET",
    "EX_KEYS_JSON",
}

_KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)")


def mask_line(line: str) -> str:
    m = _KEY_RE.match(line.rstrip("\n"))
    if m and m.group(1) in SECRET_KEYS:
        return f"{m.group(1)}=<YOUR_{m.group(1)}_HERE>\n"
    return line if line.endswith("\n") else line + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Generate .env.example from .env")
    p.add_argument("--dry-run", action="store_true",
                   help="Print to stdout instead of writing .env.example")
    args = p.parse_args()

    if not _ENV.exists():
        print(f"ERROR: {_ENV} not found")
        raise SystemExit(1)

    lines = _ENV.read_text(encoding="utf-8").splitlines(keepends=True)
    out_lines = [mask_line(line) for line in lines]

    if args.dry_run:
        print("".join(out_lines))
    else:
        _EXAMPLE.write_text("".join(out_lines), encoding="utf-8")
        masked = sum(1 for i, ln in enumerate(lines) if ln != out_lines[i])
        print(f"Written {_EXAMPLE}  ({masked} secret values masked)")


if __name__ == "__main__":
    main()
