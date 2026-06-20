"""Generate a Supervisor API JWT token for the Trady backend.

Usage
-----
    python tools/gen_supervisor_token.py
    python tools/gen_supervisor_token.py --sub trady-backend --days 90
    python tools/gen_supervisor_token.py --sub trady-dev --device android-debug --days 7

The token must be included in every API request:
    Authorization: Bearer <token>
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import sys
import time
from pathlib import Path

# Load .env from project root (no dependencies required)
_BASE = Path(__file__).resolve().parents[1]
_ENV  = _BASE / ".env"
if _ENV.exists():
    for _line in _ENV.read_text(encoding="utf-8", errors="ignore").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        _k, _v = _k.strip(), _v.strip().strip('"').strip("'")
        if _k and _k not in os.environ:
            os.environ[_k] = _v


def _b64url_enc(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _make_token(sub: str, device: str | None, ttl_days: int, secret: str) -> str:
    now = int(time.time())
    payload: dict = {"sub": sub, "iat": now, "exp": now + ttl_days * 86400}
    if device:
        payload["device"] = device
    header       = _b64url_enc(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    body         = _b64url_enc(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{header}.{body}"
    sig = hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{_b64url_enc(sig)}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Supervisor API JWT token")
    parser.add_argument("--sub",    default="trady-backend", help="Subject / actor name")
    parser.add_argument("--device", default=None,            help="Device ID (optional)")
    parser.add_argument("--days",   type=int, default=30,    help="Token validity in days")
    args = parser.parse_args()

    secret = os.environ.get("SUPERVISOR_JWT_SECRET", "")
    if not secret:
        print("ERROR: SUPERVISOR_JWT_SECRET is not set.", file=sys.stderr)
        print("", file=sys.stderr)
        print("Add to .env:", file=sys.stderr)
        print("    SUPERVISOR_JWT_SECRET=<value>", file=sys.stderr)
        print("", file=sys.stderr)
        print("Generate a value:", file=sys.stderr)
        print('    python -c "import secrets; print(secrets.token_hex(32))"', file=sys.stderr)
        sys.exit(1)

    token = _make_token(args.sub, args.device, args.days, secret)
    expiry = time.strftime("%Y-%m-%d", time.localtime(time.time() + args.days * 86400))
    print(f"Subject : {args.sub}")
    if args.device:
        print(f"Device  : {args.device}")
    print(f"Expires : {expiry} ({args.days} days)")
    print()
    print("Token:")
    print(token)
    print()
    print("Authorization header:")
    print(f"    Authorization: Bearer {token}")


if __name__ == "__main__":
    main()
