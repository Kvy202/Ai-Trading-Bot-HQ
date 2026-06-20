"""JWT + HMAC-SHA256 authentication for the Supervisor API.

Required env vars
-----------------
SUPERVISOR_JWT_SECRET   -- 32+ random bytes as hex
SUPERVISOR_HMAC_SECRET  -- 32+ random bytes as hex

Generate both with:
    python -c "import secrets; print(secrets.token_hex(32))"

HMAC signing format for API requests
--------------------------------------
Clients must include three extra headers on every POST/PUT/PATCH/DELETE:
    X-Timestamp : Unix epoch float (seconds), e.g. "1746624000.123"
    X-Nonce     : random string, unique per request
    X-Signature : HMAC-SHA256 of the canonical message (hex digest)

Canonical message (newline-separated, no trailing newline):
    METHOD
    /absolute/path
    query_string_or_empty
    timestamp
    nonce
    sha256(raw_body_hex)

Binding to method + path prevents a signed empty-body request from being
replayed to a different endpoint.

HMAC signing format for IPC command files
------------------------------------------
Commands written to supervisor_cmd.json include two signed fields:
    cmd_sig         : HMAC( "{command}:{actor}:{issued_at:.3f}" )
    approval_marker : HMAC( "approve:{id}:{actor}:{command}:{issued_at:.3f}" )
                      (only present on live-mode resume commands)

These let the executor reject commands that were not written by this API,
even if the filesystem is compromised.
"""
from __future__ import annotations

import base64
import hashlib
import hmac as _hmac
import json
import os
import time
from collections import OrderedDict
from functools import wraps

from flask import g, jsonify, request

# ---------------------------------------------------------------------------
# Nonce replay cache
# ---------------------------------------------------------------------------

_NONCE_TTL = 300.0  # seconds — must match X-Timestamp tolerance
_seen_nonces: OrderedDict[str, float] = OrderedDict()


def _prune_nonces() -> None:
    cutoff = time.time() - _NONCE_TTL
    while _seen_nonces and next(iter(_seen_nonces.values())) < cutoff:
        _seen_nonces.popitem(last=False)


# ---------------------------------------------------------------------------
# Minimal HS256 JWT (stdlib only — no PyJWT dependency)
# ---------------------------------------------------------------------------

def _b64url_enc(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_dec(s: str) -> bytes:
    rem = len(s) % 4
    if rem:
        s += "=" * (4 - rem)
    return base64.urlsafe_b64decode(s)


_JWT_HEADER = _b64url_enc(
    json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode()
)


def encode_jwt(payload: dict, secret: str | None = None) -> str:
    """Create a signed HS256 JWT. Uses SUPERVISOR_JWT_SECRET when secret is None."""
    key = (secret or os.environ["SUPERVISOR_JWT_SECRET"]).encode()
    body = _b64url_enc(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{_JWT_HEADER}.{body}"
    sig = _hmac.new(key, signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{_b64url_enc(sig)}"


def decode_jwt(token: str, secret: str | None = None) -> dict:
    """Verify and decode a HS256 JWT. Raises ValueError on any failure."""
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("malformed token")
    header, body, sig_b64 = parts
    key = (secret or os.environ["SUPERVISOR_JWT_SECRET"]).encode()
    signing_input = f"{header}.{body}"
    expected = _hmac.new(key, signing_input.encode(), hashlib.sha256).digest()
    actual = _b64url_dec(sig_b64)
    if not _hmac.compare_digest(expected, actual):
        raise ValueError("invalid signature")
    payload = json.loads(_b64url_dec(body))
    if "exp" in payload and float(payload["exp"]) < time.time():
        raise ValueError("token expired")
    if "nbf" in payload and float(payload["nbf"]) > time.time():
        raise ValueError("token not yet valid")
    return payload


# ---------------------------------------------------------------------------
# HMAC-SHA256 request signing
# ---------------------------------------------------------------------------

def verify_hmac_request(body: bytes, sig: str, ts_str: str, nonce: str,
                         method: str = "", path: str = "", query: str = "") -> bool:
    """Verify X-Signature / X-Timestamp / X-Nonce on a state-changing request.

    Canonical message (newline-separated):
        METHOD\\nPATH\\nQUERY\\nTIMESTAMP\\nNONCE\\nSHA256(body)

    Binding method+path prevents a signed empty-body request from being
    redirected to a different endpoint. Returns False (never raises) on failure.
    """
    secret = os.environ.get("SUPERVISOR_HMAC_SECRET", "")
    if not secret:
        return False
    try:
        ts = float(ts_str)
    except (TypeError, ValueError):
        return False
    if abs(time.time() - ts) > _NONCE_TTL:
        return False  # stale timestamp

    nonce_key = f"{method}:{path}:{ts_str}:{nonce}"
    _prune_nonces()
    if nonce_key in _seen_nonces:
        return False  # replay attack

    body_hash = hashlib.sha256(body).hexdigest()
    canonical = f"{method}\n{path}\n{query}\n{ts_str}\n{nonce}\n{body_hash}".encode()
    expected  = _hmac.new(secret.encode(), canonical, hashlib.sha256).hexdigest()
    if not _hmac.compare_digest(expected, sig):
        return False

    _seen_nonces[nonce_key] = time.time()
    return True


# ---------------------------------------------------------------------------
# IPC command signing (used by commands.py; verified by live_executor.py)
# ---------------------------------------------------------------------------

def sign_command(command: str, actor: str, issued_at: float) -> str:
    """Create a signature that authenticates a command file record.

    Returns "" when SUPERVISOR_HMAC_SECRET is not configured (dev mode).
    """
    secret = os.environ.get("SUPERVISOR_HMAC_SECRET", "")
    if not secret:
        return ""
    msg = f"{command}:{actor}:{issued_at:.3f}".encode()
    return _hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()


def sign_approval_marker(approval_id: str, actor: str,
                          command: str, issued_at: float) -> str:
    """One-time marker binding a confirmed approval to a specific command dispatch.

    Returns "" when SUPERVISOR_HMAC_SECRET is not configured.
    """
    secret = os.environ.get("SUPERVISOR_HMAC_SECRET", "")
    if not secret:
        return ""
    msg = f"approve:{approval_id}:{actor}:{command}:{issued_at:.3f}".encode()
    return _hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def require_auth(f):
    """Verify Bearer JWT. For POST/PUT/PATCH/DELETE also verify HMAC signature."""
    @wraps(f)
    def _inner(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return jsonify({"error": "missing Bearer token"}), 401
        token = auth[7:].strip()
        try:
            g.jwt_payload = decode_jwt(token)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 401

        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            ok = verify_hmac_request(
                body=request.get_data(),
                sig=request.headers.get("X-Signature", ""),
                ts_str=request.headers.get("X-Timestamp", ""),
                nonce=request.headers.get("X-Nonce", ""),
                method=request.method,
                path=request.path,
                query=request.query_string.decode("utf-8", errors="replace"),
            )
            if not ok:
                return jsonify({"error": "invalid HMAC signature"}), 401

        return f(*args, **kwargs)
    return _inner
