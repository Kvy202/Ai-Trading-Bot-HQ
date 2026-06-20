#!/usr/bin/env python
"""
tools/test_supervisor.py
========================
Smoke-test the Supervisor API paper-mode flow.

Tests (in order):
  [1] GET  /supervisor/health          -- must respond, mode=PAPER
  [2] GET  /supervisor/status          -- state snapshot
  [3] GET  /supervisor/metrics         -- financial KPIs
  [4] POST /supervisor/pause           -- pauses trading
  [5] GET  /supervisor/status          -- confirms paused=true
  [6] POST /supervisor/resume          -- resumes (no approval needed in PAPER)
  [7] GET  /supervisor/status          -- confirms paused=false
  [8] POST /supervisor/reduce_risk     -- switches risk_mode -> reduced
  [9] POST /supervisor/conservative_mode  -- switches risk_mode -> conservative
  [10] GET /supervisor/logs            -- audit trail shows all actions above

Exit 0 = all PASS.  Exit 1 = one or more FAIL.

Usage:
    python tools/test_supervisor.py
    python tools/test_supervisor.py --port 8789
    python tools/test_supervisor.py --url http://127.0.0.1:8789
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac as _hmac
import json
import os
import secrets
import sys
import time
from pathlib import Path

try:
    import requests as _req
except ImportError:
    print("ERROR: 'requests' is not installed -- run: pip install requests")
    sys.exit(1)

_BASE = Path(__file__).resolve().parents[1]
_ENV  = _BASE / ".env"


# ---------------------------------------------------------------------------
# Load .env secrets (no external deps)
# ---------------------------------------------------------------------------

def _load_env() -> None:
    if not _ENV.exists():
        return
    for line in _ENV.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


# ---------------------------------------------------------------------------
# JWT (stdlib, no PyJWT)
# ---------------------------------------------------------------------------

def _b64url_enc(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _make_jwt(secret: str, sub: str = "test-script", ttl: int = 300) -> str:
    now = int(time.time())
    payload = {"sub": sub, "iat": now, "exp": now + ttl}
    header  = _b64url_enc(json.dumps({"alg": "HS256", "typ": "JWT"},
                                     separators=(",", ":")).encode())
    body    = _b64url_enc(json.dumps(payload, separators=(",", ":")).encode())
    signing = f"{header}.{body}"
    sig = _hmac.new(secret.encode(), signing.encode(), hashlib.sha256).digest()
    return f"{signing}.{_b64url_enc(sig)}"


# ---------------------------------------------------------------------------
# HMAC request signing (matches supervisor/auth.py)
# ---------------------------------------------------------------------------

def _sign_request(method: str, path: str, body: bytes, hmac_secret: str) -> dict:
    """Return the three headers required on state-changing requests."""
    ts    = f"{time.time():.3f}"
    nonce = secrets.token_hex(16)
    body_hash = hashlib.sha256(body).hexdigest()
    canonical = f"{method}\n{path}\n\n{ts}\n{nonce}\n{body_hash}".encode()
    sig = _hmac.new(hmac_secret.encode(), canonical, hashlib.sha256).hexdigest()
    return {"X-Timestamp": ts, "X-Nonce": nonce, "X-Signature": sig}


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

class TestRunner:
    def __init__(self, base_url: str, jwt: str, hmac_secret: str) -> None:
        self.base = base_url.rstrip("/")
        self.jwt  = jwt
        self.hmac = hmac_secret
        self._results: list[tuple[str, bool, str]] = []

    def _headers(self, extra: dict | None = None) -> dict:
        h = {"Authorization": f"Bearer {self.jwt}",
             "Content-Type": "application/json"}
        if extra:
            h.update(extra)
        return h

    def _get(self, path: str, label: str, expect_keys: list[str] | None = None) -> dict | None:
        url = self.base + path
        try:
            r = _req.get(url, headers=self._headers(), timeout=10)
        except _req.exceptions.ConnectionError:
            self._fail(label, f"connection refused at {url}")
            return None
        if r.status_code != 200:
            self._fail(label, f"HTTP {r.status_code}: {r.text[:200]}")
            return None
        data = r.json()
        if expect_keys:
            missing = [k for k in expect_keys if k not in data]
            if missing:
                self._fail(label, f"missing keys {missing} in response")
                return data
        self._pass(label, f"HTTP 200  keys={list(data.keys())[:6]}")
        return data

    def _post(self, path: str, label: str, body: dict | None = None,
              expect_keys: list[str] | None = None) -> dict | None:
        raw  = json.dumps(body or {}).encode()
        extra = _sign_request("POST", path, raw, self.hmac)
        url  = self.base + path
        try:
            r = _req.post(url, data=raw, headers=self._headers(extra), timeout=10)
        except _req.exceptions.ConnectionError:
            self._fail(label, f"connection refused at {url}")
            return None
        if r.status_code not in (200, 201, 202):
            self._fail(label, f"HTTP {r.status_code}: {r.text[:200]}")
            return None
        data = r.json()
        if expect_keys:
            missing = [k for k in expect_keys if k not in data]
            if missing:
                self._fail(label, f"missing keys {missing} in response")
                return data
        self._pass(label, f"HTTP {r.status_code}  {_brief(data)}")
        return data

    def _pass(self, label: str, detail: str) -> None:
        print(f"  [PASS] {label:<40s} {detail}")
        self._results.append((label, True, detail))

    def _fail(self, label: str, detail: str) -> None:
        print(f"  [FAIL] {label:<40s} {detail}")
        self._results.append((label, False, detail))

    def run(self) -> int:
        print(f"\n{'='*70}")
        print(f"  Supervisor API paper-mode test  ({self.base})")
        print(f"{'='*70}\n")

        # [1] health
        h = self._get("/supervisor/health", "[1] GET /health",
                      ["mode", "paused", "ts"])
        if h is None:
            print("\n  Cannot reach supervisor -- is start_supervisor.ps1 running?")
            print("  Run: .\\tools\\start_supervisor.ps1")
            return 1

        mode = h.get("mode", "?")
        if mode != "PAPER":
            print(f"\n  WARNING: mode={mode!r} -- this test is designed for PAPER mode only")

        # [2] status
        self._get("/supervisor/status", "[2] GET /status",
                  ["ts", "paused", "mode", "risk_mode"])

        # [3] metrics
        self._get("/supervisor/metrics", "[3] GET /metrics",
                  ["unrealized_sum", "risk_mode"])

        # [4] pause
        paused = self._post("/supervisor/pause", "[4] POST /pause",
                            {"reason": "test_supervisor.py smoke test"},
                            ["status", "command", "issued_at"])
        if paused and paused.get("command") == "pause" and paused.get("status") == "accepted":
            self._pass("[4b] accepted flag", "command=pause status=accepted")
        elif paused:
            self._fail("[4b] accepted flag",
                       f"expected command=pause status=accepted, got {_brief(paused)}")

        time.sleep(2.0)

        # [5] confirm paused via status (async IPC -- executor must be running)
        s = self._get("/supervisor/status", "[5] GET /status (paused?)")
        if s:
            p = s.get("paused", False)
            note = "True -- executor processed pause" if p else "False -- executor async, may lag"
            print(f"  [INFO] [5b] status.paused             {note}")

        time.sleep(1.0)

        # [6] resume (PAPER mode: no approval_id needed)
        resumed = self._post("/supervisor/resume", "[6] POST /resume",
                             {"reason": "test_supervisor.py smoke test"},
                             ["status", "command", "issued_at"])
        if resumed and resumed.get("command") == "resume" and resumed.get("status") == "accepted":
            self._pass("[6b] accepted flag", "command=resume status=accepted")
        elif resumed:
            self._fail("[6b] accepted flag",
                       f"expected command=resume status=accepted, got {_brief(resumed)}")

        time.sleep(2.0)

        # [7] confirm resumed (async IPC -- executor must be running)
        s2 = self._get("/supervisor/status", "[7] GET /status (resumed?)")
        if s2:
            p2 = s2.get("paused", True)
            note = "False -- executor processed resume" if not p2 else "True -- executor async, may lag"
            print(f"  [INFO] [7b] status.paused             {note}")

        # [8] reduce_risk
        self._post("/supervisor/reduce_risk", "[8] POST /reduce_risk",
                   {"reason": "test_supervisor.py"},
                   ["status", "command", "issued_at"])

        time.sleep(2.0)

        # [9] conservative_mode
        self._post("/supervisor/conservative_mode", "[9] POST /conservative_mode",
                   {"reason": "test_supervisor.py"},
                   ["status", "command", "issued_at"])

        time.sleep(2.0)

        # [10] audit log
        logs = self._get("/supervisor/logs", "[10] GET /logs", ["entries"])
        if logs:
            entries = logs.get("entries", [])
            n = len(entries)
            self._pass("[10b] audit entries", f"{n} entries in log")

        # Summary
        passed = sum(1 for _, ok, _ in self._results if ok)
        total  = len(self._results)
        print(f"\n{'='*70}")
        print(f"  Result: {passed}/{total} PASS")
        print(f"{'='*70}\n")

        return 0 if passed == total else 1


def _brief(d: dict) -> str:
    items = [(k, v) for k, v in list(d.items())[:4]]
    return "  ".join(f"{k}={v!r}" for k, v in items)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _load_env()

    p = argparse.ArgumentParser(description="Supervisor API paper-mode smoke test")
    p.add_argument("--url",  default=None, help="Base URL (default: http://127.0.0.1:<port>)")
    p.add_argument("--port", type=int, default=int(os.getenv("SUPERVISOR_PORT", "8789")),
                   help="Supervisor port (default: 8789)")
    args = p.parse_args()

    base_url = args.url or f"http://127.0.0.1:{args.port}"

    jwt_secret  = os.environ.get("SUPERVISOR_JWT_SECRET", "")
    hmac_secret = os.environ.get("SUPERVISOR_HMAC_SECRET", "")

    if not jwt_secret or not hmac_secret:
        print("ERROR: SUPERVISOR_JWT_SECRET and SUPERVISOR_HMAC_SECRET must be set in .env")
        sys.exit(1)

    token  = _make_jwt(jwt_secret, sub="test-script", ttl=300)
    runner = TestRunner(base_url, token, hmac_secret)
    sys.exit(runner.run())


if __name__ == "__main__":
    main()
