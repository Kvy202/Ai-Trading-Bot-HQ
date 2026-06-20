"""
runtime/loader.py
=================
Reads config/run.json and injects all keys into os.environ as defaults.

Priority chain (highest to lowest):
  1. Existing shell env vars (already in os.environ when this runs)
  2. .env  (load_dotenv with override=True, called AFTER this)
  3. config/run.json  (this module — sets defaults only)

Usage:
    from runtime.loader import apply_run_config
    apply_run_config()               # reads <project_root>/config/run.json
    apply_run_config(base_dir=path)  # explicit project root

Call this BEFORE load_dotenv() so that .env can override run.json values.

Migration path:
  - During transition: params exist in both run.json and .env; .env wins.
  - To activate run.json control for a param: remove it from .env.

NOTE: This module was previously at config/loader.py.  It was moved here to
avoid a namespace collision: creating config/__init__.py shadows config.py
(the legacy exchange-config module), breaking data.py and exchange.py which
do `from config import EXCHANGE_ID`.  The runtime/ package avoids that conflict.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def apply_run_config(base_dir: Path | str | None = None) -> dict[str, str]:
    """
    Load config/run.json and apply non-secret keys to os.environ via setdefault.
    Returns the dict of key→value pairs that were loaded (regardless of whether
    they were already set).
    """
    root = Path(base_dir) if base_dir else _DEFAULT_ROOT
    cfg_path = root / "config" / "run.json"

    if not cfg_path.exists():
        return {}

    try:
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[runtime.loader] WARNING: could not read {cfg_path}: {exc}")
        return {}

    loaded: dict[str, str] = {}

    for section, entries in raw.items():
        if not isinstance(entries, dict):
            continue
        for key, val in entries.items():
            if not isinstance(key, str) or not key:
                continue
            str_val = str(val) if not isinstance(val, str) else val
            loaded[key] = str_val
            os.environ.setdefault(key, str_val)

    return loaded
