"""
tools/check_imports.py

Regression guard against Python namespace collisions between config.py (the
legacy exchange-config module) and config/ (the runtime config directory).

Run this after any change that touches config/, runtime/, or adds new packages
to the project root.  If it exits 0, the import graph is clean.

Usage:
    python tools/check_imports.py
"""
from __future__ import annotations

import importlib
import os
import sys
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

checks_passed = 0
checks_failed = 0


def check(label: str, fn) -> None:
    global checks_passed, checks_failed
    try:
        result = fn()
        print(f"  [{PASS}] {label}", f"-> {result}" if result is not None else "")
        checks_passed += 1
    except Exception as exc:
        print(f"  [{FAIL}] {label}")
        traceback.print_exc(limit=3)
        checks_failed += 1


print("\nImport regression check")
print("=" * 60)

# --- 1. config.py (legacy module) must be the one that resolves ---
def _config_is_module():
    import config as cfg
    eid = cfg.EXCHANGE_ID
    assert eid, "EXCHANGE_ID is empty"
    return f"EXCHANGE_ID={eid!r}"
check("config.EXCHANGE_ID (must come from config.py, not config/)", _config_is_module)

# --- 2. config has API creds attributes (even if empty) ---
def _config_has_api():
    import config as cfg
    assert hasattr(cfg, "API_KEY"), "missing API_KEY"
    assert hasattr(cfg, "API_SECRET"), "missing API_SECRET"
    return "API_KEY and API_SECRET present"
check("config.API_KEY / API_SECRET attributes present", _config_has_api)

# --- 3. runtime.loader works and loads run.json ---
def _runtime_loader():
    from runtime.loader import apply_run_config
    loaded = apply_run_config()
    assert len(loaded) > 0, "no keys loaded from config/run.json"
    assert "PRED_THRESHOLD" in loaded, "PRED_THRESHOLD missing from run.json"
    return f"{len(loaded)} keys loaded"
check("runtime.loader.apply_run_config() reads config/run.json", _runtime_loader)

# --- 4. data.py has load_prices_and_features (source check - avoids ccxt import hang) ---
def _data_importable():
    src = (_ROOT / "data.py").read_text(encoding="utf-8", errors="ignore")
    assert "def load_prices_and_features" in src, \
        "load_prices_and_features not found in data.py"
    return "load_prices_and_features present (source check)"
check("data.py has load_prices_and_features", _data_importable)

# --- 5. exchange.py has live_client (source check - avoids ccxt import hang) ---
def _exchange_importable():
    src = (_ROOT / "exchange.py").read_text(encoding="utf-8", errors="ignore")
    assert "live_client" in src, "live_client not found in exchange.py"
    return "live_client present (source check)"
check("exchange.py has live_client", _exchange_importable)

# --- 6. feature_store importable ---
def _feature_store():
    from tier2.feature_store import FeatureStore
    return "FeatureStore importable"
check("tier2.feature_store.FeatureStore importable", _feature_store)

# --- 7. config/ directory is NOT a Python package (no __init__.py) ---
def _no_config_init():
    init = _ROOT / "config" / "__init__.py"
    assert not init.exists(), \
        f"config/__init__.py exists — delete it to avoid shadowing config.py"
    return "config/__init__.py absent (correct)"
check("config/__init__.py must NOT exist", _no_config_init)

def _no_features_init():
    init = _ROOT / "features" / "__init__.py"
    assert not init.exists(), \
        "features/__init__.py exists — delete it to avoid shadowing features.py"
    return "features/__init__.py absent (correct)"
check("features/__init__.py must NOT exist", _no_features_init)

# --- 8. No stale config.loader references in Python files ---
def _no_config_loader_imports():
    bad = []
    self_path = Path(__file__).resolve()
    for py in _ROOT.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        if py.resolve() == self_path:
            continue  # skip this script (it contains the strings as literals)
        try:
            src = py.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "from config.loader" in src or "import config.loader" in src:
            bad.append(str(py.relative_to(_ROOT)))
    assert not bad, f"stale config.loader imports in: {bad}"
    return "none found"
check("No 'from config.loader' imports remain", _no_config_loader_imports)

# --- Summary ---
print()
print("=" * 60)
total = checks_passed + checks_failed
if checks_failed == 0:
    print(f"  All {total} checks passed.")
else:
    print(f"  {checks_passed}/{total} passed, {checks_failed} FAILED.")
print()

sys.exit(0 if checks_failed == 0 else 1)
