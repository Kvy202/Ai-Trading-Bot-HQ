"""
features/collectors/base.py

Abstract base class for all Tier 2 shadow-mode data collectors.

Each concrete collector:
  - implements collect(exchange, symbols, store) -> int
  - is gated by TIER2_<NAME> env var (1=enabled by default)
  - writes an atomic heartbeat JSON to logs/tier2_<name>_heartbeat.json
    after every run_once() call, whether it succeeded or failed
  - NEVER writes to live_signals.csv or influences any trading decision
"""
from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

_LOG = logging.getLogger("tier2.collector")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOGS = _PROJECT_ROOT / "logs"


class BaseCollector(ABC):
    """Abstract base for Tier 2 shadow-mode data collectors."""

    def __init__(self, name: str):
        self.name = name
        self._hb_path = _LOGS / f"tier2_{name}_heartbeat.json"

    @property
    def enabled(self) -> bool:
        """Gated by TIER2_<NAME>=0 env var. Enabled by default."""
        val = os.getenv(f"TIER2_{self.name.upper()}", "1").strip().lower()
        return val in {"1", "true", "yes"}

    @abstractmethod
    def collect(self, exchange: Any, symbols: List[str], store: Any) -> int:
        """Fetch data for all symbols and append to store.

        Returns the number of rows written.  Must not raise — catch internal
        errors and return 0 if nothing was written.
        """

    def run_once(self, exchange: Any, symbols: List[str], store: Any) -> bool:
        """Run one collection cycle.

        Returns True if the collector ran (regardless of row count), False if
        it was disabled or an uncaught exception escaped collect().
        """
        if not self.enabled:
            return False
        try:
            count = self.collect(exchange, symbols, store)
            self._write_heartbeat(ok=True, rows=count)
            return True
        except Exception as exc:
            _LOG.warning("[%s] uncaught error: %s", self.name, exc)
            self._write_heartbeat(ok=False, error=str(exc))
            return False

    def _write_heartbeat(
        self,
        ok: bool,
        rows: int = 0,
        error: str = "",
    ) -> None:
        _LOGS.mkdir(parents=True, exist_ok=True)
        payload = {
            "collector": self.name,
            "ok": ok,
            "ts": datetime.now(timezone.utc).isoformat(),
            "rows_last_cycle": rows,
            "error": error,
        }
        tmp = self._hb_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self._hb_path)
