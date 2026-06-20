#!/usr/bin/env python3
r"""test_gate_fix.py - offline tests for the agree-gate fix in predict_ensemble.

Verifies (no model artifacts needed - predict_next is monkeypatched):
  1. Agree-gate suppression returns NEUTRAL p=0.5 (centered 0.0 = FLAT),
     not 0.0 (which the writer centered to -0.5, a fake allowed SHORT).
  2. Zero-weight models (DL_MODEL_WEIGHTS e.g. tcn:0) do NOT vote in the
     agree-gate, but still appear in per_model for diagnostics.
  3. Zero-weight models contribute nothing to the blended probability.
  4. Normal agreement is unaffected.

Run:  python tools/test_gate_fix.py   ->  must end RESULT: PASS
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import ml_dl.dl_ensemble as ens

FAILS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


class FakeScaler:
    n_features_in_ = 26


def make_models(p_by_kind):
    return {k: {"scaler": FakeScaler(), "model": k} for k in p_by_kind}


def run(p_by_kind, weights, min_agree, symbol):
    """Call predict_ensemble with predict_next mocked to fixed p_long values."""
    os.environ["DL_MIN_AGREE"] = str(min_agree)
    # Neutralize per-model calibration: a loaded .env (DL_BIAS_*/DL_TEMP_*)
    # would otherwise shift the mocked probabilities and break exact asserts.
    for k in p_by_kind:
        os.environ[f"DL_BIAS_{k.upper()}"] = "0.0"
        os.environ[f"DL_TEMP_{k.upper()}"] = "1.0"
    # fresh degeneracy state so tests don't bleed into each other
    ens._DEGEN_COUNTS.clear()
    ens._DEGEN_HISTORY.clear()
    orig = ens.predict_next
    ens.predict_next = lambda xw, scaler, model, device: (0.0, 0.0, p_by_kind[model])
    try:
        x = np.zeros((8, 26), dtype=np.float32)
        return ens.predict_ensemble(x, make_models(p_by_kind), "cpu",
                                    weights=weights, symbol=symbol)
    finally:
        ens.predict_next = orig
        os.environ.pop("DL_MIN_AGREE", None)


def main():
    print("=" * 70)
    print("  AGREE-GATE FIX TESTS")
    print("=" * 70)

    # 1) Disagreement suppresses to NEUTRAL 0.5, never 0.0.
    pm, agg = run({"adv": 0.6, "lstm": 0.6, "tcn": 0.4, "tx": 0.4},
                  weights={"adv": 1, "lstm": 1, "tcn": 1, "tx": 1},
                  min_agree=3, symbol="T1")
    check("suppressed blend is neutral 0.5", abs(agg[2] - 0.5) < 1e-9,
          f"p={agg[2]}")
    check("suppressed centered value is FLAT", abs((agg[2] - 0.5) - 0.0) < 1e-9)

    # 2) Zero-weight model cannot tip the gate.
    #    voters = {adv, lstm} -> 1 bull vs 1 bear -> suppressed,
    #    even though zero-weight tcn is bullish (would have made 2 bulls).
    pm, agg = run({"adv": 0.6, "lstm": 0.4, "tcn": 0.55},
                  weights={"adv": 1, "lstm": 1, "tcn": 0},
                  min_agree=2, symbol="T2")
    check("zero-weight model does not vote (suppressed)",
          abs(agg[2] - 0.5) < 1e-9, f"p={agg[2]}")
    check("zero-weight model still logged in per_model", "tcn" in pm)

    #    same inputs with tcn weighted -> 2 bulls -> passes the gate
    pm, agg = run({"adv": 0.6, "lstm": 0.4, "tcn": 0.55},
                  weights={"adv": 1, "lstm": 1, "tcn": 1},
                  min_agree=2, symbol="T3")
    check("weighted model votes (gate passes)", abs(agg[2] - 0.5) > 1e-3,
          f"p={agg[2]:.4f}")

    # 3) Zero-weight model contributes nothing to the blend value.
    pm, agg = run({"adv": 0.60, "lstm": 0.62, "tcn": 0.90},
                  weights={"adv": 1, "lstm": 1, "tcn": 0},
                  min_agree=2, symbol="T4")
    expected = (0.60 + 0.62) / 2
    check("zero-weight excluded from blend", abs(agg[2] - expected) < 1e-6,
          f"p={agg[2]:.4f} expected={expected:.4f}")

    # 4) Normal agreement unaffected.
    pm, agg = run({"adv": 0.6, "lstm": 0.7},
                  weights={"adv": 1, "lstm": 1}, min_agree=2, symbol="T5")
    check("agreement passes with blended p", abs(agg[2] - 0.65) < 1e-6,
          f"p={agg[2]:.4f}")

    print("-" * 70)
    print(f"RESULT: {'FAIL (' + ', '.join(FAILS) + ')' if FAILS else 'PASS'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
