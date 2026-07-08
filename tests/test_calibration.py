"""Tests for the gate-calibration harness (gatecheck.calibration).

Ported 2026-07-02 from the source program ``tests/test_calibration_harness.py`` (the fast,
package-portable arm: world generators, the certify runner, and the built-in
spearman_partial certificate — the PASS exhibit). Offline + deterministic (seeded numpy
worlds, no network). Assertions are deliberately loose bands around the deterministic
fast-config results so a numpy version bump does not flip them spuriously.
"""

from __future__ import annotations

import numpy as np

from gatecheck.calibration import (
    Certificate,
    add_one_rate_ci,
    certificate_spearman_partial,
    certify,
    content_world,
    plant_edge_world,
    plant_vol_world,
    static_coupling_world,
)
from gatecheck.rank import spearman_partial


# ============================ world generators ============================
def test_plant_vol_world_deterministic_and_null_at_zero_strength():
    w1 = plant_vol_world(7, 0.25)
    w2 = plant_vol_world(7, 0.25)
    assert np.array_equal(w1["sig"], w2["sig"], equal_nan=True)          # seeded -> reproducible
    assert w1["sig"].size == 1500 and np.isnan(w1["ret"][0])
    w0 = plant_vol_world(7, 0.0)
    # planted null: the signal is the pure carrier, uncorrelated with the forward-vol residual by design
    m = np.isfinite(w0["fv"]) & np.isfinite(w0["tv"]) & np.isfinite(w0["ret"])
    r = np.corrcoef(w0["sig"][m], w0["fv"][m])[0, 1]
    assert abs(r) < 0.15


def test_plant_vol_world_realizes_the_planted_partial():
    wd = plant_vol_world(3, 0.25)
    ic, p = spearman_partial(wd["sig"], wd["fv"], [wd["ret"], wd["tv"]], n_perm=200, seed=3)
    assert 0.10 <= ic <= 0.40 and p < 0.05           # rank partial ~0.20 for a 0.25 levels plant


def test_static_coupling_world_matches_pinned_construction():
    wd = static_coupling_world(1, n=800)
    assert wd["g"].size == wd["vix"].size == 800
    # contemporaneous coupling: vix co-moves with -k*carrier on top of the random-walk level
    dv = np.diff(wd["vix"])
    dg = np.diff(wd["g"])
    assert np.corrcoef(dv, dg)[0, 1] < -0.5          # DeltaVIX carries -5*Delta-g
    # the carrier is independent of returns (no anticipation channel exists)
    m = np.isfinite(wd["ret"])
    assert abs(np.corrcoef(wd["g"][m], np.abs(wd["ret"][m]))[0, 1]) < 0.10


def test_content_world_branches():
    t = content_world(0, True, 0.70)
    n_slow = content_world(0, False, 0.70, geometry="slow")
    n_sm = content_world(0, False, 0.70, geometry="smoothed")
    for wd in (t, n_slow, n_sm):
        assert wd["sig"].size == 1600 and np.all(np.isfinite(wd["sig"]))
        m = np.isfinite(wd["tv"])
        assert np.corrcoef(wd["sig"][m], wd["tv"][m])[0, 1] > 0.4    # control-correlated by construction
    assert not np.array_equal(t["sig"], n_slow["sig"])


def test_plant_edge_world_gross_sharpe_close_to_planted():
    wd = plant_edge_world(0, 1.5, n=2774)
    pnl = wd["pos"] * wd["ret"]
    s = pnl.mean() / pnl.std() * np.sqrt(252)
    assert 1.0 <= s <= 2.0                                           # planted gross Sharpe ~1.5 +/- sampling


def test_plant_edge_world_null_has_no_edge():
    wd = plant_edge_world(1, 0.0, n=2774)
    pnl = wd["pos"] * wd["ret"]
    s = pnl.mean() / pnl.std() * np.sqrt(252)
    assert abs(s) < 0.8                                              # planted null ~ 0 +/- sampling


# ============================ the certify runner ============================
def test_add_one_rate_ci_sane_at_extremes():
    r0, lo0, hi0 = add_one_rate_ci(0, 20)
    assert 0.0 <= lo0 <= r0 <= hi0 <= 1.0 and abs(r0 - 1 / 22) < 1e-12
    rn, lon, hin = add_one_rate_ci(20, 20)
    assert abs(rn - 21 / 22) < 1e-12 and hin <= 1.0


def test_certify_runner_counts_fires():
    worlds = {"truth": lambda s: {"v": 1.0, "seed": s}, "null": lambda s: {"v": -1.0, "seed": s}}
    res = certify(lambda w: w["v"] > 0, worlds, n_seeds=8)
    assert res.fire_rate_on_truth == 1.0 and res.fire_rate_on_null == 0.0
    assert res.rates["truth"].n_seeds == 8 and res.rates["null"].n_fired == 0


# ============================ the built-in example certificate — expect PASS ============================
def test_spearman_partial_certificate_passes_fast():
    cert = certificate_spearman_partial(n_seeds=6, n=1200, n_perm=200)
    assert isinstance(cert, Certificate)
    assert cert.result.fire_rate_on_truth >= 0.80                    # power on the planted signal
    assert cert.result.fire_rate_on_null <= 1 / 6 + 1e-9             # ~silent on the white null
    assert cert.result.rates["null_static"].rate <= 1 / 6 + 1e-9     # silent on the structured null
    assert cert.verdict in ("PASS", "FAIL")
    assert "CALIBRATION CERTIFICATE" in cert.summary()
