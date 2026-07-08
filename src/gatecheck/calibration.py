"""Gate-calibration harness — planted-truth / planted-null certification of statistical gates.

**Bring your gate function, get a certificate.** This is the "tests for the tests" module:
a research program that certifies its *claims* against nulls but never certifies its *gates*
against worlds where the truth is KNOWN has no idea whether its instruments can detect what
they claim to adjudicate. This harness plants truth and plants null, runs your gate over
many seeded draws of each world, and records fire rates with binomial CIs — a
PASS/FAIL/CHARACTERIZED certificate per gate.

Lineage: extracted 2026-07-02 from the source research program
(``market_os/evaluation/calibration.py``, post-audit — upstream repair R5). Upstream, this
harness certified four production gates; the certification honestly FAILED one of them
(a content gate with zero power in its live regime, quarantined as a result) and
CHARACTERIZED the undisclosed detection floor of another (a DSR leg blind below ~1.0
planted gross Sharpe at its own sample geometry). That is the point: a gate you have not
calibrated is a gate you do not understand.

Worlds (numpy, seeded, fully offline)
-------------------------------------
* :func:`plant_vol_world` — an AR(1) log-vol daily world with a plantable forward-vol
  signal of (approximately) known partial correlation ``signal_strength`` given the
  controls ``[ret_t, TV_t]``. ``signal_strength=0`` -> the planted null (a persistent but
  content-free carrier).
* :func:`static_coupling_world` — AR(1) carrier, iid returns, an index level coupled
  contemporaneously to the carrier — ZERO anticipation anywhere. A structured planted null
  for any forward-decode gate: the carrier is heavily structured and coupled to a
  contemporaneous observable, yet carries NO forward information.
* :func:`content_world` — an implied-vol-like signal correlated with its own trailing-vol
  control, with genuine returns-coupled forward content (``has_content=True``:
  overshoot-and-correct on the underlier's own ``|ret|`` shocks) or without
  (``has_content=False``; ``geometry="slow"`` = a smooth persistence-only mixture,
  ``geometry="smoothed"`` = a pure smoothed function of the control itself — a
  false-pass geometry upstream).
* :func:`plant_edge_world` — persistent ±1 positions and returns with a planted
  conditional drift of known gross Sharpe; ``sharpe=0`` -> the planted null. For
  characterizing the detection floor of Sharpe/DSR-style gates
  (:mod:`gatecheck.deflation`).

Runner
------
:func:`certify` — run a boolean gate ``gate_fn(world) -> bool`` over ``n_seeds``
independent draws of each named world; returns per-world fire rates with add-one
(Laplace rule-of-succession) binomial CIs. The conventional world names are ``"truth"``
and ``"null"``; add as many structured nulls as you can think of.

Built-in example certificate
----------------------------
:func:`certificate_spearman_partial` — certifies :func:`gatecheck.rank.spearman_partial`
(at a |IC| >= 0.05, p < .05 materiality) on :func:`plant_vol_world` truth, its white null,
and the :func:`static_coupling_world` structured null. This is the PASS exhibit: upstream,
the same construction fired >= 80% on the planted signal and stayed <= 5% on both nulls.
Use it as the template for certifying your own gates.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .rank import residualize, spearman_partial

__all__ = [
    "trailing_vol",
    "forward_vol",
    "plant_vol_world",
    "static_coupling_world",
    "content_world",
    "plant_edge_world",
    "add_one_rate_ci",
    "WorldFireRate",
    "CertifyResult",
    "certify",
    "Certificate",
    "certificate_spearman_partial",
]


# ============================================================================
# small helpers
# ============================================================================

def _zscore(x: np.ndarray) -> np.ndarray:
    """Z-score over finite entries; non-finite entries map to 0.0 (kept full-length for np.roll surrogates)."""
    out = np.zeros(x.shape, dtype=float)
    m = np.isfinite(x)
    if int(np.sum(m)) < 2:
        return out
    v = x[m]
    sd = float(v.std())
    if sd == 0.0:
        return out
    out[m] = (v - v.mean()) / sd
    return out


def _ar1(n: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    """Unit-innovation AR(1) path, standardized to unit variance."""
    x = np.zeros(n)
    z = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + z[t]
    return _zscore(x)


def trailing_vol(ret: np.ndarray, w: int) -> np.ndarray:
    """Trailing realized vol over ``[t-w+1 .. t]`` (ends at t, EOD-knowable). ``w==1`` => ``|ret[t]|``."""
    ret = np.asarray(ret, dtype=float)
    n = ret.size
    vol = np.full(n, np.nan)
    if w <= 1:
        return np.abs(ret)
    for t in range(w - 1, n):
        window = ret[t - w + 1: t + 1]
        if np.all(np.isfinite(window)):
            vol[t] = float(np.sqrt(np.mean(window * window)))
    return vol


def forward_vol(ret: np.ndarray, w: int) -> np.ndarray:
    """STRICTLY-forward realized vol over ``[t+1 .. t+w]`` (uses only returns AFTER t). NaN past the end."""
    ret = np.asarray(ret, dtype=float)
    n = ret.size
    fv = np.full(n, np.nan)
    for t in range(0, n - w):
        window = ret[t + 1: t + w + 1]
        if window.size == w and np.all(np.isfinite(window)):
            fv[t] = float(np.sqrt(np.mean(window * window)))
    return fv


# ============================================================================
# world generators
# ============================================================================

def plant_vol_world(seed: int, signal_strength: float, *, n: int = 1500, w: int = 5,
                    vol_phi: float = 0.97, vol_sigma: float = 0.10,
                    carrier_phi: float = 0.90) -> dict:
    """An AR(1)-vol daily world with a plantable forward-vol signal of known partial correlation.

    Returns: an AR(1) log-vol process drives daily returns (vol clustering, the stylized
    equity-index world). The signal is ``signal_strength * e + sqrt(1-ss^2) * carrier``
    where ``e`` is the STANDARDIZED residual of forward-vol ``FV[t]`` (over ``[t+1..t+w]``)
    after the controls ``[ret_t, TV_t]`` and the carrier is a content-free persistent AR(1)
    (phi=``carrier_phi``, positioning-like persistence). The signal's levels partial
    correlation with FV given the controls is therefore ~``signal_strength`` by construction
    (rank-space Spearman partials read slightly attenuated). ``signal_strength=0`` -> the
    planted null.

    Keys: ``ret`` (ret[0]=nan), ``sig``, ``fv``, ``tv``, ``w``, ``signal_strength``, ``seed``.
    """
    ss = float(signal_strength)
    if not -1.0 < ss < 1.0:
        raise ValueError("signal_strength must be in (-1, 1)")
    rng = np.random.default_rng(int(seed))
    mu = math.log(0.01)
    h = np.full(n, mu)
    eps = rng.standard_normal(n)
    for t in range(1, n):
        h[t] = mu + vol_phi * (h[t - 1] - mu) + vol_sigma * eps[t]
    vol = np.exp(h)
    ret = np.full(n, np.nan)
    ret[1:] = vol[1:] * rng.standard_normal(n - 1)
    fv = forward_vol(ret, int(w))
    tv = trailing_vol(ret, int(w))

    # the plantable component: standardized residual of FORWARD vol after the controls
    m = np.isfinite(ret) & np.isfinite(fv) & np.isfinite(tv)
    e = np.zeros(n)
    if int(np.sum(m)) >= 30:
        e[m] = _zscore(residualize(fv[m], [ret[m], tv[m]]))
    carrier = _ar1(n, carrier_phi, rng)
    sig = ss * e + math.sqrt(max(1.0 - ss * ss, 0.0)) * carrier
    return {"ret": ret, "sig": sig, "fv": fv, "tv": tv, "w": int(w),
            "signal_strength": ss, "seed": int(seed)}


def static_coupling_world(seed: int, *, n: int = 2200, phi: float = 0.9, innov: float = 0.3,
                          k: float = 5.0) -> dict:
    """A structured planted null: contemporaneous coupling, ZERO anticipation anywhere.

    AR(1) carrier ``g`` (``phi``, ``innov``); iid daily returns (0.01 sd, INDEPENDENT of
    ``g``); an implied-vol-like index ``vix = 18 + random walk - k*g`` (contemporaneous
    coupling only). Nothing anywhere anticipates anything: ``g`` carries NO forward
    information about realized vol, so any forward-decode gate fed the ``g`` carrier must
    stay silent. Persistence + contemporaneous coupling is exactly the geometry that fools
    naive i.i.d. permutation nulls. Keys: ``g``, ``ret`` (ret[0]=nan), ``price``, ``vix``,
    ``seed``.
    """
    rng = np.random.default_rng(int(seed))
    g = np.zeros(n)
    for t in range(1, n):
        g[t] = phi * g[t - 1] + innov * rng.standard_normal()
    ret = np.full(n, np.nan)
    ret[1:] = 0.01 * rng.standard_normal(n - 1)
    price = 100.0 * np.cumprod(1.0 + np.where(np.isfinite(ret), ret, 0.0))
    vix = 18.0 + np.cumsum(rng.standard_normal(n)) - k * g
    return {"g": g, "ret": ret, "price": price, "vix": vix, "seed": int(seed)}


def content_world(seed: int, has_content: bool, control_corr: float = 0.85, *,
                  n: int = 1600, w: int = 21, geometry: str = "slow") -> dict:
    """An implied-vol-like signal correlated with its own trailing-vol control.

    Underlier: AR(1) log-vol returns; the control is ``trailing_vol(ret, w)``. For
    certifying gates that must distinguish *genuine forward content* from *mechanical
    persistence* in a signal that is heavily correlated with its own control (the
    attenuation regime — the hardest honest case).

    * ``has_content=True`` — GENUINE, returns-coupled forward content: the signal mixes the
      standardized control (weight ``control_corr``) with an overshoot-and-correct state
      driven by the underlier's own ``|ret|`` shocks (jumps on shock days, then decays).
      Its own forward change is genuinely predictable from the observed shock history
      beyond bounded persistence — a correctly-designed content gate should FIRE.
    * ``has_content=False, geometry="slow"`` — mechanical: the control mixed with an
      INDEPENDENT slow AR(1) (pure bounded persistence, no content) — the gate should stay
      SILENT.
    * ``has_content=False, geometry="smoothed"`` — a false-pass geometry: a pure 5-bar
      smoothing of the control itself (content-free by construction; conditioning on the
      control isolates a fast-reverting smoothing gap). Upstream, this geometry
      false-fired a production content gate — which is exactly what this world exists to
      catch.

    Keys: ``ret``, ``sig``, ``tv``, ``w``, ``has_content``, ``geometry``, ``seed``.
    """
    rng = np.random.default_rng(int(seed))
    mu = math.log(0.01)
    h = np.full(n, mu)
    for t in range(1, n):
        h[t] = mu + 0.98 * (h[t - 1] - mu) + 0.12 * rng.standard_normal()
    ret = np.full(n, np.nan)
    ret[1:] = np.exp(h[1:]) * rng.standard_normal(n - 1)
    tv = trailing_vol(ret, int(w))
    ztv = _zscore(tv)
    cc = float(control_corr)
    if not 0.0 <= cc < 1.0:
        raise ValueError("control_corr must be in [0, 1)")
    mix = math.sqrt(1.0 - cc * cc)

    if has_content:
        a = _zscore(np.abs(ret))                      # the underlier's own shock series
        o = np.zeros(n)
        for t in range(1, n):
            o[t] = 0.6 * o[t - 1] + a[t]              # overshoot on shocks, then correct (decay)
        sig = cc * ztv + mix * _zscore(o)
    elif geometry == "smoothed":
        k5 = 5                                        # false-pass geometry: smoothed control
        ma = np.convolve(ztv, np.ones(k5) / k5, mode="full")[: n]
        sig = _zscore(ma) + 0.05 * rng.standard_normal(n)
    else:
        sig = cc * ztv + mix * _ar1(n, 0.97, rng)     # independent slow persistence — content-free
    return {"ret": ret, "sig": sig, "tv": tv, "w": int(w),
            "has_content": bool(has_content), "geometry": str(geometry), "seed": int(seed)}


def plant_edge_world(seed: int, sharpe: float, *, n: int = 2774, sigma: float = 0.01,
                     flip_prob: float = 0.05) -> dict:
    """A planted-Sharpe edge world: persistent ±1 positions and returns with a planted conditional drift.

    ``ret = mu*pos + sigma*z`` with ``mu = sharpe*sigma/sqrt(252)``: the strategy pnl
    ``pos*ret`` has GROSS annualized Sharpe ~``sharpe`` by construction. Unconditional
    drift is ~0 (pos is symmetric), so buy&hold is ~flat. ``sharpe=0`` -> the planted null.

    Use it to characterize the detection floor of a Sharpe/DSR-style gate
    (:mod:`gatecheck.deflation`): sweep ``sharpe`` over a grid, record the fire rate at
    each level, and report the planted Sharpe at which the gate crosses 50% / 80% fire —
    a gate's detection floor is a property worth disclosing, not a pass/fail. Upstream,
    this world revealed a DSR leg structurally blind below ~1.0 planted gross Sharpe at
    its own sample geometry. Keys: ``pos``, ``ret``, ``sharpe``, ``seed``.
    """
    rng = np.random.default_rng(int(seed))
    flips = np.where(rng.random(n) < float(flip_prob), -1.0, 1.0)
    pos = np.cumprod(flips)
    mu = float(sharpe) * float(sigma) / math.sqrt(252.0)
    ret = mu * pos + float(sigma) * rng.standard_normal(n)
    return {"pos": pos, "ret": ret, "sharpe": float(sharpe), "seed": int(seed)}


# ============================================================================
# the certify runner
# ============================================================================

def add_one_rate_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Add-one (Laplace rule-of-succession) binomial rate + normal CI: sane at k=0 and k=n.

    Returns ``(rate, lo, hi)`` where ``rate = (k+1)/(n+2)`` and the interval is the normal
    approximation on the add-one estimate, clipped to [0, 1].
    """
    if n < 1:
        return float("nan"), float("nan"), float("nan")
    p = (k + 1) / (n + 2)
    half = z * math.sqrt(p * (1.0 - p) / (n + 2))
    return float(p), float(max(0.0, p - half)), float(min(1.0, p + half))


@dataclass
class WorldFireRate:
    """Fire count for one named world across seeds."""

    world: str
    n_seeds: int
    n_fired: int

    @property
    def rate(self) -> float:
        """Raw fire fraction k/n (used for the PASS-threshold comparisons)."""
        return self.n_fired / self.n_seeds if self.n_seeds else float("nan")

    @property
    def add_one(self) -> tuple[float, float, float]:
        """(rate, lo, hi) — add-one estimate with its 95% CI."""
        return add_one_rate_ci(self.n_fired, self.n_seeds)

    def render(self) -> str:
        r, lo, hi = self.add_one
        return (f"{self.world:<16} fired {self.n_fired:>3}/{self.n_seeds:<3} "
                f"raw {self.rate:5.1%}   add-one {r:5.1%} [{lo:5.1%}, {hi:5.1%}]")


@dataclass
class CertifyResult:
    """Per-world fire rates for one gate. ``truth`` / ``null`` are the conventional world names."""

    rates: dict[str, WorldFireRate] = field(default_factory=dict)

    def _rate(self, name: str) -> float:
        r = self.rates.get(name)
        return r.rate if r is not None else float("nan")

    @property
    def fire_rate_on_truth(self) -> float:
        return self._rate("truth")

    @property
    def fire_rate_on_null(self) -> float:
        return self._rate("null")


def certify(gate_fn, worlds: dict, n_seeds: int = 20, *, base_seed: int = 0) -> CertifyResult:
    """Run ``gate_fn(world) -> bool`` over ``n_seeds`` draws of each named world factory.

    ``worlds`` maps a name (conventionally including ``"truth"`` and ``"null"``) to a factory
    ``factory(seed) -> world-dict``. Seeds are ``base_seed .. base_seed+n_seeds-1`` per world
    (deterministic). Returns fire rates with add-one binomial CIs.

    This is the whole public contract: bring any boolean gate over any world dict, get a
    fire-rate table back. A well-calibrated detector fires on ``truth`` (power) and stays
    silent on every ``null`` you can construct (size) — including *structured* nulls that
    share the carrier's nuisance geometry (persistence, coupling, smoothing), not just
    white noise.
    """
    result = CertifyResult()
    for name, factory in worlds.items():
        fired = 0
        for i in range(int(n_seeds)):
            if bool(gate_fn(factory(base_seed + i))):
                fired += 1
        result.rates[name] = WorldFireRate(name, int(n_seeds), fired)
    return result


@dataclass
class Certificate:
    """One gate's calibration certificate: what was planted, fire rates, and the honest verdict."""

    target: str
    planted: str
    verdict: str                 # "PASS" | "FAIL" | "CHARACTERIZED"
    result: CertifyResult | None
    details: dict = field(default_factory=dict)
    runtime_s: float = 0.0

    def summary(self) -> str:
        lines = ["=" * 96,
                 f"CALIBRATION CERTIFICATE — {self.target}   [{self.verdict}]",
                 "=" * 96,
                 f"planted : {self.planted}"]
        if self.result is not None:
            for name in self.result.rates:
                lines.append("  " + self.result.rates[name].render())
        for key, val in self.details.items():
            lines.append(f"{key} : {val}")
        lines.append(f"runtime : {self.runtime_s:.1f}s")
        lines.append("=" * 96)
        return "\n".join(lines)


# ============================================================================
# built-in example certificate: gatecheck.rank.spearman_partial (the PASS exhibit)
# ============================================================================

#: Materiality floor for the example certificate: a partial |IC| below this is noise-scale
#: regardless of its p-value.
MATERIAL_IC = 0.05


def _spearman_partial_gate_fires(sig, fv, ret, tv, *, n_perm: int, seed: int) -> bool:
    """The example gate: material (|partial IC| >= MATERIAL_IC) AND significant (p < .05)."""
    ic, p = spearman_partial(sig, fv, [ret, tv], n_perm=int(n_perm), seed=int(seed))
    return bool(np.isfinite(ic) and abs(ic) >= MATERIAL_IC and p < 0.05)


def certificate_spearman_partial(*, n_seeds: int = 25, n: int = 1500, w: int = 5,
                                 signal_strength: float = 0.25, n_perm: int = 1000,
                                 base_seed: int = 0) -> Certificate:
    """Certify :func:`gatecheck.rank.spearman_partial` on planted truth, a white planted
    null, and the static-coupling structured null — the built-in PASS exhibit.

    The gate under certification: "the carrier reads forward vol beyond the controls" at
    the |IC| >= 0.05, p < .05 materiality. The planted levels-partial of 0.25 realizes a
    rank-space Spearman partial of ~0.20. PASS requires: fire rate >= 80% on the planted
    signal AND <= 5% on BOTH nulls (raw fractions; add-one CIs recorded).

    This function doubles as the template for certifying your own gate: wrap your gate as
    ``gate(world) -> bool``, name your worlds, call :func:`certify`, and decide a pass rule
    BEFORE you look at the numbers.
    """
    t0 = time.time()

    def gate(world: dict) -> bool:
        if "g" in world:          # structured null: the coupled carrier vs its own world's forward vol
            fv = forward_vol(world["ret"], w)
            tv = trailing_vol(world["ret"], w)
            return _spearman_partial_gate_fires(world["g"], fv, world["ret"], tv,
                                                n_perm=n_perm, seed=world["seed"])
        return _spearman_partial_gate_fires(world["sig"], world["fv"], world["ret"], world["tv"],
                                            n_perm=n_perm, seed=world["seed"])

    worlds = {
        "truth": lambda s: plant_vol_world(s, signal_strength, n=n, w=w),
        "null": lambda s: plant_vol_world(s + 100_000, 0.0, n=n, w=w),
        "null_static": lambda s: static_coupling_world(s + 200_000),
    }
    res = certify(gate, worlds, n_seeds=n_seeds, base_seed=base_seed)
    ok = (res.fire_rate_on_truth >= 0.80 and res.fire_rate_on_null <= 0.05
          and res.rates["null_static"].rate <= 0.05)
    return Certificate(
        target="rank.spearman_partial (|IC|>=0.05 & p<.05 materiality)",
        planted=(f"AR(1)-vol world, planted forward partial corr ~{signal_strength:+.2f} "
                 f"(truth) / 0 (null) / static-coupling carrier (structured null); n={n}, w={w}"),
        verdict="PASS" if ok else "FAIL",
        result=res,
        details={"n_perm": n_perm,
                 "pass_rule": "truth >= 80% fire AND both nulls <= 5% fire (raw fractions)"},
        runtime_s=time.time() - t0)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Run the built-in example calibration certificate "
                    "(rank.spearman_partial on plant_vol_world)")
    ap.add_argument("--fast", action="store_true", help="reduced, test-scale configuration")
    args = ap.parse_args(argv)
    if args.fast:
        cert = certificate_spearman_partial(n_seeds=6, n=1200, n_perm=200)
    else:
        cert = certificate_spearman_partial()
    print(cert.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
