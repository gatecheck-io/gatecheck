"""Recovered posted-claim schema and July benchmark orchestration."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from bench_engine import EdgeSpec, INDICATORS, RULES, run_edge_test
POSTED_N_TRIALS = 10

@dataclass
class PostedClaim:
    """A tradable-edge claim mined from a posted thread / paper, mapped onto the EdgeSpec vocabulary.

    The mining agent fills ``text`` + ``source`` from the post and proposes the formal mapping
    (``indicator``/``window``/``rule``/``threshold``/...). Unmappable claims are still recorded (with the
    fields left at defaults and ``benchable`` resolving False) so nothing is silently dropped.
    """

    claim_id: str
    text: str
    source: str
    asset: str = "SPY"
    indicator: str = "rsi"
    window: int = 2
    rule: str = "long_below"
    threshold: float = 10.0
    horizon: int = 1
    direction: int = 1
    claimed_sharpe: float | None = None
    n_trials: int = POSTED_N_TRIALS

    def to_spec(self) -> EdgeSpec:
        spec = EdgeSpec(
            name=self.claim_id, indicator=self.indicator, window=int(self.window), rule=self.rule,
            threshold=float(self.threshold), horizon=int(self.horizon), direction=int(self.direction),
            asset=self.asset, source=self.source, claimed_sharpe=self.claimed_sharpe,
            n_trials=max(1, int(self.n_trials)),
        )
        spec.validate()
        return spec

    @property
    def mappable(self) -> bool:
        return self.indicator in INDICATORS and self.rule in RULES

@dataclass
class ClaimVerdict:
    """The honest bench verdict for one posted claim."""

    claim_id: str
    asset: str
    text: str
    source: str
    benchable: bool
    passed: bool
    net_sharpe: float
    buy_hold_sharpe: float
    dsr: float
    discounted_sharpe: float
    oos_mean_sharpe: float
    n: int
    note: str = ""

    @property
    def beats_passive(self) -> bool:
        return bool(np.isfinite(self.net_sharpe) and self.net_sharpe > self.buy_hold_sharpe)

    @property
    def verdict(self) -> str:
        if not self.benchable:
            return "NOT_BENCHABLE"
        return "TRADABLE" if self.passed else "NOT_TRADABLE"

def bench_posted_claim(claim: PostedClaim, prices, *, cost_per_turnover: float = 5e-4,
                       periods_per_year: float = 252.0, n_splits: int = 5, embargo: int = 5,
                       seed: int = 0) -> ClaimVerdict:
    """Map one claim to an EdgeSpec and run it through the DSR skeptic bench against ``prices``.

    Returns a NOT_BENCHABLE verdict (never raises) when the claim is unmappable or the series is too short.
    """
    base = dict(claim_id=claim.claim_id, asset=claim.asset, text=claim.text, source=claim.source,
                net_sharpe=float("nan"), buy_hold_sharpe=float("nan"), dsr=float("nan"),
                discounted_sharpe=float("nan"), oos_mean_sharpe=float("nan"))
    p = np.asarray(prices, dtype=float)
    p = p[np.isfinite(p)]
    try:
        spec = claim.to_spec()
    except Exception as e:  # unmappable indicator/rule/params
        return ClaimVerdict(**base, benchable=False, passed=False, n=int(p.size),
                            note=f"unmappable: {e}")
    need = max(spec.window + 2, n_splits + embargo + 2, 252)
    if p.size < need:
        return ClaimVerdict(**base, benchable=False, passed=False, n=int(p.size),
                            note=f"too few bars ({p.size} < {need}) to bench")
    v = run_edge_test(spec, p, cost_per_turnover=cost_per_turnover, periods_per_year=periods_per_year,
                      n_splits=n_splits, embargo=embargo, seed=seed)
    base.update(net_sharpe=v.net_sharpe, buy_hold_sharpe=v.buy_hold_sharpe, dsr=v.dsr,
                discounted_sharpe=v.discounted_sharpe, oos_mean_sharpe=v.oos_mean_sharpe)
    return ClaimVerdict(**base, benchable=True, passed=bool(v.passed), n=int(p.size),
                        note=("beats buy&hold; DSR clears; OOS+" if v.passed else "fails the gate"))

@dataclass
class PostedEdgeReport:
    verdicts: list[ClaimVerdict] = field(default_factory=list)

    @property
    def n_tradable(self) -> int:
        return sum(v.verdict == "TRADABLE" for v in self.verdicts)

    @property
    def n_not_tradable(self) -> int:
        return sum(v.verdict == "NOT_TRADABLE" for v in self.verdicts)

    @property
    def n_not_benchable(self) -> int:
        return sum(v.verdict == "NOT_BENCHABLE" for v in self.verdicts)

    def summary(self) -> str:
        lines = [
            "=" * 84,
            "POSTED-EDGE BENCH — mine a posted claim, run it through the DSR skeptic gate",
            "=" * 84,
            f"{len(self.verdicts)} claim(s):  {self.n_tradable} TRADABLE · {self.n_not_tradable} "
            f"NOT_TRADABLE · {self.n_not_benchable} NOT_BENCHABLE",
            "-" * 84,
            f"{'claim':<26}{'asset':>6}{'net Sh':>9}{'b&h':>7}{'DSR':>7}{'OOS':>7}  verdict",
        ]
        for v in self.verdicts:
            f = lambda x: (f"{x:+.2f}" if np.isfinite(x) else "  —")
            lines.append(f"{v.claim_id[:25]:<26}{v.asset:>6}{f(v.net_sharpe):>9}{f(v.buy_hold_sharpe):>7}"
                         f"{(f'{v.dsr:.2f}' if np.isfinite(v.dsr) else '  —'):>7}{f(v.oos_mean_sharpe):>7}"
                         f"  {v.verdict}")
        lines += [
            "-" * 84,
            "Posted edges carry a multiplicity haircut (n_trials) into the DSR — they are survivors of an",
            "unknown private search. The modal honest outcome is NOT_TRADABLE: a real effect that doesn't",
            "beat passive buy-&-hold after costs + deflation. TRADABLE here is only a candidate for the",
            "program's deeper capacity / decay tests. NOT_BENCHABLE = outside the single-asset daily harness.",
            "=" * 84,
        ]
        return "\n".join(lines)

def _fetch_prices(asset, source, lookback_days):
    from snapshot_source import now_epoch
    end = now_epoch()
    observations = source.fetch([f"{asset}.close"], end-lookback_days*86400, end)
    return np.array([o.value for o in sorted(observations, key=lambda o:o.t_event)], dtype=float)
