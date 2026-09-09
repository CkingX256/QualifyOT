from __future__ import annotations

"""Patient-level benchmark tests for fixed held-out prediction contrasts.

These comparators are deliberately simple and transparent.  They are intended
for head-to-head calibration studies, not as hidden gates in the primary
Movement–Utility–Retention decision.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.stats import t as student_t

from .bayesian_evidence import bayesian_bootstrap_mean
from .inference import one_sided_lcb


@dataclass(frozen=True)
class BenchmarkResult:
    n_patients: int
    mean_effect: float
    t_p_one_sided: float
    t_pass: bool
    signflip_p_one_sided: float
    signflip_pass: bool
    percentile_lcb: float
    percentile_pass: bool
    bootstrap_t_lcb: float
    bootstrap_t_pass: bool
    bayesian_bootstrap_probability_positive: float
    bayesian_bootstrap_pass: bool
    alpha: float
    posterior_probability_threshold: float

    def to_dict(self):
        return asdict(self)


def _x(values) -> np.ndarray:
    x = np.asarray(values, float).reshape(-1)
    if len(x) < 2:
        raise ValueError("at least two independent patient effects are required")
    if not np.isfinite(x).all():
        raise ValueError("patient effects must be finite")
    return x


def paired_t_one_sided(values, *, margin: float = 0.0) -> tuple[float, float]:
    x = _x(values) - float(margin)
    n = len(x)
    s = x.std(ddof=1)
    if s <= 1e-15:
        if x.mean() > 0:
            return math.inf, 0.0
        if x.mean() < 0:
            return -math.inf, 1.0
        return 0.0, 1.0
    t = float(x.mean() / (s / math.sqrt(n)))
    p = float(student_t.sf(t, n - 1))
    return t, p


def signflip_test_one_sided(
    values,
    *,
    margin: float = 0.0,
    permutations: int = 10_000,
    seed: int = 0,
    exact_max_n: int = 16,
) -> float:
    """Paired sign-flip randomization p-value for a positive mean effect.

    Exact enumeration is used for n<=``exact_max_n``; otherwise Monte Carlo
    signs are used with a +1 correction.  Validity requires sign-exchangeable
    centered patient effects under the null, so this is a benchmark rather
    than a universal replacement for patient bootstrap inference.
    """
    x = _x(values) - float(margin)
    obs = float(x.mean())
    n = len(x)
    if n <= exact_max_n:
        total = 1 << n
        ge = 0
        for mask in range(total):
            signs = np.ones(n, float)
            for j in range(n):
                if (mask >> j) & 1:
                    signs[j] = -1.0
            ge += int(float(np.mean(signs * x)) >= obs - 1e-15)
        return float(ge / total)
    if int(permutations) < 100:
        raise ValueError("permutations must be >=100")
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(int(permutations), n))
    stats = (signs * x[None, :]).mean(axis=1)
    return float((1 + np.sum(stats >= obs - 1e-15)) / (1 + len(stats)))


def benchmark_patient_effects(
    values,
    *,
    margin: float = 0.0,
    alpha: float = 0.05,
    bootstrap: int = 5_000,
    permutations: int = 10_000,
    bayes_draws: int = 20_000,
    posterior_probability_threshold: float = 0.95,
    signflip_exact_max_n: int = 16,
    seed: int = 0,
) -> BenchmarkResult:
    x = _x(values)
    if not (0 < alpha < 1):
        raise ValueError("alpha must lie in (0,1)")
    if not (0 < posterior_probability_threshold < 1):
        raise ValueError("posterior_probability_threshold must lie in (0,1)")
    _, p_t = paired_t_one_sided(x, margin=margin)
    p_sf = signflip_test_one_sided(x, margin=margin, permutations=permutations, seed=seed + 1, exact_max_n=signflip_exact_max_n)
    lcb_pct = one_sided_lcb(x - float(margin), method="percentile", alpha=alpha, bootstrap=bootstrap, seed=seed + 2)
    lcb_bt = one_sided_lcb(x - float(margin), method="bootstrap_t", alpha=alpha, bootstrap=bootstrap, seed=seed + 3)
    bb = bayesian_bootstrap_mean(x, margin=margin, draws=bayes_draws, seed=seed + 4)
    return BenchmarkResult(
        n_patients=int(len(x)),
        mean_effect=float(x.mean()),
        t_p_one_sided=float(p_t),
        t_pass=bool(p_t < alpha),
        signflip_p_one_sided=float(p_sf),
        signflip_pass=bool(p_sf < alpha),
        percentile_lcb=float(lcb_pct + margin),
        percentile_pass=bool(lcb_pct > 0),
        bootstrap_t_lcb=float(lcb_bt + margin),
        bootstrap_t_pass=bool(lcb_bt > 0),
        bayesian_bootstrap_probability_positive=float(bb.probability_above_margin),
        bayesian_bootstrap_pass=bool(bb.probability_above_margin >= posterior_probability_threshold),
        alpha=float(alpha),
        posterior_probability_threshold=float(posterior_probability_threshold),
    )
