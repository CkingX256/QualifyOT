from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.stats import t as student_t


@dataclass(frozen=True)
class MeanInterval:
    estimate: float
    lower: float
    upper: float
    method: str
    confidence: float
    n: int

    def to_dict(self) -> dict:
        return {
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "method": self.method,
            "confidence": self.confidence,
            "n": self.n,
        }


def _finite_sample(x) -> np.ndarray:
    x = np.asarray(x, float).reshape(-1)
    if x.size < 2:
        raise ValueError("at least two independent patient values are required")
    if not np.isfinite(x).all():
        raise ValueError("patient values must be finite")
    return x


def patient_equal_values(values, patients) -> tuple[np.ndarray, np.ndarray]:
    """Collapse row-level values to one equally weighted value per physical patient."""
    v = np.asarray(values, float)
    p = np.asarray(patients).astype(str)
    if v.ndim != 1 or len(v) != len(p):
        raise ValueError("values must be one-dimensional and match patients")
    if not np.isfinite(v).all():
        raise ValueError("values must be finite")
    up = np.unique(p)
    return up, np.asarray([v[p == pid].mean() for pid in up], float)


def mean_interval(
    x,
    *,
    method: str = "t",
    confidence: float = 0.95,
    bootstrap: int = 2000,
    seed: int = 0,
) -> MeanInterval:
    """Two-sided patient-level interval for a mean.

    Supported methods are ``t``, ``percentile`` and ``bootstrap_t``.  The
    bootstrap-t implementation studentizes each bootstrap mean with its own
    bootstrap standard error.  Degenerate zero-variance samples return the
    point mass interval rather than fabricating uncertainty.
    """
    x = _finite_sample(x)
    if not (0 < confidence < 1):
        raise ValueError("confidence must lie in (0, 1)")
    n = len(x)
    mu = float(x.mean())
    sd = float(x.std(ddof=1))
    if sd <= 1e-15:
        return MeanInterval(mu, mu, mu, method, float(confidence), n)
    alpha = 1.0 - confidence
    if method == "t":
        q = float(student_t.ppf(1 - alpha / 2, n - 1))
        half = q * sd / math.sqrt(n)
        return MeanInterval(mu, mu - half, mu + half, method, float(confidence), n)

    if int(bootstrap) < 100:
        raise ValueError("bootstrap must be >= 100 for bootstrap intervals")
    rng = np.random.default_rng(seed)
    B = int(bootstrap)
    idx = rng.integers(0, n, size=(B, n))
    xb = x[idx]
    mb = xb.mean(axis=1)
    if method == "percentile":
        lo, hi = np.quantile(mb, [alpha / 2, 1 - alpha / 2])
        return MeanInterval(mu, float(lo), float(hi), method, float(confidence), n)
    if method == "bootstrap_t":
        seb = xb.std(axis=1, ddof=1) / math.sqrt(n)
        valid = np.isfinite(seb) & (seb > 1e-15)
        if not valid.any():
            return MeanInterval(mu, mu, mu, method, float(confidence), n)
        tstar = (mb[valid] - mu) / seb[valid]
        qlo, qhi = np.quantile(tstar, [alpha / 2, 1 - alpha / 2])
        se = sd / math.sqrt(n)
        # Bootstrap-t inversion: [mu - q_(1-a/2) se, mu - q_(a/2) se].
        return MeanInterval(mu, float(mu - qhi * se), float(mu - qlo * se), method, float(confidence), n)
    raise ValueError("method must be one of: t, percentile, bootstrap_t")


def one_sided_lcb(
    x,
    *,
    method: str = "t",
    alpha: float = 0.05,
    bootstrap: int = 2000,
    seed: int = 0,
) -> float:
    """One-sided lower confidence bound for a patient-level mean."""
    x = _finite_sample(x)
    if not (0 < alpha < 1):
        raise ValueError("alpha must lie in (0, 1)")
    n = len(x)
    mu = float(x.mean())
    sd = float(x.std(ddof=1))
    if sd <= 1e-15:
        return mu
    if method == "t":
        return float(mu - student_t.ppf(1 - alpha, n - 1) * sd / math.sqrt(n))
    if int(bootstrap) < 100:
        raise ValueError("bootstrap must be >= 100 for bootstrap intervals")
    rng = np.random.default_rng(seed)
    B = int(bootstrap)
    idx = rng.integers(0, n, size=(B, n))
    xb = x[idx]
    mb = xb.mean(axis=1)
    if method == "percentile":
        return float(np.quantile(mb, alpha))
    if method == "bootstrap_t":
        seb = xb.std(axis=1, ddof=1) / math.sqrt(n)
        valid = np.isfinite(seb) & (seb > 1e-15)
        if not valid.any():
            return mu
        tstar = (mb[valid] - mu) / seb[valid]
        q = float(np.quantile(tstar, 1 - alpha))
        return float(mu - q * sd / math.sqrt(n))
    raise ValueError("method must be one of: t, percentile, bootstrap_t")
