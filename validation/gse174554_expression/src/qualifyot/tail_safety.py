from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .model import mae


@dataclass(frozen=True)
class TailSafetyProfile:
    n_patients: int
    tail_fraction: float
    mean_harm: float
    mean_harm_ci: tuple[float, float]
    harmed_fraction: float
    upper_quantile_harm: float
    cvar_harm: float
    cvar_ci: tuple[float, float]
    mean_nonharm_supported: bool
    tail_nonharm_supported: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _patient_harm(target, reference, retained, patients) -> np.ndarray:
    target = np.asarray(target, float)
    reference = np.asarray(reference, float)
    retained = np.asarray(retained, float)
    patients = np.asarray(patients).astype(str)
    h = mae(target, retained) - mae(target, reference)
    return np.asarray([h[patients == p].mean() for p in np.unique(patients)], float)


def _cvar_upper(x: np.ndarray, tail_fraction: float) -> float:
    n_tail = max(1, int(np.ceil(tail_fraction * len(x))))
    return float(np.sort(x)[-n_tail:].mean())


def tail_safety_profile(
    target,
    reference,
    retained,
    patients,
    *,
    tail_fraction: float = 0.10,
    alpha: float = 0.05,
    bootstrap: int = 5000,
    seed: int = 0,
    nonharm_margin: float = 0.0,
) -> TailSafetyProfile:
    """Summarize average and upper-tail patient harm relative to a reference.

    Positive harm is worse than the reference. Tail non-harm is intentionally
    reported as an orthogonal safety axis, not as a mandatory qualification
    gate in the current implementation.
    """

    if not (0 < tail_fraction <= 1):
        raise ValueError("tail_fraction must lie in (0, 1]")
    x = _patient_harm(target, reference, retained, patients)
    n = len(x)
    if n < 2:
        raise ValueError("at least two patients are required")
    q = float(np.quantile(x, 1 - tail_fraction))
    cvar = _cvar_upper(x, tail_fraction)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(bootstrap), n))
    z = x[idx]
    mean_boot = z.mean(axis=1)
    cvar_boot = np.asarray([_cvar_upper(v, tail_fraction) for v in z], float)
    lo_q, hi_q = alpha / 2, 1 - alpha / 2
    mean_ci = (float(np.quantile(mean_boot, lo_q)), float(np.quantile(mean_boot, hi_q)))
    cvar_ci = (float(np.quantile(cvar_boot, lo_q)), float(np.quantile(cvar_boot, hi_q)))
    return TailSafetyProfile(
        n_patients=n,
        tail_fraction=float(tail_fraction),
        mean_harm=float(x.mean()),
        mean_harm_ci=mean_ci,
        harmed_fraction=float((x > nonharm_margin).mean()),
        upper_quantile_harm=q,
        cvar_harm=cvar,
        cvar_ci=cvar_ci,
        mean_nonharm_supported=bool(mean_ci[1] < nonharm_margin),
        tail_nonharm_supported=bool(cvar_ci[1] < nonharm_margin),
    )
