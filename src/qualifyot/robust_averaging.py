from __future__ import annotations

"""Training-only graph averaging with an explicit upper-tail non-harm screen."""

from dataclasses import asdict, dataclass
import math

import numpy as np

from .safe_averaging import simplex_grid, _patient_mean_losses, _validate_inputs
from .weight_selection import simultaneous_ucb


@dataclass(frozen=True)
class RobustAverageResult:
    weights: np.ndarray
    chosen_index: int
    frozen_index: int
    frozen_grid_index: int
    mean_risk: float
    frozen_risk: float
    mean_harm_ucb: float
    mean_harm_margin: float
    tail_fraction: float
    tail_harm: float
    tail_harm_ucb: float
    tail_harm_margin: float
    feasible_count: int
    grid_size: int
    mean_critical_value: float
    tail_critical_deviation: float
    one_se: bool

    @property
    def fallback_used(self) -> bool:
        return bool(self.chosen_index == self.frozen_grid_index)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["weights"] = self.weights.tolist()
        d["fallback_used"] = self.fallback_used
        return d


def _cvar_matrix(harm: np.ndarray, tail_fraction: float) -> np.ndarray:
    """Upper-tail CVaR for each column of patients x candidates harm matrix."""
    n = harm.shape[0]
    k = max(1, int(math.ceil(float(tail_fraction) * n)))
    part = np.partition(harm, n - k, axis=0)[n - k :, :]
    return part.mean(axis=0)


def robust_graph_average(
    target,
    predictions,
    patients,
    *,
    frozen_index: int = 0,
    step: float = 0.1,
    alpha_mean: float = 0.05,
    mean_harm_margin: float = 0.0,
    tail_fraction: float = 0.20,
    alpha_tail: float = 0.05,
    tail_harm_margin: float = 0.0,
    bootstrap: int = 5_000,
    seed: int = 0,
    one_se: bool = True,
) -> RobustAverageResult:
    """Select a graph mixture under simultaneous mean- and tail-harm screens.

    The mean screen is the same simultaneous max-t band used by SGA.  The
    upper-tail screen uses a patient bootstrap over the full pre-specified
    mixture grid and calibrates the maximum positive CVaR deviation from the
    observed CVaR.  This second screen is an *empirical robustness procedure*;
    it is not advertised as an exact finite-sample subgroup-safety theorem.

    The frozen graph is always feasible by identity and is used as fallback if
    no non-frozen mixture passes both screens.
    """
    target, pred, patients = _validate_inputs(target, predictions, patients)
    if not (0 < tail_fraction <= 1):
        raise ValueError("tail_fraction must lie in (0,1]")
    if not (0 < alpha_mean < 1 and 0 < alpha_tail < 1):
        raise ValueError("alpha levels must lie in (0,1)")
    if int(bootstrap) < 100:
        raise ValueError("bootstrap must be >=100")
    m = pred.shape[1]
    if not (0 <= frozen_index < m):
        raise ValueError("invalid frozen_index")
    W = simplex_grid(m, step)
    fw = np.zeros(m); fw[frozen_index] = 1.0
    ids_f = np.flatnonzero(np.all(np.isclose(W, fw, atol=1e-12), axis=1))
    if len(ids_f) != 1:
        raise RuntimeError("frozen fallback missing or duplicated in simplex grid")
    fgi = int(ids_f[0])

    R = _patient_mean_losses(target, pred, patients, W)  # patient x mixture
    H = R - R[:, fgi][:, None]
    mean_ucb, _, mean_q = simultaneous_ucb(H, alpha=alpha_mean, B=bootstrap, seed=seed, zero_index=fgi)
    mean_ok = mean_ucb <= float(mean_harm_margin) + 1e-12

    cvar = _cvar_matrix(H, tail_fraction)
    rng = np.random.default_rng(seed + 104729)
    n = H.shape[0]
    max_dev = np.empty(int(bootstrap), float)
    batch = 100
    for st in range(0, int(bootstrap), batch):
        b = min(batch, int(bootstrap) - st)
        idx = rng.integers(0, n, size=(b, n))
        for j in range(b):
            cb = _cvar_matrix(H[idx[j]], tail_fraction)
            max_dev[st + j] = float(np.max(cb - cvar))
    tail_q = float(np.quantile(max_dev, 1 - alpha_tail))
    tail_ucb = cvar + tail_q
    # identity fallback has exactly zero harm under every patient draw
    cvar[fgi] = 0.0
    tail_ucb[fgi] = 0.0
    tail_ok = tail_ucb <= float(tail_harm_margin) + 1e-12

    feasible = np.isfinite(mean_ucb) & np.isfinite(tail_ucb) & mean_ok & tail_ok
    feasible[fgi] = True
    eligible = np.flatnonzero(feasible)
    if not len(eligible):
        raise RuntimeError("internal error: frozen fallback must remain feasible")
    mr = R.mean(axis=0)
    best = int(eligible[np.argmin(mr[eligible])])
    chosen = best
    if one_se and n > 1:
        se_best = float(R[:, best].std(ddof=1) / math.sqrt(n))
        near = eligible[mr[eligible] <= mr[best] + se_best + 1e-15]
        chosen = int(sorted(
            near,
            key=lambda j: (
                -W[j, frozen_index],
                int(np.count_nonzero(W[j] > 1e-12)),
                mr[j],
                tuple(W[j]),
            ),
        )[0])

    return RobustAverageResult(
        weights=W[chosen].copy(),
        chosen_index=chosen,
        frozen_index=int(frozen_index),
        frozen_grid_index=fgi,
        mean_risk=float(mr[chosen]),
        frozen_risk=float(mr[fgi]),
        mean_harm_ucb=float(mean_ucb[chosen]),
        mean_harm_margin=float(mean_harm_margin),
        tail_fraction=float(tail_fraction),
        tail_harm=float(cvar[chosen]),
        tail_harm_ucb=float(tail_ucb[chosen]),
        tail_harm_margin=float(tail_harm_margin),
        feasible_count=int(feasible.sum()),
        grid_size=int(len(W)),
        mean_critical_value=float(mean_q),
        tail_critical_deviation=float(tail_q),
        one_se=bool(one_se),
    )
