from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math

import numpy as np

from .weight_selection import simultaneous_ucb


@dataclass(frozen=True)
class SafeAverageResult:
    weights: np.ndarray
    chosen_index: int
    frozen_index: int
    frozen_grid_index: int
    mean_risk: float
    frozen_risk: float
    safety_ucb: float
    safety_margin: float
    critical_value: float
    feasible_count: int
    grid_size: int
    grid_step: float

    @property
    def improved_point_risk(self) -> bool:
        return bool(self.mean_risk < self.frozen_risk)

    @property
    def fallback_used(self) -> bool:
        return bool(self.chosen_index == self.frozen_grid_index)

    @property
    def certified_nonharm_on_selection_sample(self) -> bool:
        return bool(self.safety_ucb <= self.safety_margin + 1e-12)

    def to_dict(self) -> dict:
        return {
            "weights": self.weights.tolist(),
            "chosen_index": self.chosen_index,
            "frozen_index": self.frozen_index,
            "frozen_grid_index": self.frozen_grid_index,
            "mean_risk": self.mean_risk,
            "frozen_risk": self.frozen_risk,
            "safety_ucb": self.safety_ucb,
            "safety_margin": self.safety_margin,
            "critical_value": self.critical_value,
            "feasible_count": self.feasible_count,
            "grid_size": self.grid_size,
            "grid_step": self.grid_step,
            "improved_point_risk": self.improved_point_risk,
            "fallback_used": self.fallback_used,
            "certified_nonharm_on_selection_sample": self.certified_nonharm_on_selection_sample,
        }


def simplex_grid(n_models: int, step: float = 0.1) -> np.ndarray:
    if n_models < 1:
        raise ValueError("n_models must be >= 1")
    if not math.isfinite(float(step)) or not (0 < step <= 1):
        raise ValueError("step must lie in (0,1]")
    inv = round(1.0 / step)
    if not np.isclose(inv * step, 1.0, atol=1e-12):
        raise ValueError("step must evenly divide 1")
    inv = int(inv)
    rows = []
    for z in product(range(inv + 1), repeat=n_models):
        if sum(z) == inv:
            rows.append(np.asarray(z, float) / inv)
    return np.vstack(rows)


def _validate_inputs(target, predictions, patients):
    target = np.asarray(target, float)
    pred = np.asarray(predictions, float)
    patients = np.asarray(patients).astype(str)
    if target.ndim != 2:
        raise ValueError("target must have shape rows x states")
    if pred.ndim != 3:
        raise ValueError("predictions must have shape rows x models x states")
    if pred.shape[0] != len(target) or len(patients) != len(target) or pred.shape[2] != target.shape[1]:
        raise ValueError("row/state dimensions do not match")
    if pred.shape[1] < 1:
        raise ValueError("at least one graph prediction is required")
    if len(np.unique(patients)) < 2:
        raise ValueError("at least two physical patients are required")
    if not np.isfinite(target).all() or not np.isfinite(pred).all():
        raise ValueError("targets and predictions must be finite")
    if (target < -1e-10).any() or (pred < -1e-10).any():
        raise ValueError("targets and predictions must be non-negative compositions")
    if not np.allclose(target.sum(1), 1.0, atol=1e-6):
        raise ValueError("targets must lie on the probability simplex")
    if not np.allclose(pred.sum(2), 1.0, atol=1e-6):
        raise ValueError("every graph prediction must lie on the probability simplex")
    return target, pred, patients


def _patient_mean_losses(target, predictions, patients, weights) -> np.ndarray:
    target, pred, patients = _validate_inputs(target, predictions, patients)
    mixed = np.einsum("wm,rms->rws", weights, pred)
    row_loss = np.mean(np.abs(mixed - target[:, None, :]), axis=2)
    up = np.unique(patients)
    return np.vstack([row_loss[patients == p].mean(axis=0) for p in up])


def safe_graph_average(
    target,
    predictions,
    patients,
    *,
    frozen_index: int = 0,
    step: float = 0.1,
    alpha: float = 0.05,
    safety_margin: float = 0.0,
    bootstrap: int = 5000,
    seed: int = 0,
    one_se: bool = True,
) -> SafeAverageResult:
    """Choose a convex graph ensemble using training patients only.

    For every weight vector w on a pre-specified simplex grid, let
    D_i(w)=L_i(q_w)-L_i(q_frozen).  A simultaneous upper confidence band is
    calibrated over the *entire* weight grid.  Only mixtures with
    UCB[D(w)] <= safety_margin are eligible; the frozen one-hot vector is
    included by identity as a deterministic fallback.

    Conditional guarantee: on the simultaneous coverage event, every selected
    non-fallback mixture has population risk difference no larger than the
    specified safety margin *for the fixed prediction library and calibration
    population*.  The statement does not guarantee no harm after cohort shift
    and is not a theorem for dependent outer-LOPO refitting.
    """
    target, pred, patients = _validate_inputs(target, predictions, patients)
    m = pred.shape[1]
    if not (0 <= frozen_index < m):
        raise ValueError("invalid frozen_index")
    if not (0 < alpha < 1):
        raise ValueError("alpha must lie in (0,1)")
    if not math.isfinite(float(safety_margin)):
        raise ValueError("safety_margin must be finite")
    if int(bootstrap) < 50:
        raise ValueError("bootstrap must be >=50")

    W = simplex_grid(m, step)
    frozen_w = np.zeros(m, float)
    frozen_w[frozen_index] = 1.0
    match = np.flatnonzero(np.all(np.isclose(W, frozen_w, atol=1e-12), axis=1))
    if not len(match):
        raise RuntimeError("simplex grid does not contain frozen fallback")
    frozen_grid_index = int(match[0])

    R = _patient_mean_losses(target, pred, patients, W)
    frozen_patient = R[:, frozen_grid_index]
    D = R - frozen_patient[:, None]
    ucb, _, q = simultaneous_ucb(D, alpha=alpha, B=bootstrap, seed=seed, zero_index=frozen_grid_index)
    feasible = np.isfinite(ucb) & (ucb <= safety_margin + 1e-12)
    feasible[frozen_grid_index] = True
    ids = np.flatnonzero(feasible)
    if not len(ids):
        raise RuntimeError("internal error: frozen fallback must remain feasible")

    mean_risk = R.mean(axis=0)
    best = int(ids[np.argmin(mean_risk[ids])])
    chosen = best
    if one_se and R.shape[0] > 1:
        se_best = float(R[:, best].std(ddof=1) / math.sqrt(R.shape[0]))
        eligible = ids[mean_risk[ids] <= mean_risk[best] + se_best + 1e-15]
        chosen = int(sorted(
            eligible,
            key=lambda j: (
                -W[j, frozen_index],
                int(np.count_nonzero(W[j] > 1e-12)),
                mean_risk[j],
                tuple(W[j]),
            ),
        )[0])

    return SafeAverageResult(
        weights=W[chosen].copy(),
        chosen_index=chosen,
        frozen_index=frozen_index,
        frozen_grid_index=frozen_grid_index,
        mean_risk=float(mean_risk[chosen]),
        frozen_risk=float(mean_risk[frozen_grid_index]),
        safety_ucb=float(ucb[chosen]),
        safety_margin=float(safety_margin),
        critical_value=float(q),
        feasible_count=int(feasible.sum()),
        grid_size=int(len(W)),
        grid_step=float(step),
    )


@dataclass(frozen=True)
class HonestSafeAverageResult:
    weights: np.ndarray
    chosen_index: int
    frozen_grid_index: int
    mean_risk: float
    frozen_risk: float
    safety_ucb: float
    safety_margin: float
    feasible_count: int
    grid_size: int
    alpha: float
    bound: str = "localized_hoeffding_union"

    @property
    def fallback_used(self) -> bool:
        return bool(self.chosen_index == self.frozen_grid_index)

    def to_dict(self) -> dict:
        return {
            "weights": self.weights.tolist(),
            "chosen_index": self.chosen_index,
            "frozen_grid_index": self.frozen_grid_index,
            "mean_risk": self.mean_risk,
            "frozen_risk": self.frozen_risk,
            "safety_ucb": self.safety_ucb,
            "safety_margin": self.safety_margin,
            "feasible_count": self.feasible_count,
            "grid_size": self.grid_size,
            "alpha": self.alpha,
            "bound": self.bound,
            "fallback_used": self.fallback_used,
        }


def honest_safe_graph_average(
    target,
    predictions,
    patients,
    *,
    frozen_index: int = 0,
    step: float = 0.1,
    alpha: float = 0.05,
    safety_margin: float = 0.0,
    one_se: bool = True,
) -> HonestSafeAverageResult:
    """Finite-sample safe graph averaging on an independent calibration set.

    The graph prediction library and simplex weight grid must be frozen before
    observing these calibration targets.  For each patient and mixture w,

        |L_i(q_w) - L_i(q_frozen)| <= C_i(w),

    where C_i(w) is the patient-average MAE distance between the two *fixed*
    predictions.  Conditional on the frozen prediction arrays/source
    covariates, Hoeffding's inequality for independent bounded patient losses
    gives a simultaneous upper band over the finite grid after a union bound.
    This is deliberately conservative but is a genuine finite-sample fallback;
    the existing bootstrap max-t SGA remains the efficient development mode.
    """
    target, pred, patients = _validate_inputs(target, predictions, patients)
    if not (0 < alpha < 1):
        raise ValueError("alpha must lie in (0,1)")
    if not math.isfinite(float(safety_margin)):
        raise ValueError("safety_margin must be finite")
    m = pred.shape[1]
    if not (0 <= frozen_index < m):
        raise ValueError("invalid frozen_index")
    W = simplex_grid(m, step)
    frozen_w = np.zeros(m, float); frozen_w[frozen_index] = 1.0
    match = np.flatnonzero(np.all(np.isclose(W, frozen_w, atol=1e-12), axis=1))
    if not len(match):
        raise RuntimeError("simplex grid does not contain frozen fallback")
    frozen_grid_index = int(match[0])

    # Patient mean target losses for every mixture.
    R = _patient_mean_losses(target, pred, patients, W)
    D = R - R[:, frozen_grid_index][:, None]

    # Target-free Lipschitz envelope C_i(w): MAE(q_w,q_frozen), averaged
    # within physical patient.  Reverse triangle inequality yields |D_i|<=C_i.
    mixed = np.einsum("wm,rms->rws", W, pred)
    frozen_pred = pred[:, frozen_index, :]
    row_c = np.mean(np.abs(mixed - frozen_pred[:, None, :]), axis=2)
    up = np.unique(patients)
    C = np.vstack([row_c[patients == p].mean(axis=0) for p in up])
    if (np.abs(D) > C + 1e-10).any():
        raise RuntimeError("internal Lipschitz envelope violation")

    n = len(up); M = len(W)
    logterm = math.log(M / alpha)
    radius = np.sqrt(2.0 * np.sum(C * C, axis=0) * logterm) / n
    ucb = D.mean(axis=0) + radius
    ucb[frozen_grid_index] = 0.0
    feasible = ucb <= float(safety_margin) + 1e-12
    feasible[frozen_grid_index] = True
    ids = np.flatnonzero(feasible)
    mean_risk = R.mean(axis=0)
    best = int(ids[np.argmin(mean_risk[ids])])
    chosen = best
    if one_se and R.shape[0] > 1:
        se_best = float(R[:, best].std(ddof=1) / math.sqrt(R.shape[0]))
        eligible = ids[mean_risk[ids] <= mean_risk[best] + se_best + 1e-15]
        chosen = int(sorted(
            eligible,
            key=lambda j: (
                -W[j, frozen_index],
                int(np.count_nonzero(W[j] > 1e-12)),
                mean_risk[j],
                tuple(W[j]),
            ),
        )[0])
    return HonestSafeAverageResult(
        weights=W[chosen].copy(),
        chosen_index=chosen,
        frozen_grid_index=frozen_grid_index,
        mean_risk=float(mean_risk[chosen]),
        frozen_risk=float(mean_risk[frozen_grid_index]),
        safety_ucb=float(ucb[chosen]),
        safety_margin=float(safety_margin),
        feasible_count=int(feasible.sum()),
        grid_size=int(M),
        alpha=float(alpha),
    )
