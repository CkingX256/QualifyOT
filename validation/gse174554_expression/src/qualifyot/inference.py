from __future__ import annotations

"""Patient-level inference utilities for QualifyOT.

The module deliberately separates two evidence modes.

1. ``honest_confirmation_iut``: all modelling/selection is assumed frozen on
   an independent development sample.  Confirmation-patient contributions are
   then independent conditional on the development sample, allowing explicit
   finite-sample Hoeffding bounds for bounded losses.
2. ``lopo_*`` helpers: efficient leave-one-patient-out analyses.  They provide
   stability diagnostics and conventional/studentized intervals, but the
   module never labels those procedures as exact finite-sample guarantees.

Movement is tested through the *margin contrast*

    G_M = E[A_i - delta_M B_i]

rather than by directly testing a ratio.  When E[B_i] > 0 this is exactly
 equivalent to E[A_i]/E[B_i] > delta_M, while propagating numerator/denominator
covariation and avoiding division by a near-zero denominator in the test.
The ratio remains available as an interpretable descriptive effect size.
"""

from dataclasses import dataclass, asdict
import hashlib
import json
import math
from typing import Iterable, Mapping

import numpy as np
from scipy.stats import norm, t as student_t

from .basic_inference import MeanInterval, patient_equal_values, mean_interval, one_sided_lcb


@dataclass(frozen=True)
class MovementMarginEvidence:
    ratio: float
    numerator: float
    denominator: float
    contrast: float
    contrast_lcb: float
    contrast_ucb: float
    method: str
    alpha: float
    denominator_floor: float
    ratio_identifiable: bool
    n_patients: int

    @property
    def supported(self) -> bool:
        return bool(self.contrast_lcb > 0.0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["supported"] = self.supported
        return d


@dataclass(frozen=True)
class HonestAxisEvidence:
    estimate: float
    lower_bound: float
    lower_support: float
    upper_support: float
    alpha: float
    n: int

    @property
    def supported(self) -> bool:
        return bool(self.lower_bound > 0.0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["supported"] = self.supported
        return d


@dataclass(frozen=True)
class HonestConfirmationResult:
    movement: HonestAxisEvidence
    utility: HonestAxisEvidence
    retention: HonestAxisEvidence
    qualified: bool
    component_alpha: float
    mode: str = "honest_confirmation"
    estimand_scope: str = "unspecified"

    def to_dict(self) -> dict:
        return {
            "movement": self.movement.to_dict(),
            "utility": self.utility.to_dict(),
            "retention": self.retention.to_dict(),
            "qualified": self.qualified,
            "component_alpha": self.component_alpha,
            "mode": self.mode,
            "estimand_scope": self.estimand_scope,
        }


@dataclass(frozen=True)
class StabilityDiagnostic:
    full_estimate: float
    leave_one_patient_out_estimates: tuple[float, ...]
    max_abs_shift: float
    rms_shift: float
    selection_gap: float | None = None

    def to_dict(self) -> dict:
        return {
            "full_estimate": self.full_estimate,
            "leave_one_patient_out_estimates": list(self.leave_one_patient_out_estimates),
            "max_abs_shift": self.max_abs_shift,
            "rms_shift": self.rms_shift,
            "selection_gap": self.selection_gap,
        }


@dataclass(frozen=True)
class FrozenAnalysisContract:
    candidate: str
    reference_rule: str
    ontology: tuple[str, ...]
    loss: str
    movement_margin: float
    lambda_grid: tuple[float, ...]
    inference_rule: str
    # Patient identity is a first-class scientific choice. Historical contracts
    # may leave both fields empty for backward reproducibility. New confirmatory
    # contracts should provide the physical patient IDs; the canonical patient
    # hash is then populated automatically and checked if supplied explicitly.
    patient_ids: tuple[str, ...] = ()
    patient_set_sha256: str | None = None
    extra: tuple[tuple[str, str], ...] = ()
    # Extended estimand-defining fields. They are optional for historical
    # compatibility; when left empty they are omitted from canonical_dict(),
    # preserving hashes of contracts created before this extension.
    prediction_task: str = ""
    eligibility_rule: str = ""
    preprocessing_protocol: str = ""
    hyperparameter_protocol: str = ""
    split_seed_rule: str = ""
    candidate_family: tuple[str, ...] = ()
    target_access_rule: str = ""

    def __post_init__(self) -> None:
        # A patient *set* is scientifically unordered. Canonicalising here avoids
        # contract hashes changing merely because metadata rows were reordered.
        ids = tuple(sorted({str(x).strip() for x in self.patient_ids if str(x).strip()}))
        object.__setattr__(self, "patient_ids", ids)

        supplied = self.patient_set_sha256
        if supplied is not None:
            supplied = str(supplied).strip().lower()
            if len(supplied) != 64 or any(c not in "0123456789abcdef" for c in supplied):
                raise ValueError("patient_set_sha256 must be a 64-character hexadecimal SHA256 digest")
            object.__setattr__(self, "patient_set_sha256", supplied)

        if ids:
            raw = json.dumps(list(ids), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            actual = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            if supplied is not None and supplied != actual:
                raise ValueError("patient_set_sha256 does not match canonical patient_ids")
            object.__setattr__(self, "patient_set_sha256", actual)

    def canonical_dict(self) -> dict:
        d = {
            "candidate": self.candidate,
            "reference_rule": self.reference_rule,
            "ontology": list(self.ontology),
            "loss": self.loss,
            "movement_margin": self.movement_margin,
            "lambda_grid": list(self.lambda_grid),
            "inference_rule": self.inference_rule,
            "patient_ids": list(self.patient_ids),
            "patient_set_sha256": self.patient_set_sha256,
            "extra": {k: v for k, v in self.extra},
        }
        # Do not perturb historical hashes when the extended fields are absent.
        extended = {
            "prediction_task": self.prediction_task,
            "eligibility_rule": self.eligibility_rule,
            "preprocessing_protocol": self.preprocessing_protocol,
            "hyperparameter_protocol": self.hyperparameter_protocol,
            "split_seed_rule": self.split_seed_rule,
            "candidate_family": list(self.candidate_family) if self.candidate_family else [],
            "target_access_rule": self.target_access_rule,
        }
        d.update({k: v for k, v in extended.items() if (v if isinstance(v, list) else str(v).strip())})
        return d

    @property
    def sha256(self) -> str:
        raw = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _finite_vector(x, name: str) -> np.ndarray:
    x = np.asarray(x, float).reshape(-1)
    if x.size < 2:
        raise ValueError(f"{name} requires at least two independent patients")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} must be finite")
    return x


def _patient_pair_means(values, patients) -> np.ndarray:
    _, x = patient_equal_values(values, patients)
    return np.asarray(x, float)


def movement_patient_components(numerator_rows, denominator_rows, patients, *, movement_margin: float = 0.01):
    """Return one numerator, denominator and margin contrast per patient."""
    if not math.isfinite(float(movement_margin)) or movement_margin < 0:
        raise ValueError("movement_margin must be finite and non-negative")
    a = _patient_pair_means(numerator_rows, patients)
    b = _patient_pair_means(denominator_rows, patients)
    if len(a) != len(b):
        raise RuntimeError("patient collapse produced mismatched Movement components")
    if (a < -1e-12).any() or (b < -1e-12).any():
        raise ValueError("Movement distance components must be non-negative")
    g = a - float(movement_margin) * b
    return a, b, g


def _one_sided_lcb_t(x: np.ndarray, alpha: float) -> float:
    n = len(x)
    mu = float(x.mean())
    sd = float(x.std(ddof=1))
    if sd <= 1e-15:
        return mu
    return float(mu - student_t.ppf(1 - alpha, n - 1) * sd / math.sqrt(n))


def _one_sided_ucb_t(x: np.ndarray, alpha: float) -> float:
    n = len(x)
    mu = float(x.mean())
    sd = float(x.std(ddof=1))
    if sd <= 1e-15:
        return mu
    return float(mu + student_t.ppf(1 - alpha, n - 1) * sd / math.sqrt(n))


def _bca_quantile_mean(x: np.ndarray, q: float, *, bootstrap: int, seed: int) -> float:
    """BCa quantile for the sample mean.

    This is a diagnostic resampling interval, not a finite-sample guarantee.
    """
    if not (0 < q < 1):
        raise ValueError("q must lie in (0,1)")
    n = len(x)
    if n < 3:
        return float(np.quantile(x, q))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(bootstrap), n))
    boot_means = x[idx].mean(axis=1)
    theta = float(x.mean())
    # Bias correction. Half-weight ties avoid infinite z0 in discrete cases.
    prop = (np.sum(boot_means < theta) + 0.5 * np.sum(boot_means == theta)) / len(boot_means)
    eps = 0.5 / len(boot_means)
    prop = float(np.clip(prop, eps, 1 - eps))
    z0 = float(norm.ppf(prop))
    # Jackknife acceleration for the mean.
    jk = np.asarray([(x.sum() - x[i]) / (n - 1) for i in range(n)], float)
    jbar = float(jk.mean())
    num = float(np.sum((jbar - jk) ** 3))
    den = float(6.0 * (np.sum((jbar - jk) ** 2) ** 1.5))
    acc = 0.0 if den <= 1e-30 else num / den
    zq = float(norm.ppf(q))
    denom = 1.0 - acc * (z0 + zq)
    adj_z = z0 + (z0 + zq) / (denom if abs(denom) > 1e-12 else np.sign(denom) * 1e-12)
    adj_q = float(np.clip(norm.cdf(adj_z), 0.0, 1.0))
    return float(np.quantile(boot_means, adj_q))


def movement_margin_evidence(
    numerator_rows,
    denominator_rows,
    patients,
    *,
    movement_margin: float = 0.01,
    alpha: float = 0.05,
    method: str = "t",
    bootstrap: int = 5000,
    seed: int = 0,
    denominator_floor: float = 1e-8,
) -> MovementMarginEvidence:
    """Inference for Movement via the margin contrast A - delta_M B.

    ``method`` may be ``t``, ``percentile`` or ``bca``.  Percentile and BCa
    resample *patient-level numerator and denominator jointly* through their
    already-collapsed contrast, so their covariance is preserved.
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must lie in (0,1)")
    a, b, g = movement_patient_components(
        numerator_rows, denominator_rows, patients, movement_margin=movement_margin
    )
    A = float(a.mean())
    B = float(b.mean())
    G = float(g.mean())
    if method == "t":
        lo = _one_sided_lcb_t(g, alpha)
        hi = _one_sided_ucb_t(g, alpha)
    else:
        if int(bootstrap) < 100:
            raise ValueError("bootstrap must be >=100")
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(g), size=(int(bootstrap), len(g)))
        gb = g[idx].mean(axis=1)
        if method == "percentile":
            lo = float(np.quantile(gb, alpha))
            hi = float(np.quantile(gb, 1 - alpha))
        elif method == "bca":
            lo = _bca_quantile_mean(g, alpha, bootstrap=int(bootstrap), seed=seed)
            hi = _bca_quantile_mean(g, 1 - alpha, bootstrap=int(bootstrap), seed=seed + 1)
        else:
            raise ValueError("method must be one of: t, percentile, bca")
    identifiable = bool(B > float(denominator_floor))
    ratio = float(A / B) if identifiable else float("nan")
    return MovementMarginEvidence(
        ratio=ratio,
        numerator=A,
        denominator=B,
        contrast=G,
        contrast_lcb=float(lo),
        contrast_ucb=float(hi),
        method=method,
        alpha=float(alpha),
        denominator_floor=float(denominator_floor),
        ratio_identifiable=identifiable,
        n_patients=len(g),
    )


def hoeffding_lcb(x, *, lower: float, upper: float, alpha: float = 0.05) -> float:
    """Distribution-free one-sided Hoeffding lower bound for an iid bounded mean."""
    x = _finite_vector(x, "x")
    if not (0 < alpha < 1):
        raise ValueError("alpha must lie in (0,1)")
    if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
        raise ValueError("finite support with lower < upper is required")
    if (x < lower - 1e-12).any() or (x > upper + 1e-12).any():
        raise ValueError("observations fall outside declared support")
    radius = (upper - lower) * math.sqrt(math.log(1.0 / alpha) / (2.0 * len(x)))
    return float(x.mean() - radius)


def honest_confirmation_iut(
    movement_contrast,
    utility,
    retention,
    *,
    movement_margin: float = 0.01,
    alpha: float = 0.05,
) -> HonestConfirmationResult:
    """Population-level finite-sample IUT using universal deterministic support.

    This backward-compatible entry point targets the unconditional mean over
    iid confirmation patients. Candidate/reference/deployment-retention rules
    must be frozen using data independent of confirmation outcomes. Normalized
    Hellinger gives G_M in [-delta_M,1]. The universal MAE-difference support
    [-1,1] is valid for simplex compositions with at least two states. For the
    sharper K-state bound [-2/K,2/K], use
    :func:`honest_confirmation_iut_population`.
    """
    gm = _finite_vector(movement_contrast, "movement_contrast")
    u = _finite_vector(utility, "utility")
    r = _finite_vector(retention, "retention")
    if not (len(gm) == len(u) == len(r)):
        raise ValueError("all confirmation axes must use the same patients")
    ml = hoeffding_lcb(gm, lower=-float(movement_margin), upper=1.0, alpha=alpha)
    ul = hoeffding_lcb(u, lower=-1.0, upper=1.0, alpha=alpha)
    rl = hoeffding_lcb(r, lower=-1.0, upper=1.0, alpha=alpha)
    me = HonestAxisEvidence(float(gm.mean()), ml, -float(movement_margin), 1.0, alpha, len(gm))
    ue = HonestAxisEvidence(float(u.mean()), ul, -1.0, 1.0, alpha, len(u))
    re = HonestAxisEvidence(float(r.mean()), rl, -1.0, 1.0, alpha, len(r))
    return HonestConfirmationResult(
        me, ue, re, bool(me.supported and ue.supported and re.supported), float(alpha),
        mode="honest_confirmation_population_universal", estimand_scope="population"
    )


def leave_one_patient_stability(values, *, selection_gap: float | None = None) -> StabilityDiagnostic:
    """A transparent sensitivity diagnostic for a patient-level mean.

    This is not a proof of algorithmic stability; it quantifies the realized
    sensitivity of an already-computed patient-level estimand to deleting one
    independent patient.
    """
    x = _finite_vector(values, "values")
    full = float(x.mean())
    loo = np.asarray([(x.sum() - x[i]) / (len(x) - 1) for i in range(len(x))], float)
    shift = loo - full
    return StabilityDiagnostic(
        full_estimate=full,
        leave_one_patient_out_estimates=tuple(float(v) for v in loo),
        max_abs_shift=float(np.max(np.abs(shift))),
        rms_shift=float(np.sqrt(np.mean(shift ** 2))),
        selection_gap=None if selection_gap is None else float(selection_gap),
    )


def nearest_grid_risk_regret_bound(*, grid_step: float, n_states: int) -> float:
    """Worst-case MAE regret from rounding lambda to the nearest uniform grid point.

    For simplex predictions, ||p_c-p_r||_1 <= 2 and mean absolute composition
    loss is K^{-1}||p-y||_1.  Nearest-grid rounding is at most h/2 in lambda,
    yielding regret at most h/K.
    """
    if not math.isfinite(float(grid_step)) or not (0 < grid_step <= 1):
        raise ValueError("grid_step must lie in (0,1]")
    if int(n_states) < 1:
        raise ValueError("n_states must be positive")
    return float(grid_step) / int(n_states)


def heterogeneous_hoeffding_lcb(values, lower_bounds, upper_bounds, *, alpha: float = 0.05) -> float:
    """One-sided Hoeffding LCB for independent, non-identically bounded means.

    If X_i in [a_i,b_i] independently, then

        P(mean(X)-E mean(X) >= t) <= exp(-2 n^2 t^2 / sum_i (b_i-a_i)^2).

    The function is conditional on whatever frozen covariates/predictions make
    the supplied bounds deterministic.
    """
    x = _finite_vector(values, "values")
    lo = np.asarray(lower_bounds, float).reshape(-1)
    hi = np.asarray(upper_bounds, float).reshape(-1)
    if len(lo) != len(x) or len(hi) != len(x):
        raise ValueError("bound arrays must match values")
    if not np.isfinite(lo).all() or not np.isfinite(hi).all() or (hi <= lo).any():
        raise ValueError("every observation needs finite lower < upper bounds")
    if (x < lo - 1e-10).any() or (x > hi + 1e-10).any():
        raise ValueError("observed values violate supplied bounds")
    if not (0 < alpha < 1):
        raise ValueError("alpha must lie in (0,1)")
    widths = hi - lo
    radius = math.sqrt(float(np.sum(widths * widths)) * math.log(1.0 / alpha) / (2.0 * len(x) ** 2))
    return float(x.mean() - radius)


def heterogeneous_hoeffding_pvalue(values, lower_bounds, upper_bounds) -> float:
    """Valid one-sided p-value for H0: E[mean(X)] <= 0.

    For independent X_i in deterministic intervals [a_i,b_i], Hoeffding gives
    P(mean(X) >= t) <= exp(-2 n^2 t^2 / sum_i (b_i-a_i)^2)
    under any null distribution with E[mean(X)] <= 0.  The returned p-value
    is therefore finite-sample valid under the same independence and frozen-
    contract conditions as ``heterogeneous_hoeffding_lcb``.
    """
    x = _finite_vector(values, "values")
    lo = np.asarray(lower_bounds, float).reshape(-1)
    hi = np.asarray(upper_bounds, float).reshape(-1)
    if len(lo) != len(x) or len(hi) != len(x):
        raise ValueError("bound arrays must match values")
    if not np.isfinite(lo).all() or not np.isfinite(hi).all() or (hi < lo).any():
        raise ValueError("every observation needs finite lower <= upper bounds")
    if (x < lo - 1e-10).any() or (x > hi + 1e-10).any():
        raise ValueError("observed values violate supplied bounds")
    t = float(x.mean())
    if t <= 0.0:
        return 1.0
    widths = hi - lo
    denom = float(np.sum(widths * widths))
    if denom <= 0.0:
        # A strictly positive deterministic contrast cannot occur under a
        # null whose conditional mean is <=0.
        return 0.0
    expo = -2.0 * len(x) ** 2 * t * t / denom
    return float(min(1.0, math.exp(expo)))


def honest_confirmation_pvalues_localized(
    movement_contrast,
    movement_numerator,
    utility,
    utility_abs_bound,
    retention,
    retention_abs_bound,
    *,
    movement_margin: float = 0.01,
) -> dict:
    """Finite-sample component and candidate-IUT p-values for confirmation.

    The candidate-level IUT p-value is max(p_M, p_U, p_N), because
    qualification requires all three necessary alternatives simultaneously.
    Holm correction, when several candidate predictors are tested, is then
    applied across these candidate-level p-values rather than across 3C
    component tests.
    """
    gm = _finite_vector(movement_contrast, "movement_contrast")
    a = _finite_vector(movement_numerator, "movement_numerator")
    u = _finite_vector(utility, "utility")
    cu = _finite_vector(utility_abs_bound, "utility_abs_bound")
    r = _finite_vector(retention, "retention")
    cr = _finite_vector(retention_abs_bound, "retention_abs_bound")
    n = len(gm)
    if not all(len(z) == n for z in (a, u, cu, r, cr)):
        raise ValueError("all patient-level arrays must have the same length")
    if (cu < 0).any() or (cr < 0).any():
        raise ValueError("absolute bounds must be non-negative")
    p_m = heterogeneous_hoeffding_pvalue(gm, a-float(movement_margin), a)
    p_u = heterogeneous_hoeffding_pvalue(u, -cu, cu)
    p_n = heterogeneous_hoeffding_pvalue(r, -cr, cr)
    return {
        "movement": float(p_m),
        "utility": float(p_u),
        "retention": float(p_n),
        "candidate_iut": float(max(p_m, p_u, p_n)),
    }


def honest_confirmation_iut_cohort_conditional(
    movement_contrast,
    movement_numerator,
    utility,
    utility_abs_bound,
    retention,
    retention_abs_bound,
    *,
    movement_margin: float = 0.01,
    alpha: float = 0.05,
) -> HonestConfirmationResult:
    """Cohort-conditional finite-sample IUT using localized envelopes.

    The estimand is the average conditional mean over the realized target-blind
    confirmation covariates, n^{-1} sum_i E[score_i | D, X_i], not the
    unconditional population mean E[score]. Conditional on the frozen
    development analysis D and the realized confirmation inputs/predictions:

    * Movement: G_i=A_i-delta B_i with B_i in [0,1], hence
      G_i in [A_i-delta, A_i].  Its interval width is delta, not 1+delta.
    * Utility: reverse triangle inequality gives
      |L_i(ref)-L_i(cand)| <= C_Ui, where C_Ui is patient-average MAE between
      the two fixed predictions.
    * Retention has the analogous bound C_Ni between reference and retained
      predictions.

    Hoeffding is then applied conditionally to independent confirmation targets
    with these heterogeneous deterministic ranges. No distributional shape or
    asymptotic approximation is required. This localized bound must not be
    interpreted as an unconditional population-mean guarantee unless additional
    arguments integrate over X with valid global support.
    """
    gm = _finite_vector(movement_contrast, "movement_contrast")
    a = _finite_vector(movement_numerator, "movement_numerator")
    u = _finite_vector(utility, "utility")
    cu = _finite_vector(utility_abs_bound, "utility_abs_bound")
    r = _finite_vector(retention, "retention")
    cr = _finite_vector(retention_abs_bound, "retention_abs_bound")
    n = len(gm)
    if not all(len(z) == n for z in (a,u,cu,r,cr)):
        raise ValueError("all patient-level arrays must have the same length")
    if (cu < 0).any() or (cr < 0).any():
        raise ValueError("absolute bounds must be non-negative")
    # Zero-width bounds are valid for deterministic zero effects; enlarge by a
    # floating epsilon solely for numerical validation, without changing the
    # theoretical radius materially.
    eps = 1e-15
    ml = heterogeneous_hoeffding_lcb(gm, a-float(movement_margin), a+eps, alpha=alpha)
    ul = heterogeneous_hoeffding_lcb(u, -cu-eps, cu+eps, alpha=alpha)
    rl = heterogeneous_hoeffding_lcb(r, -cr-eps, cr+eps, alpha=alpha)
    me = HonestAxisEvidence(float(gm.mean()), ml, float(np.min(a-movement_margin)), float(np.max(a)), alpha, n)
    ue = HonestAxisEvidence(float(u.mean()), ul, float(np.min(-cu)), float(np.max(cu)), alpha, n)
    re = HonestAxisEvidence(float(r.mean()), rl, float(np.min(-cr)), float(np.max(cr)), alpha, n)
    return HonestConfirmationResult(
        me, ue, re, bool(me.supported and ue.supported and re.supported), float(alpha),
        mode="honest_confirmation_cohort_conditional", estimand_scope="cohort_conditional"
    )


def honest_confirmation_iut_localized(*args, **kwargs) -> HonestConfirmationResult:
    """Backward-compatible alias for cohort-conditional Honest Confirmation."""
    return honest_confirmation_iut_cohort_conditional(*args, **kwargs)


def honest_confirmation_iut_population(
    movement_contrast, utility, deployment_retention, *, n_states: int,
    movement_margin: float = 0.01, alpha: float = 0.05,
) -> HonestConfirmationResult:
    """Unconditional population Honest Confirmation with K-state global support.

    For iid confirmation patients from the target population, normalized
    Hellinger gives G_M in [-delta_M,1]. For K-state simplex MAE, each loss is
    in [0,2/K], hence both Utility and development-selected Deployment
    Retention risk differences lie in [-2/K,2/K]. These deterministic bounds
    do not depend on confirmation covariates and therefore target the
    unconditional population means.
    """
    k = int(n_states)
    if k < 2:
        raise ValueError("n_states must be at least 2 for composition inference")
    gm = _finite_vector(movement_contrast, "movement_contrast")
    u = _finite_vector(utility, "utility")
    n_d = _finite_vector(deployment_retention, "deployment_retention")
    if not (len(gm) == len(u) == len(n_d)):
        raise ValueError("all confirmation axes must use the same patients")
    max_mae = 2.0 / k
    ml = hoeffding_lcb(gm, lower=-float(movement_margin), upper=1.0, alpha=alpha)
    ul = hoeffding_lcb(u, lower=-max_mae, upper=max_mae, alpha=alpha)
    nl = hoeffding_lcb(n_d, lower=-max_mae, upper=max_mae, alpha=alpha)
    me = HonestAxisEvidence(float(gm.mean()), ml, -float(movement_margin), 1.0, alpha, len(gm))
    ue = HonestAxisEvidence(float(u.mean()), ul, -max_mae, max_mae, alpha, len(u))
    ne = HonestAxisEvidence(float(n_d.mean()), nl, -max_mae, max_mae, alpha, len(n_d))
    return HonestConfirmationResult(
        me, ue, ne, bool(me.supported and ue.supported and ne.supported), float(alpha),
        mode="honest_confirmation_population", estimand_scope="population"
    )


def honest_confirmation_pvalues_population(
    movement_contrast, utility, deployment_retention, *, n_states: int, movement_margin: float = 0.01
) -> dict:
    """Population component/IUT p-values from deterministic global support."""
    k = int(n_states)
    if k < 2:
        raise ValueError("n_states must be at least 2 for composition inference")
    gm = _finite_vector(movement_contrast, "movement_contrast")
    u = _finite_vector(utility, "utility")
    n_d = _finite_vector(deployment_retention, "deployment_retention")
    if not (len(gm) == len(u) == len(n_d)):
        raise ValueError("all confirmation axes must use the same patients")
    max_mae = 2.0 / k
    p_m = heterogeneous_hoeffding_pvalue(
        gm, np.full(len(gm), -float(movement_margin)), np.full(len(gm), 1.0)
    )
    p_u = heterogeneous_hoeffding_pvalue(
        u, np.full(len(u), -max_mae), np.full(len(u), max_mae)
    )
    p_n = heterogeneous_hoeffding_pvalue(
        n_d, np.full(len(n_d), -max_mae), np.full(len(n_d), max_mae)
    )
    return {
        "movement": float(p_m), "utility": float(p_u), "retention": float(p_n),
        "candidate_iut": float(max(p_m, p_u, p_n)), "estimand_scope": "population"
    }


def _validate_simplex_rows(z: np.ndarray, name: str, tol: float = 1e-8) -> None:
    if z.ndim != 2 or z.shape[1] < 2:
        raise ValueError(f"{name} must be a rows x states matrix with at least two states")
    if (z < -tol).any() or (z > 1.0 + tol).any():
        raise ValueError(f"{name} contains values outside the probability simplex")
    if not np.allclose(z.sum(axis=1), 1.0, atol=tol, rtol=0.0):
        raise ValueError(f"{name} rows must sum to one for population MAE support")


def honest_confirm_predictions(
    target,
    source,
    reference,
    candidate,
    retained,
    patients,
    *,
    movement_margin: float = 0.01,
    alpha: float = 0.05,
    scope: str = "cohort_conditional",
    return_both: bool = False,
):
    """End-to-end Honest Confirmation from frozen prediction arrays.

    ``scope='cohort_conditional'`` conditions on the realized target-blind
    confirmation inputs/predictions and uses prediction-localized envelopes.
    ``scope='population'`` targets the unconditional iid patient population and
    uses deterministic global simplex/normalized-Hellinger support.

    ``return_both=True`` returns both valid analyses without using confirmation
    outcomes for any model, reference, weight or scope selection. The retained
    array is interpreted as the predictor induced by a Deployment Retention
    weight selected on development data only.
    """
    from .model import hellinger, mae
    y = np.asarray(target, float)
    s0 = np.asarray(source, float)
    r0 = np.asarray(reference, float)
    c = np.asarray(candidate, float)
    rt = np.asarray(retained, float)
    p = np.asarray(patients).astype(str)
    if not (
        y.ndim == 2 and s0.shape == y.shape and r0.shape == y.shape
        and c.shape == y.shape and rt.shape == y.shape and len(p) == len(y)
    ):
        raise ValueError("all prediction/target arrays must have matching rows x states shape")
    if not all(np.isfinite(z).all() for z in (y, s0, r0, c, rt)):
        raise ValueError("prediction/target arrays must be finite")

    # Row contributions and prediction-only localized envelopes.
    arow = hellinger(c, s0)
    brow = hellinger(y, s0)
    grow = arow - float(movement_margin) * brow
    urow = mae(y, r0) - mae(y, c)
    ndrow = mae(y, r0) - mae(y, rt)
    curow = mae(r0, c)
    cndrow = mae(r0, rt)
    _, a = patient_equal_values(arow, p)
    _, g = patient_equal_values(grow, p)
    _, u = patient_equal_values(urow, p)
    _, n_d = patient_equal_values(ndrow, p)
    _, cu = patient_equal_values(curow, p)
    _, cnd = patient_equal_values(cndrow, p)

    cohort = honest_confirmation_iut_cohort_conditional(
        g, a, u, cu, n_d, cnd, movement_margin=movement_margin, alpha=alpha
    )

    # Population support requires legal simplex-valued target/predictions.
    for arr, name in ((y, "target"), (s0, "source"), (r0, "reference"),
                      (c, "candidate"), (rt, "retained")):
        _validate_simplex_rows(arr, name)
    population = honest_confirmation_iut_population(
        g, u, n_d, n_states=y.shape[1], movement_margin=movement_margin, alpha=alpha
    )

    detail = {
        "movement_numerator": a,
        "movement_contrast": g,
        "utility": u,
        "retention": n_d,
        "deployment_retention": n_d,
        "utility_abs_bound": cu,
        "retention_abs_bound": cnd,
        "deployment_retention_abs_bound": cnd,
        "n_states": int(y.shape[1]),
    }
    if return_both:
        return {"cohort_conditional": cohort, "population": population}, detail
    scope_norm = str(scope).strip().lower().replace("-", "_")
    if scope_norm in {"cohort_conditional", "conditional", "localized"}:
        return cohort, detail
    if scope_norm in {"population", "unconditional"}:
        return population, detail
    raise ValueError("scope must be 'cohort_conditional' or 'population'")

