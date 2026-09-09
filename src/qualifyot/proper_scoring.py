from __future__ import annotations

"""Proper scoring-rule sensitivity for compositional predictions.

QualifyOT's locked primary loss remains patient-equal MAE for backward
compatibility.  This module supplies a log-score sensitivity that is natural
for probability-simplex predictions.  Because the pipeline already produces
actual held-out-patient predictions, no PSIS approximation is needed merely to
obtain a leave-one-patient-out predictive score.
"""

from dataclasses import asdict, dataclass

import numpy as np

from .bayesian_evidence import bayesian_bootstrap_mean
from .inference import mean_interval


def compositional_log_score(target, prediction, *, eps: float = 1e-12) -> np.ndarray:
    """Per-row normalized multinomial log score, higher is better.

    The target is a composition rather than a count vector, so each row is
    normalized per cell/observation.  This prevents deeper cell sequencing from
    silently giving a patient more inferential weight.
    """
    y=np.asarray(target,float); p=np.asarray(prediction,float)
    if y.shape!=p.shape or y.ndim!=2:
        raise ValueError("target and prediction must have the same rows x states shape")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("target and prediction must be finite")
    if (y < -1e-10).any() or (p < -1e-10).any():
        raise ValueError("compositions must be non-negative")
    if not np.allclose(y.sum(1),1,atol=1e-6) or not np.allclose(p.sum(1),1,atol=1e-6):
        raise ValueError("target and prediction must lie on the simplex")
    q=np.clip(p,float(eps),1.0); q=q/q.sum(1,keepdims=True)
    return np.sum(y*np.log(q),axis=1)


def patient_equal_contrast(values, patients) -> np.ndarray:
    v=np.asarray(values,float); p=np.asarray(patients).astype(str)
    if len(v)!=len(p) or v.ndim!=1: raise ValueError("values/patients mismatch")
    up=np.unique(p)
    return np.asarray([v[p==x].mean() for x in up],float)


@dataclass(frozen=True)
class LogScoreSensitivity:
    n_patients: int
    mean_candidate_minus_reference: float
    interval_95: tuple[float,float]
    bayesian_bootstrap_probability_positive: float
    method: str = "patient-equal normalized multinomial log score"

    def to_dict(self): return asdict(self)


def log_score_sensitivity(target, reference, candidate, patients, *, bootstrap=5000, seed=0) -> LogScoreSensitivity:
    d=compositional_log_score(target,candidate)-compositional_log_score(target,reference)
    x=patient_equal_contrast(d,patients)
    ci=mean_interval(x,method="bootstrap_t",bootstrap=max(100,int(bootstrap)),seed=seed)
    bb=bayesian_bootstrap_mean(x,draws=max(1000,int(bootstrap)),seed=seed+1)
    return LogScoreSensitivity(len(x),float(x.mean()),(float(ci.lower),float(ci.upper)),float(bb.probability_above_margin))
