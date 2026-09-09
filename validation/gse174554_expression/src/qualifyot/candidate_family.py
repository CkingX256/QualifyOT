from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.stats import t as student_t


@dataclass(frozen=True)
class CandidateFamilyResult:
    candidate: str
    component_p: dict
    iut_p: float
    holm_threshold: float | None
    holm_rejected: bool
    rank: int

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate,
            "component_p": dict(self.component_p),
            "iut_p": float(self.iut_p),
            "holm_threshold": None if self.holm_threshold is None else float(self.holm_threshold),
            "holm_rejected": bool(self.holm_rejected),
            "rank": int(self.rank),
        }


def one_sided_t_pvalue(x, margin: float = 0.0) -> float:
    """One-sided p-value for H0: E[x] <= margin versus H1: E[x] > margin.

    This is an inferential building block, not a claim of exact finite-sample
    validity under arbitrary dependent-LOPO learning.  The caller must ensure
    that ``x`` contains one value per physical patient and should calibrate the
    procedure for the learning design in use.
    """
    x = np.asarray(x, float).reshape(-1)
    if x.size < 2:
        raise ValueError("at least two independent patient values are required")
    if not np.isfinite(x).all() or not math.isfinite(float(margin)):
        raise ValueError("values and margin must be finite")
    y = x - float(margin)
    mu = float(y.mean())
    sd = float(y.std(ddof=1))
    if sd <= 1e-15:
        return 0.0 if mu > 0 else 1.0
    tstat = mu / (sd / math.sqrt(len(y)))
    return float(student_t.sf(tstat, df=len(y) - 1))


def iut_pvalue(component_p) -> float:
    """Valid IUT p-value for a conjunction of necessary component claims.

    For H0 = union_j H0j, max_j p_j is a valid p-value whenever each p_j is
    valid for its component null.  This is the p-value analogue of the usual
    intersection-union decision and does not introduce a Bonferroni split
    across necessary components.
    """
    p = np.asarray(list(component_p), float)
    if p.size == 0 or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("component p-values must be a non-empty finite set in [0,1]")
    return float(p.max())


def holm_rejections(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, tuple[bool, int, float]]:
    """Holm step-down FWER control over candidate-level IUT p-values.

    The multiple-testing family is the set of *candidate qualification claims*,
    not the necessary axes inside one candidate.  This preserves the power
    advantage of IUT within candidate while controlling post-search candidate
    multiplicity with Holm under arbitrary dependence, conditional on valid
    candidate-level p-values.
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must lie in (0,1)")
    items = [(str(k), float(v)) for k, v in pvalues.items()]
    if not items:
        raise ValueError("at least one candidate p-value is required")
    if any((not math.isfinite(p)) or p < 0 or p > 1 for _, p in items):
        raise ValueError("candidate p-values must lie in [0,1]")
    ordered = sorted(items, key=lambda z: (z[1], z[0]))
    m = len(ordered)
    out: dict[str, tuple[bool, int, float]] = {}
    still_rejecting = True
    for r, (name, p) in enumerate(ordered, start=1):
        thr = alpha / (m - r + 1)
        reject = bool(still_rejecting and p <= thr)
        if not reject:
            still_rejecting = False
        out[name] = (reject, r, thr)
    return out


def familywise_iut_t(
    effects: dict[str, dict[str, np.ndarray]],
    *,
    alpha: float = 0.05,
    component_margins: dict[str, float] | None = None,
) -> list[CandidateFamilyResult]:
    """Search-aware qualification for a frozen finite candidate family.

    ``effects[candidate][axis]`` must contain exactly one effect contribution
    per physical patient.  Expected signs are positive.  Recommended axes are
    ``Movement``, ``Utility`` and ``Retention`` where Movement has already been
    transformed to an additive margin contribution
    ``d(candidate,source) - delta_M*d(target,source)``.

    Each candidate receives an IUT p-value ``max_j p_cj``.  Holm is then
    applied *across candidates*.  This avoids the unnecessary and often severe
    penalty of treating all candidate-by-axis statements as separate target
    discoveries while still auditing candidate shopping.
    """
    if not effects:
        raise ValueError("effects must contain at least one candidate")
    margins = {} if component_margins is None else dict(component_margins)
    iut = {}
    comps = {}
    n_axes = None
    axis_names = None
    for cand, axes in effects.items():
        if not axes:
            raise ValueError(f"{cand}: no component effects")
        if axis_names is None:
            axis_names = tuple(axes.keys())
            n_axes = len(axis_names)
        elif tuple(axes.keys()) != axis_names:
            raise ValueError("all candidates must provide the same ordered axes")
        cp = {a: one_sided_t_pvalue(x, margin=float(margins.get(a, 0.0))) for a, x in axes.items()}
        comps[str(cand)] = cp
        iut[str(cand)] = iut_pvalue(cp.values())
    holm = holm_rejections(iut, alpha=alpha)
    ans=[]
    for cand in sorted(iut, key=lambda c: (holm[c][1], c)):
        rej, rank, thr = holm[cand]
        ans.append(CandidateFamilyResult(cand, comps[cand], iut[cand], thr, rej, rank))
    return ans
