from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GraphConfidenceSet:
    members: tuple[str, ...]
    excluded: tuple[str, ...]
    pairwise: pd.DataFrame
    critical_value: float
    alpha: float
    core_edges: tuple[tuple[str, str], ...] = ()
    optional_edges: tuple[tuple[str, str], ...] = ()
    excluded_edges: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict:
        return {
            "members": list(self.members),
            "excluded": list(self.excluded),
            "critical_value": self.critical_value,
            "alpha": self.alpha,
            "core_edges": [list(e) for e in self.core_edges],
            "optional_edges": [list(e) for e in self.optional_edges],
            "excluded_edges": [list(e) for e in self.excluded_edges],
        }


def _patient_mean_matrix(losses, patients) -> tuple[np.ndarray, np.ndarray]:
    losses = np.asarray(losses, float)
    patients = np.asarray(patients).astype(str)
    if losses.ndim != 2:
        raise ValueError("losses must be rows x models")
    if len(patients) != losses.shape[0]:
        raise ValueError("patients length must match loss rows")
    up = np.unique(patients)
    out = np.vstack([losses[patients == p].mean(axis=0) for p in up])
    return up, out


def simultaneous_pairwise_intervals(
    losses,
    patients,
    model_names: Iterable[str],
    *,
    alpha: float = 0.05,
    bootstrap: int = 5000,
    seed: int = 0,
) -> tuple[pd.DataFrame, float]:
    """Selection-aware simultaneous pairwise loss-difference intervals.

    For every pair (a, b), the estimand is E[L_a - L_b]. A single max-|t|
    critical value is calibrated over *all* pairwise contrasts. Because the
    intervals are simultaneous over the full pair family, subsequent use of
    the empirically best model does not require a second unadjusted test.
    """

    names = tuple(str(x) for x in model_names)
    _, X = _patient_mean_matrix(losses, patients)
    n, m = X.shape
    if m != len(names):
        raise ValueError("number of model_names must match loss columns")
    if m < 2:
        raise ValueError("at least two models are required")
    if n < 2:
        raise ValueError("at least two patients are required")

    pairs = list(combinations(range(m), 2))
    D = np.column_stack([X[:, i] - X[:, j] for i, j in pairs])
    mean = D.mean(axis=0)
    se = D.std(axis=0, ddof=1) / np.sqrt(n)
    valid = se > 1e-14

    rng = np.random.default_rng(seed)
    max_t = np.zeros(int(bootstrap), float)
    for b in range(int(bootstrap)):
        idx = rng.integers(0, n, size=n)
        mb = D[idx].mean(axis=0)
        t = np.zeros(len(pairs), float)
        t[valid] = np.abs((mb[valid] - mean[valid]) / se[valid])
        max_t[b] = t.max(initial=0.0)
    q = float(np.quantile(max_t, 1 - alpha))
    lo = mean - q * se
    hi = mean + q * se

    rows = []
    for k, (i, j) in enumerate(pairs):
        rows.append(
            {
                "model_a": names[i],
                "model_b": names[j],
                "mean_loss_a_minus_b": float(mean[k]),
                "se": float(se[k]),
                "sim_lo": float(lo[k]),
                "sim_hi": float(hi[k]),
                "a_worse_than_b": bool(lo[k] > 0),
                "b_worse_than_a": bool(hi[k] < 0),
            }
        )
    return pd.DataFrame(rows), q


def graph_confidence_set(
    losses,
    patients,
    model_names: Iterable[str],
    *,
    graph_edges: Mapping[str, Iterable[tuple[str, str]]] | None = None,
    alpha: float = 0.05,
    bootstrap: int = 5000,
    seed: int = 0,
) -> GraphConfidenceSet:
    """Return graphs not simultaneously shown to be worse than a competitor.

    A model is excluded if any other model has significantly smaller risk under
    the simultaneous pairwise family. The retained set is therefore a
    partial-identification set rather than a forced single winner.
    """

    names = tuple(str(x) for x in model_names)
    pairwise, q = simultaneous_pairwise_intervals(
        losses, patients, names, alpha=alpha, bootstrap=bootstrap, seed=seed
    )
    excluded = set()
    for r in pairwise.itertuples(index=False):
        if r.a_worse_than_b:
            excluded.add(r.model_a)
        if r.b_worse_than_a:
            excluded.add(r.model_b)
    members = tuple(n for n in names if n not in excluded)

    core_edges: tuple[tuple[str, str], ...] = ()
    optional_edges: tuple[tuple[str, str], ...] = ()
    excluded_edges: tuple[tuple[str, str], ...] = ()
    if graph_edges is not None and members:
        edge_sets = {k: set(tuple(e) for e in graph_edges[k]) for k in names}
        member_sets = [edge_sets[k] for k in members]
        core = set.intersection(*member_sets) if member_sets else set()
        union = set.union(*member_sets) if member_sets else set()
        all_edges = set.union(*(edge_sets[k] for k in names))
        core_edges = tuple(sorted(core))
        optional_edges = tuple(sorted(union - core))
        excluded_edges = tuple(sorted(all_edges - union))

    return GraphConfidenceSet(
        members=members,
        excluded=tuple(n for n in names if n in excluded),
        pairwise=pairwise,
        critical_value=q,
        alpha=float(alpha),
        core_edges=core_edges,
        optional_edges=optional_edges,
        excluded_edges=excluded_edges,
    )
