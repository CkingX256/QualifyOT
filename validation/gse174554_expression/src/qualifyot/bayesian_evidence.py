from __future__ import annotations

"""Lightweight Bayesian-bootstrap evidence profiles.

This module deliberately does *not* label its outputs Bayes factors or causal
edge posterior probabilities.  The Rubin Bayesian bootstrap places a
Dirichlet(1,...,1) posterior over the empirical distribution of independent
patient-level contributions.  It is useful as an orthogonal small-n evidence
profile while preserving the physical patient as the inferential unit.

For graph libraries, draw-wise best-risk frequencies are reported as
*predictive model-selection probabilities under the Bayesian bootstrap*.
They depend on the frozen graph library, loss and tie rule; they are not
posterior probabilities that an edge is biologically causal.
"""

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BayesianBootstrapMean:
    n_patients: int
    estimate: float
    posterior_mean: float
    posterior_median: float
    credible_interval: tuple[float, float]
    probability_above_margin: float
    margin: float
    draws: int
    seed: int

    def to_dict(self) -> dict:
        return asdict(self)


def _finite_vector(x, *, name: str = "values") -> np.ndarray:
    x = np.asarray(x, float).reshape(-1)
    if x.size < 2:
        raise ValueError(f"{name} requires at least two independent patient values")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} must be finite")
    return x


def bayesian_bootstrap_mean(
    patient_values,
    *,
    margin: float = 0.0,
    credible: float = 0.95,
    draws: int = 20_000,
    seed: int = 0,
) -> BayesianBootstrapMean:
    """Bayesian-bootstrap posterior profile for a patient-level mean.

    ``patient_values`` must already contain one equally weighted value per
    physical patient.  The posterior is conditional on these patient-level
    contributions; if the upstream prediction pipeline itself is to be
    treated as random, full-pipeline resampling remains a separate analysis.
    """
    x = _finite_vector(patient_values, name="patient_values")
    if not (0 < credible < 1):
        raise ValueError("credible must lie in (0,1)")
    if int(draws) < 100:
        raise ValueError("draws must be >=100")
    if not np.isfinite(float(margin)):
        raise ValueError("margin must be finite")
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.ones(len(x)), size=int(draws))
    theta = w @ x
    a = (1.0 - credible) / 2.0
    lo, hi = np.quantile(theta, [a, 1.0 - a])
    return BayesianBootstrapMean(
        n_patients=int(len(x)),
        estimate=float(x.mean()),
        posterior_mean=float(theta.mean()),
        posterior_median=float(np.median(theta)),
        credible_interval=(float(lo), float(hi)),
        probability_above_margin=float(np.mean(theta > float(margin))),
        margin=float(margin),
        draws=int(draws),
        seed=int(seed),
    )


@dataclass(frozen=True)
class BayesianGraphProfile:
    model_table: pd.DataFrame
    edge_table: pd.DataFrame
    draws: int
    seed: int
    tie_tolerance: float

    def to_dict(self) -> dict:
        return {
            "models": self.model_table.to_dict(orient="records"),
            "edges": self.edge_table.to_dict(orient="records"),
            "draws": self.draws,
            "seed": self.seed,
            "tie_tolerance": self.tie_tolerance,
        }


def _patient_mean_matrix(losses, patients) -> tuple[np.ndarray, np.ndarray]:
    losses = np.asarray(losses, float)
    patients = np.asarray(patients).astype(str)
    if losses.ndim != 2:
        raise ValueError("losses must have shape rows x models")
    if len(patients) != losses.shape[0]:
        raise ValueError("patients must match loss rows")
    if losses.shape[1] < 2:
        raise ValueError("at least two models are required")
    if not np.isfinite(losses).all():
        raise ValueError("losses must be finite")
    up = np.unique(patients)
    if len(up) < 2:
        raise ValueError("at least two physical patients are required")
    X = np.vstack([losses[patients == p].mean(axis=0) for p in up])
    return up, X


def bayesian_bootstrap_graph_profile(
    losses,
    patients,
    model_names: Iterable[str],
    *,
    graph_edges: Mapping[str, Iterable[tuple[str, str]]] | None = None,
    draws: int = 20_000,
    seed: int = 0,
    tie_tolerance: float = 1e-12,
) -> BayesianGraphProfile:
    """Predictive graph-library uncertainty under a Bayesian bootstrap.

    Each draw places Dirichlet weights over physical patients, calculates the
    patient-weighted predictive risk of every *pre-specified* graph and assigns
    equal probability mass to all models tied for minimum risk within
    ``tie_tolerance``.  Model frequencies therefore quantify uncertainty in
    which library member has the lowest patient-average predictive loss.

    If ``graph_edges`` is supplied, edge inclusion scores average the draw-wise
    best-model mass over graphs containing each edge.  These are predictive
    library scores, not causal edge posterior probabilities.
    """
    names = tuple(str(x) for x in model_names)
    _, X = _patient_mean_matrix(losses, patients)
    if X.shape[1] != len(names):
        raise ValueError("model_names length must match loss columns")
    if int(draws) < 100:
        raise ValueError("draws must be >=100")
    if tie_tolerance < 0 or not np.isfinite(float(tie_tolerance)):
        raise ValueError("tie_tolerance must be finite and non-negative")
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.ones(X.shape[0]), size=int(draws))
    risks = w @ X  # draws x models
    mins = risks.min(axis=1, keepdims=True)
    tied = risks <= mins + float(tie_tolerance)
    mass = tied / tied.sum(axis=1, keepdims=True)
    prob_best = mass.mean(axis=0)
    mean_risk = X.mean(axis=0)
    min_risk = risks.min(axis=1)
    regret = risks - min_risk[:, None]

    model_table = pd.DataFrame({
        "model": names,
        "mean_patient_risk": mean_risk.astype(float),
        "bb_probability_best": prob_best.astype(float),
        "bb_mean_regret_to_draw_best": regret.mean(axis=0).astype(float),
    }).sort_values(["bb_probability_best", "mean_patient_risk"], ascending=[False, True], ignore_index=True)

    edge_rows: list[dict] = []
    if graph_edges is not None:
        missing = [n for n in names if n not in graph_edges]
        if missing:
            raise KeyError(f"graph_edges missing models: {missing}")
        edge_sets = {n: set(tuple(e) for e in graph_edges[n]) for n in names}
        all_edges = sorted(set().union(*(edge_sets[n] for n in names)))
        raw_prob = {n: float(prob_best[j]) for j, n in enumerate(names)}
        for e in all_edges:
            p = sum(raw_prob[n] for n in names if e in edge_sets[n])
            edge_rows.append({
                "edge_from": str(e[0]),
                "edge_to": str(e[1]),
                "bb_predictive_inclusion": float(p),
            })
    edge_table = pd.DataFrame(edge_rows, columns=["edge_from", "edge_to", "bb_predictive_inclusion"])
    return BayesianGraphProfile(model_table, edge_table, int(draws), int(seed), float(tie_tolerance))
