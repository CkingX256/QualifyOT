from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .candidate_api import CandidatePredictor
from .model import feature_matrix, fit_graphflow, incidence, residual_flow
from .safe_averaging import safe_graph_average


def _graph_key(edges) -> str:
    return "|".join(f"{a}->{b}" for a, b in edges)


class SafeGraphAveragingCandidate(CandidatePredictor):
    """Training-only convex averaging over a pre-specified graph library.

    Every call to ``fit`` selects graph weights using only patient-split
    out-of-fold predictions from the rows supplied to that call. The frozen
    graph is always present as a fallback. The selected mixture is therefore a
    redesigned *candidate* that must still be evaluated on held-out patients by
    the outer QualifyOT evidence engine.
    """

    def __init__(
        self, states, graph_library, frozen_edges, *, alpha=1.0, patient_balanced=True,
        cv_folds=3, grid_step=0.25, safety_alpha=0.05, safety_margin=0.0,
        selector_bootstrap=300, seed=20260830, one_se=True, name="SafeGraphAverage"
    ):
        self.states=list(states)
        self.graph_library=[tuple(tuple(e) for e in g) for g in graph_library]
        self.frozen_edges=tuple(tuple(e) for e in frozen_edges)
        self.alpha=float(alpha)
        self.patient_balanced=bool(patient_balanced)
        self.cv_folds=int(cv_folds)
        self.grid_step=float(grid_step)
        self.safety_alpha=float(safety_alpha)
        self.safety_margin=float(safety_margin)
        self.selector_bootstrap=int(selector_bootstrap)
        self.seed=int(seed)
        self.one_se=bool(one_se)
        self.name=str(name)
        self._models=None
        self.selected_weights=None
        self.selection_result=None
        if self.frozen_edges not in self.graph_library:
            raise ValueError("frozen_edges must be included in graph_library")
        if len(self.graph_library)<2:
            raise ValueError("graph_library must contain at least two graphs")

    def fresh(self):
        return SafeGraphAveragingCandidate(
            states=self.states,
            graph_library=self.graph_library,
            frozen_edges=self.frozen_edges,
            alpha=self.alpha,
            patient_balanced=self.patient_balanced,
            cv_folds=self.cv_folds,
            grid_step=self.grid_step,
            safety_alpha=self.safety_alpha,
            safety_margin=self.safety_margin,
            selector_bootstrap=self.selector_bootstrap,
            seed=self.seed,
            one_se=self.one_se,
            name=self.name,
        )

    def _fold_map(self, patients):
        up = sorted(set(map(str, patients)))
        k = min(self.cv_folds, len(up))
        if k < 2:
            raise ValueError("at least two patients are required")
        ordered = sorted(up, key=lambda x: hashlib.sha256(x.encode()).hexdigest())
        return {p: j % k for j, p in enumerate(ordered)}, k

    def _flow_labels(self, S, T, graph):
        B = incidence(self.states, graph)
        return B, np.vstack([residual_flow(s, t, Bmat=B)[0] for s, t in zip(S, T)])

    def fit(self, pairs: pd.DataFrame):
        pairs = pairs.reset_index(drop=True).copy()
        X, S, T = feature_matrix(pairs, states=self.states)
        P = pairs.patient_id.astype(str).to_numpy()
        if len(np.unique(P)) < 3:
            raise ValueError("SafeGraphAveragingCandidate requires at least three training patients")
        fmap, k = self._fold_map(P)
        oof = np.zeros((len(pairs), len(self.graph_library), len(self.states)), float)
        BF = []
        for gi, graph in enumerate(self.graph_library):
            B, F = self._flow_labels(S, T, graph)
            BF.append((B, F))
            for fold in range(k):
                te = np.asarray([fmap[x] == fold for x in P])
                tr = ~te
                if not te.any() or len(np.unique(P[tr])) < 2:
                    continue
                gm = fit_graphflow(
                    X[tr], S[tr], T[tr], alpha=self.alpha, flow_labels=F[tr],
                    Bmat=B, edges=graph, patients=P[tr], patient_balanced=self.patient_balanced,
                )
                oof[te, gi] = gm.predict(X[te], S[te])
        if np.any(oof.sum(axis=2) == 0):
            raise RuntimeError("incomplete graph-library OOF predictions")

        frozen_index = self.graph_library.index(self.frozen_edges)
        sel = safe_graph_average(
            T, oof, P, frozen_index=frozen_index, step=self.grid_step,
            alpha=self.safety_alpha, safety_margin=self.safety_margin,
            bootstrap=self.selector_bootstrap, seed=self.seed + len(np.unique(P)) * 97,
            one_se=self.one_se,
        )
        self.selected_weights = sel.weights.copy()
        self.selection_result = sel
        self._models = []
        for graph, (B, F) in zip(self.graph_library, BF):
            self._models.append(
                fit_graphflow(
                    X, S, T, alpha=self.alpha, flow_labels=F, Bmat=B, edges=graph,
                    patients=P, patient_balanced=self.patient_balanced,
                )
            )
        return self

    def predict(self, pairs: pd.DataFrame):
        if self._models is None or self.selected_weights is None:
            raise RuntimeError("candidate not fitted")
        X, S, _ = feature_matrix(pairs, states=self.states)
        pred = np.stack([m.predict(X, S) for m in self._models], axis=1)
        out = np.einsum("m,rms->rs", self.selected_weights, pred)
        return self.validate_predictions(out, len(pairs))

    def selected_graph_weights(self) -> dict[str, float]:
        if self.selected_weights is None:
            raise RuntimeError("candidate not fitted")
        return {_graph_key(g): float(w) for g, w in zip(self.graph_library, self.selected_weights)}

class RobustGraphAveragingCandidate(SafeGraphAveragingCandidate):
    """Training-only graph averaging with both mean- and CVaR-harm screens.

    The class preserves the normal CandidatePredictor contract.  It is a
    design-mode extension: any fitted mixture is still a *new candidate* and
    must pass the outer QualifyOT qualification procedure.  The tail screen is
    empirical and should be reported as an orthogonal safety profile rather
    than interpreted as a causal subgroup guarantee.
    """
    def __init__(
        self, states, graph_library, frozen_edges, *, alpha=1.0, patient_balanced=True,
        cv_folds=3, grid_step=0.25, safety_alpha=0.05, safety_margin=0.0,
        tail_fraction=0.20, tail_alpha=0.05, tail_harm_margin=0.0,
        selector_bootstrap=500, seed=20260830, one_se=True, name="RobustGraphAverage"
    ):
        super().__init__(states,graph_library,frozen_edges,alpha=alpha,patient_balanced=patient_balanced,
                         cv_folds=cv_folds,grid_step=grid_step,safety_alpha=safety_alpha,
                         safety_margin=safety_margin,selector_bootstrap=selector_bootstrap,
                         seed=seed,one_se=one_se,name=name)
        self.tail_fraction=float(tail_fraction)
        self.tail_alpha=float(tail_alpha)
        self.tail_harm_margin=float(tail_harm_margin)

    def fresh(self):
        return RobustGraphAveragingCandidate(
            states=self.states, graph_library=self.graph_library, frozen_edges=self.frozen_edges,
            alpha=self.alpha, patient_balanced=self.patient_balanced, cv_folds=self.cv_folds,
            grid_step=self.grid_step, safety_alpha=self.safety_alpha, safety_margin=self.safety_margin,
            tail_fraction=self.tail_fraction, tail_alpha=self.tail_alpha, tail_harm_margin=self.tail_harm_margin,
            selector_bootstrap=self.selector_bootstrap, seed=self.seed, one_se=self.one_se, name=self.name,
        )

    def fit(self, pairs: pd.DataFrame):
        from .robust_averaging import robust_graph_average
        pairs = pairs.reset_index(drop=True).copy()
        X, S, T = feature_matrix(pairs, states=self.states)
        P = pairs.patient_id.astype(str).to_numpy()
        if len(np.unique(P)) < 3:
            raise ValueError("RobustGraphAveragingCandidate requires at least three training patients")
        fmap, k = self._fold_map(P)
        oof = np.zeros((len(pairs), len(self.graph_library), len(self.states)), float)
        BF = []
        for gi, graph in enumerate(self.graph_library):
            B, F = self._flow_labels(S, T, graph)
            BF.append((B, F))
            for fold in range(k):
                te = np.asarray([fmap[x] == fold for x in P])
                tr = ~te
                if not te.any() or len(np.unique(P[tr])) < 2:
                    continue
                gm = fit_graphflow(
                    X[tr], S[tr], T[tr], alpha=self.alpha, flow_labels=F[tr],
                    Bmat=B, edges=graph, patients=P[tr], patient_balanced=self.patient_balanced,
                )
                oof[te, gi] = gm.predict(X[te], S[te])
        if np.any(oof.sum(axis=2) == 0):
            raise RuntimeError("incomplete graph-library OOF predictions")
        frozen_index = self.graph_library.index(self.frozen_edges)
        sel = robust_graph_average(
            T, oof, P, frozen_index=frozen_index, step=self.grid_step,
            alpha_mean=self.safety_alpha, mean_harm_margin=self.safety_margin,
            tail_fraction=self.tail_fraction, alpha_tail=self.tail_alpha,
            tail_harm_margin=self.tail_harm_margin, bootstrap=self.selector_bootstrap,
            seed=self.seed + len(np.unique(P)) * 97, one_se=self.one_se,
        )
        self.selected_weights = sel.weights.copy()
        self.selection_result = sel
        self._models = []
        for graph, (B, F) in zip(self.graph_library, BF):
            self._models.append(
                fit_graphflow(
                    X, S, T, alpha=self.alpha, flow_labels=F, Bmat=B, edges=graph,
                    patients=P, patient_balanced=self.patient_balanced,
                )
            )
        return self
