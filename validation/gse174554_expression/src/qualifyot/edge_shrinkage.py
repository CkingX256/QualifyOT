from __future__ import annotations

"""Continuous edge attenuation for design-mode structured prediction.

This is deliberately called *edge shrinkage*, not graphical LASSO: the latter
has a specific meaning for sparse precision-matrix estimation.  Here a fixed
pre-specified directed supergraph is retained, and non-negative attenuation
coefficients a_e in [0,1] shrink predicted graph flows continuously.  The
coefficients and regularization strength are chosen from training-patient
cross-fitted predictions only; held-out outer targets are never used.
"""

from dataclasses import dataclass
import hashlib
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .candidate_api import CandidatePredictor
from .model import feature_matrix, fit_graphflow, incidence, residual_flow, project_flow, mae


@dataclass(frozen=True)
class EdgeShrinkagePathRow:
    gamma: float
    patient_equal_risk: float
    standard_error: float
    edge_weight_sum: float
    active_edges: int
    weights: tuple[float, ...]


class ContinuousEdgeShrinkageCandidate(CandidatePredictor):
    """Training-only continuous attenuation over a fixed directed supergraph."""

    def __init__(
        self,
        states,
        edges,
        *,
        gamma_grid=(0.0, 0.001, 0.003, 0.01, 0.03),
        alpha=1.0,
        patient_balanced=True,
        cv_folds=5,
        active_threshold=0.05,
        seed=20260830,
        name="ContinuousEdgeShrinkage",
    ):
        self.states=list(states)
        self.edges=tuple(tuple(e) for e in edges)
        if not self.edges:
            raise ValueError("ContinuousEdgeShrinkageCandidate requires at least one edge")
        self.gamma_grid=tuple(float(x) for x in gamma_grid)
        if not self.gamma_grid or any((not np.isfinite(x) or x < 0) for x in self.gamma_grid):
            raise ValueError("gamma_grid must contain finite non-negative values")
        self.alpha=float(alpha)
        self.patient_balanced=bool(patient_balanced)
        self.cv_folds=int(cv_folds)
        self.active_threshold=float(active_threshold)
        self.seed=int(seed)
        self.name=str(name)
        self.selected_edge_weights=None
        self.selected_gamma=None
        self.path_=None
        self._model=None
        self._B=incidence(self.states,self.edges)

    def fresh(self):
        return ContinuousEdgeShrinkageCandidate(
            self.states,self.edges,gamma_grid=self.gamma_grid,alpha=self.alpha,
            patient_balanced=self.patient_balanced,cv_folds=self.cv_folds,
            active_threshold=self.active_threshold,seed=self.seed,name=self.name,
        )

    def _fold_map(self, patients):
        up=sorted(set(map(str,patients)))
        k=min(self.cv_folds,len(up))
        if k<2: raise ValueError("at least two training patients are required")
        ordered=sorted(up,key=lambda x: hashlib.sha256((str(self.seed)+"|"+x).encode()).hexdigest())
        return {p:j%k for j,p in enumerate(ordered)},k

    def _predict_from_flows(self,S,F,a):
        out=[]
        a=np.asarray(a,float)
        for s,f in zip(S,F):
            out.append(project_flow(s,np.asarray(f)*a,self._B)[1])
        return np.vstack(out)

    def _patient_risk(self,T,pred,P):
        l=mae(T,pred); up=np.unique(P)
        vals=np.asarray([l[P==p].mean() for p in up],float)
        return vals,float(vals.mean()),float(vals.std(ddof=1)/math.sqrt(len(vals))) if len(vals)>1 else 0.0

    def _optimize(self,S,T,F,P,gamma):
        m=len(self.edges)
        def objective(a):
            pred=self._predict_from_flows(S,F,a)
            _,risk,_=self._patient_risk(T,pred,P)
            return risk+float(gamma)*float(np.sum(a))/m
        x0=np.ones(m,float)
        res=minimize(objective,x0,method="L-BFGS-B",bounds=[(0.0,1.0)]*m,options={"maxiter":300,"ftol":1e-12})
        if not res.success and not np.isfinite(res.fun):
            raise RuntimeError("edge-shrinkage optimization failed: "+str(res.message))
        a=np.clip(np.asarray(res.x,float),0,1)
        pred=self._predict_from_flows(S,F,a)
        _,risk,se=self._patient_risk(T,pred,P)
        return a,risk,se

    def fit(self,pairs: pd.DataFrame):
        pairs=pairs.reset_index(drop=True).copy()
        X,S,T=feature_matrix(pairs,states=self.states); P=pairs.patient_id.astype(str).to_numpy()
        if len(np.unique(P))<3:
            raise ValueError("ContinuousEdgeShrinkageCandidate requires at least three patients")
        B=self._B
        Fall=np.vstack([residual_flow(s,t,Bmat=B)[0] for s,t in zip(S,T)])
        fmap,k=self._fold_map(P)
        Foof=np.zeros_like(Fall)
        for fold in range(k):
            te=np.asarray([fmap[p]==fold for p in P]); tr=~te
            if not te.any(): continue
            gm=fit_graphflow(X[tr],S[tr],T[tr],alpha=self.alpha,flow_labels=Fall[tr],Bmat=B,edges=self.edges,
                             patients=P[tr],patient_balanced=self.patient_balanced)
            Foof[te]=gm.predict_flows(X[te])
        if not np.isfinite(Foof).all():
            raise RuntimeError("non-finite OOF flow predictions")
        rows=[]
        candidates=[]
        for gamma in self.gamma_grid:
            a,risk,se=self._optimize(S,T,Foof,P,gamma)
            candidates.append((gamma,a,risk,se))
        best=min(candidates,key=lambda x:(x[2],x[0]))
        threshold=best[2]+best[3]
        near=[x for x in candidates if x[2]<=threshold+1e-15]
        # One-standard-error complexity preference: among statistically similar
        # risks, choose the strongest shrinkage / lowest edge mass.
        chosen=min(near,key=lambda x:(float(np.sum(x[1])),int(np.sum(x[1]>self.active_threshold)),-x[0],x[2]))
        self.selected_gamma=float(chosen[0]); self.selected_edge_weights=np.asarray(chosen[1],float)
        for gamma,a,risk,se in candidates:
            rows.append(EdgeShrinkagePathRow(float(gamma),float(risk),float(se),float(a.sum()),
                                             int(np.sum(a>self.active_threshold)),tuple(map(float,a))))
        self.path_=tuple(rows)
        self._model=fit_graphflow(X,S,T,alpha=self.alpha,flow_labels=Fall,Bmat=B,edges=self.edges,
                                  patients=P,patient_balanced=self.patient_balanced)
        return self

    def predict(self,pairs: pd.DataFrame):
        if self._model is None or self.selected_edge_weights is None:
            raise RuntimeError("candidate not fitted")
        X,S,_=feature_matrix(pairs,states=self.states)
        F=self._model.predict_flows(X)
        p=self._predict_from_flows(S,F,self.selected_edge_weights)
        return self.validate_predictions(p,len(pairs))

    def edge_weights(self) -> dict[str,float]:
        if self.selected_edge_weights is None:
            raise RuntimeError("candidate not fitted")
        return {f"{a}->{b}":float(w) for (a,b),w in zip(self.edges,self.selected_edge_weights)}

    def regularization_path(self) -> pd.DataFrame:
        if self.path_ is None:
            raise RuntimeError("candidate not fitted")
        rows=[]
        for r in self.path_:
            d={"gamma":r.gamma,"patient_equal_risk":r.patient_equal_risk,"standard_error":r.standard_error,
               "edge_weight_sum":r.edge_weight_sum,"active_edges":r.active_edges}
            for e,w in zip(self.edges,r.weights): d[f"weight__{e[0]}->{e[1]}"]=w
            rows.append(d)
        return pd.DataFrame(rows)
