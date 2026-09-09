from __future__ import annotations

"""Transparent comparator candidates for longitudinal composition prediction.

These comparators operate on the same patient-level composition tables as
QualifyOT.  ``CompositionEntropicOTCandidate`` is deliberately named as a
state-aggregated entropic optimal-transport comparator; it is *not* presented
as Waddington-OT, MIOFlow, scNODE, CellRank, or any other cell-level method.
Those methods require expression-level inputs and their own modelling
assumptions.  The distinction prevents an unfair or fictitious benchmark.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .candidate_api import CandidatePredictor
from .model import feature_matrix, source_feature_matrix, patient_sample_weights
from .utils import normalize_rows


def _sinkhorn_coupling(a: np.ndarray, b: np.ndarray, epsilon: float, max_iter: int = 2000,
                       tol: float = 1e-10) -> np.ndarray:
    """Entropic OT coupling on a fixed identity/off-diagonal state cost.

    The cost is 0 for staying in the same state and 1 for changing state.  This
    gives a neutral composition-level transport comparator when no expression-
    space geometry is available.  A small numerical floor is used only inside
    Sinkhorn scaling; returned marginals are checked after convergence.
    """
    a=np.asarray(a,float); b=np.asarray(b,float)
    a=np.clip(a,0,None); b=np.clip(b,0,None)
    if a.sum() <= 0 or b.sum() <= 0:
        raise ValueError('composition masses must be positive')
    a=a/a.sum(); b=b/b.sum(); k=len(a)
    if len(b)!=k: raise ValueError('source/target dimension mismatch')
    C=np.ones((k,k),float)-np.eye(k)
    K=np.exp(-C/float(epsilon))
    u=np.ones(k,float); v=np.ones(k,float)
    tiny=np.finfo(float).tiny
    for _ in range(int(max_iter)):
        up=u.copy()
        Kv=K@v
        u=a/np.maximum(Kv,tiny)
        KTu=K.T@u
        v=b/np.maximum(KTu,tiny)
        if np.max(np.abs(u-up)) < tol:
            break
    G=(u[:,None]*K)*v[None,:]
    # Iterative scaling is occasionally one iteration short at stringent tol.
    # A few exact marginal-balancing sweeps close the residual deterministically.
    for _ in range(20):
        rs=G.sum(1); G*=np.divide(a,rs,out=np.ones_like(a),where=rs>0)[:,None]
        cs=G.sum(0); G*=np.divide(b,cs,out=np.ones_like(b),where=cs>0)[None,:]
    return G


@dataclass
class CompositionEntropicOTCandidate(CandidatePredictor):
    """Patient-balanced state-aggregated entropic-OT transition comparator.

    Each training longitudinal pair supplies an entropic coupling from source
    to target composition under a frozen state cost (0 diagonal, 1 off-diagonal).
    Couplings are aggregated with one unit of total mass per physical patient,
    converted to a row-stochastic transition matrix, and applied to held-out
    source compositions.  ``epsilon`` is fixed before scoring; no held-out target
    participates in fitting or tuning.

    This is a fair comparator for *composition tables*.  It must not be called
    Waddington-OT because WOT works in cell-expression space and uses additional
    geometry/growth information unavailable in these processed tables.
    """
    states: tuple | list
    epsilon: float = 0.25
    patient_balanced: bool = True
    name: str = 'CompositionEntropicOT_e0.25'
    _transition: np.ndarray | None = None

    def fresh(self):
        return type(self)(states=list(self.states),epsilon=self.epsilon,
                          patient_balanced=self.patient_balanced,name=self.name)

    def fit(self,pairs: pd.DataFrame):
        _,S,T=feature_matrix(pairs,states=self.states)
        if self.epsilon <= 0: raise ValueError('epsilon must be positive')
        if self.patient_balanced:
            w=patient_sample_weights(pairs.patient_id.astype(str).to_numpy())
        else:
            w=np.ones(len(pairs),float)
        Gsum=np.zeros((len(self.states),len(self.states)),float)
        source_mass=np.zeros(len(self.states),float)
        for j in range(len(pairs)):
            G=_sinkhorn_coupling(S[j],T[j],self.epsilon)
            Gsum += w[j]*G
            source_mass += w[j]*S[j]
        P=np.zeros_like(Gsum)
        for r in range(len(self.states)):
            if source_mass[r] > 1e-12:
                P[r]=Gsum[r]/source_mass[r]
            else:
                P[r,r]=1.0
        self._transition=normalize_rows(P)
        return self

    def predict(self,pairs: pd.DataFrame):
        if self._transition is None: raise RuntimeError('candidate not fitted')
        _,S=source_feature_matrix(pairs,states=self.states)
        p=normalize_rows(S@self._transition)
        return self.validate_predictions(p,len(pairs))

    def candidate_diagnostics(self):
        return {'epsilon':float(self.epsilon),'transition':self._transition.tolist()}
