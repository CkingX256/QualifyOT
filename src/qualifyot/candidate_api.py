from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import copy
import numpy as np
import pandas as pd

from .model import STATES, EDGES, incidence, feature_matrix, fit_graphflow, patient_sample_weights
from .utils import normalize_rows

class CandidatePredictor(ABC):
    """Minimal predictive contract for the generic qualification engine.

    A candidate is fitted only on training-patient rows and returns one
    simplex-valued prediction per supplied row. ``states`` is explicit so the
    evidence layer is architecture- and ontology-agnostic rather than tied to
    the manuscript's five-state T-cell examples.
    """
    name: str = 'Candidate'
    states = None

    def fresh(self):
        return copy.deepcopy(self)

    def state_names(self):
        return list(STATES if self.states is None else self.states)

    @abstractmethod
    def fit(self, pairs: pd.DataFrame) -> 'CandidatePredictor': ...

    @abstractmethod
    def predict(self, pairs: pd.DataFrame) -> np.ndarray: ...

    def validate_predictions(self, pred: np.ndarray, n_rows: int, atol: float=1e-7):
        pred=np.asarray(pred,float); states=self.state_names()
        if pred.shape != (n_rows,len(states)):
            raise ValueError(f'{self.name}: prediction shape {pred.shape}, expected {(n_rows,len(states))}')
        if not np.isfinite(pred).all(): raise ValueError(f'{self.name}: non-finite predictions')
        if (pred < -atol).any() or (np.abs(pred.sum(1)-1)>atol).any():
            raise ValueError(f'{self.name}: predictions do not lie on the simplex')
        return pred

@dataclass
class GraphFlowCandidate(CandidatePredictor):
    alpha: float = 1.0
    patient_balanced: bool = False
    states: tuple | list | None = None
    edges: tuple | list | None = None
    name: str = 'GraphFlow'
    _model: object = None

    def fit(self, pairs: pd.DataFrame):
        states=self.state_names(); edges=list(EDGES if self.edges is None else self.edges)
        X,S,T=feature_matrix(pairs,states=states); patients=pairs.patient_id.to_numpy(str)
        B=incidence(states,edges)
        self._model=fit_graphflow(X,S,T,alpha=self.alpha,Bmat=B,edges=edges,patients=patients,patient_balanced=self.patient_balanced)
        return self

    def predict(self, pairs: pd.DataFrame):
        if self._model is None: raise RuntimeError('candidate not fitted')
        X,S,_=feature_matrix(pairs,states=self.state_names()); p=self._model.predict(X,S)
        return self.validate_predictions(p,len(pairs))

@dataclass
class MeanDeltaCandidate(CandidatePredictor):
    patient_balanced: bool = False
    states: tuple | list | None = None
    name: str = 'MeanDeltaCandidate'
    _delta: np.ndarray = None

    def fit(self, pairs: pd.DataFrame):
        _,S,T=feature_matrix(pairs,states=self.state_names()); d=T-S
        if self.patient_balanced:
            w=patient_sample_weights(pairs.patient_id.to_numpy(str)); self._delta=np.average(d,axis=0,weights=w)
        else: self._delta=d.mean(0)
        return self

    def predict(self, pairs: pd.DataFrame):
        if self._delta is None: raise RuntimeError('candidate not fitted')
        _,S,_=feature_matrix(pairs,states=self.state_names()); p=normalize_rows(S+self._delta)
        return self.validate_predictions(p,len(pairs))

class CallableCandidate(CandidatePredictor):
    """Adapter for external models supplied through fit/predict callables."""
    def __init__(self,fit_fn,predict_fn,name='ExternalCandidate',states=None):
        self.fit_fn=fit_fn; self.predict_fn=predict_fn; self.name=name; self.states=states; self._state=None
    def fresh(self): return CallableCandidate(self.fit_fn,self.predict_fn,self.name,self.states)
    def fit(self,pairs): self._state=self.fit_fn(pairs.copy()); return self
    def predict(self,pairs):
        if self._state is None: raise RuntimeError('candidate not fitted')
        p=np.asarray(self.predict_fn(self._state,pairs.copy()),float)
        return self.validate_predictions(p,len(pairs))

class PrecomputedPredictionCandidate(CandidatePredictor):
    """Frozen external-prediction adapter.

    Expects pair-table columns ``<prefix>__<state>``. ``fit`` deliberately does
    nothing: the external method must have been trained without the held-out
    outcomes before these predictions are supplied. This makes CellRank,
    scVelo, Neural-OT or any other external predictor auditable without making
    QualifyOT depend on that method's software stack.
    """
    def __init__(self,states,prefix='candidate',name='PrecomputedExternal'):
        self.states=list(states); self.prefix=prefix; self.name=name
    def fresh(self): return PrecomputedPredictionCandidate(self.states,self.prefix,self.name)
    def fit(self,pairs): return self
    def predict(self,pairs):
        cols=[f'{self.prefix}__{s}' for s in self.states]
        missing=[c for c in cols if c not in pairs.columns]
        if missing: raise KeyError(f'{self.name}: missing precomputed prediction columns: {missing}')
        return self.validate_predictions(pairs[cols].to_numpy(float),len(pairs))
