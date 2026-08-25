from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import Ridge

from .candidate_api import CandidatePredictor, MeanDeltaCandidate
from .model import feature_matrix, patient_sample_weights
from .utils import normalize_rows


def _safe_feature_scaling(X: np.ndarray, median=None, scale=None):
    X=np.asarray(X,float)
    if median is None:
        median=np.nanmedian(X,axis=0)
        median=np.where(np.isfinite(median),median,0.0)
    X=np.where(np.isfinite(X),X,median)
    if scale is None:
        scale=np.std(X,axis=0)
        scale=np.where(np.isfinite(scale) & (scale>1e-8),scale,1.0)
    return (X-median)/scale, median, scale


@dataclass
class DirectDeltaRidgeCandidate(CandidatePredictor):
    states: tuple | list
    alpha: float = 10.0
    patient_balanced: bool = True
    name: str = 'DirectDeltaRidge'
    _ridge: object = None
    _median: np.ndarray | None = None
    _scale: np.ndarray | None = None

    def fit(self,pairs):
        X,S,T=feature_matrix(pairs,states=self.states)
        Xs,self._median,self._scale=_safe_feature_scaling(X)
        w=patient_sample_weights(pairs.patient_id.astype(str).to_numpy()) if self.patient_balanced else None
        self._ridge=Ridge(alpha=self.alpha).fit(Xs,T-S,sample_weight=w)
        return self

    def predict(self,pairs):
        if self._ridge is None: raise RuntimeError('candidate not fitted')
        X,S,_=feature_matrix(pairs,states=self.states)
        Xs,_,_=_safe_feature_scaling(X,self._median,self._scale)
        p=normalize_rows(S+self._ridge.predict(Xs))
        return self.validate_predictions(p,len(pairs))


@dataclass
class TargetRidgeCandidate(CandidatePredictor):
    states: tuple | list
    alpha: float = 10.0
    patient_balanced: bool = True
    name: str = 'TargetRidge_a10_PB'
    _ridge: object = None
    _median: np.ndarray | None = None
    _scale: np.ndarray | None = None

    def fit(self,pairs):
        X,S,T=feature_matrix(pairs,states=self.states)
        Xs,self._median,self._scale=_safe_feature_scaling(X)
        w=patient_sample_weights(pairs.patient_id.astype(str).to_numpy()) if self.patient_balanced else None
        self._ridge=Ridge(alpha=self.alpha).fit(Xs,T,sample_weight=w)
        return self

    def predict(self,pairs):
        if self._ridge is None: raise RuntimeError('candidate not fitted')
        X,_,_=feature_matrix(pairs,states=self.states)
        Xs,_,_=_safe_feature_scaling(X,self._median,self._scale)
        p=normalize_rows(self._ridge.predict(Xs))
        return self.validate_predictions(p,len(pairs))


@dataclass
class RobustBlendCandidate(CandidatePredictor):
    """Low-variance generic candidate with fixed cohort-independent settings.

    The candidate uses:
      50% patient-balanced MeanDelta + 50% patient-balanced DirectDeltaRidge,
      Ridge alpha=10, source/time features only, simplex projection.

    It is a predictive candidate, not a mechanistic biological prior.
    """
    states: tuple | list
    alpha: float = 10.0
    blend: float = 0.5
    name: str = 'RobustBlend'
    _mean_delta: object = None
    _ridge_delta: object = None

    def fit(self,pairs):
        self._mean_delta=MeanDeltaCandidate(states=self.states,patient_balanced=True,name='MeanDelta_PB').fit(pairs)
        self._ridge_delta=DirectDeltaRidgeCandidate(states=self.states,alpha=self.alpha,patient_balanced=True,name='DirectDeltaRidge').fit(pairs)
        return self

    def predict(self,pairs):
        if self._mean_delta is None or self._ridge_delta is None: raise RuntimeError('candidate not fitted')
        a=self._mean_delta.predict(pairs); b=self._ridge_delta.predict(pairs)
        p=normalize_rows((1.0-self.blend)*a+self.blend*b)
        return self.validate_predictions(p,len(pairs))
