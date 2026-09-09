from __future__ import annotations
from dataclasses import dataclass
import copy, hashlib
import itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .candidate_api import CandidatePredictor
from .candidates import RobustBlendCandidate
from .model import source_feature_matrix, feature_matrix, patient_sample_weights, mae
from .utils import normalize_rows, patient_equal


def _clr(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    x=np.asarray(x,float)
    z=np.log(np.clip(x,eps,None))
    return z-z.mean(axis=1,keepdims=True)

def _softmax(z: np.ndarray) -> np.ndarray:
    z=np.asarray(z,float); z=z-z.max(axis=1,keepdims=True)
    e=np.exp(z); return e/e.sum(axis=1,keepdims=True)

def _source_design(pairs: pd.DataFrame, states, *, quadratic=False, logratio=False):
    X,S=source_feature_matrix(pairs,states=states)
    dt=np.asarray(X[:,-1],float)[:,None]
    cols=[]
    if logratio: cols.append(_clr(S))
    cols.append(S)
    cols.append(np.log1p(np.clip(dt,0,None)))
    if quadratic:
        cols.append(S*S)
        inter=[]
        for a in range(S.shape[1]):
            for b in range(a+1,S.shape[1]): inter.append((S[:,a]*S[:,b])[:,None])
        if inter: cols.append(np.hstack(inter))
        cols.append(S*np.log1p(np.clip(dt,0,None)))
    return np.hstack(cols),S

def _standardize_fit(X):
    med=np.nanmedian(X,axis=0); med=np.where(np.isfinite(med),med,0.0)
    X=np.where(np.isfinite(X),X,med)
    scale=np.std(X,axis=0); scale=np.where(np.isfinite(scale)&(scale>1e-8),scale,1.0)
    return (X-med)/scale,med,scale

def _standardize_apply(X,med,scale):
    X=np.where(np.isfinite(X),X,med)
    return (X-med)/scale

def _patient_weighted_standardize_fit(X, weights):
    """Replication-invariant feature standardization for patient-unit fits.

    Both centering and scaling use the same patient-unit observation weights as
    the downstream penalized estimator.  Mechanical replication of rows from a
    patient therefore leaves the weighted feature moments unchanged, closing a
    subtle invariance gap that remains if unweighted medians/standard deviations
    are used before a patient-normalized Ridge fit.
    """
    X=np.asarray(X,float); w=np.asarray(weights,float).reshape(-1)
    if X.ndim!=2 or len(w)!=len(X) or len(w)==0:
        raise ValueError('X/weights shape mismatch')
    if not np.isfinite(w).all() or (w<0).any() or w.sum()<=0:
        raise ValueError('weights must be finite, non-negative and non-zero')
    finite=np.isfinite(X)
    den=(finite*w[:,None]).sum(axis=0)
    num=np.where(finite,X,0.0)*w[:,None]
    center=np.divide(num.sum(axis=0),den,out=np.zeros(X.shape[1],float),where=den>0)
    Xi=np.where(finite,X,center)
    var=((Xi-center)**2*w[:,None]).sum(axis=0)/w.sum()
    scale=np.sqrt(np.maximum(var,0.0))
    scale=np.where(np.isfinite(scale)&(scale>1e-8),scale,1.0)
    return (Xi-center)/scale,center,scale

def _deterministic_fold_map(patients, k=5):
    up=sorted(set(map(str,patients)))
    k=max(2,min(int(k),len(up)))
    ordered=sorted(up,key=lambda x:hashlib.sha256(x.encode()).hexdigest())
    return {p:j%k for j,p in enumerate(ordered)},k

@dataclass
class LogRatioDeltaRidgeCandidate(CandidatePredictor):
    """Patient-balanced compositional ridge in centered-log-ratio delta space.

    The model predicts CLR(target)-CLR(source), then maps back through softmax.
    Alpha may be selected *inside training data only* by patient-split OOF risk.
    This preserves the CandidatePredictor contract and never uses held-out targets
    during predict().
    """
    states: tuple|list
    alpha: float = 10.0
    alpha_grid: tuple|list|None = (1.0,10.0,100.0)
    cv_folds: int = 5
    one_se: bool = True
    patient_balanced: bool = True
    eps: float = 1e-5
    name: str = 'CLRDeltaRidge'
    _ridge: object = None
    _med: np.ndarray|None = None
    _scale: np.ndarray|None = None
    selected_alpha: float|None = None

    def fresh(self): return copy.deepcopy(self.__class__(states=list(self.states),alpha=self.alpha,alpha_grid=self.alpha_grid,cv_folds=self.cv_folds,one_se=self.one_se,patient_balanced=self.patient_balanced,eps=self.eps,name=self.name))

    def _fit_alpha(self,pairs,alpha):
        X,S,T=feature_matrix(pairs,states=self.states); D=_clr(T,self.eps)-_clr(S,self.eps)
        Z,_=_source_design(pairs,self.states,logratio=True)
        Zs,med,scale=_standardize_fit(Z)
        w=patient_sample_weights(pairs.patient_id.astype(str).to_numpy()) if self.patient_balanced else None
        r=Ridge(alpha=float(alpha)).fit(Zs,D,sample_weight=w)
        return r,med,scale

    def _pred_model(self,pairs,r,med,scale):
        Z,S=_source_design(pairs,self.states,logratio=True); Zs=_standardize_apply(Z,med,scale)
        return _softmax(_clr(S,self.eps)+r.predict(Zs))

    def _choose_alpha(self,pairs):
        grid=[float(x) for x in (self.alpha_grid or [self.alpha])]
        P=pairs.patient_id.astype(str).to_numpy(); up=np.unique(P)
        if len(grid)==1 or len(up)<4: return grid[0]
        fmap,k=_deterministic_fold_map(P,self.cv_folds)
        risks=[]; ses=[]
        for a in grid:
            pred=np.zeros((len(pairs),len(self.states)),float)
            for fold in range(k):
                te=np.asarray([fmap[x]==fold for x in P]); tr=~te
                if not te.any(): continue
                r,m,s=self._fit_alpha(pairs.loc[tr].reset_index(drop=True),a)
                pred[te]=self._pred_model(pairs.loc[te].reset_index(drop=True),r,m,s)
            T=pairs[[f'target__{s}' for s in self.states]].to_numpy(float)
            pl=[]
            for p in up:
                m=P==p; pl.append(float(mae(T[m],pred[m]).mean()))
            pl=np.asarray(pl,float); risks.append(pl.mean()); ses.append(pl.std(ddof=1)/np.sqrt(len(pl)) if len(pl)>1 else np.inf)
        best=int(np.argmin(risks))
        if not self.one_se:return grid[best]
        threshold=risks[best]+ses[best]
        eligible=[(a,r) for a,r in zip(grid,risks) if r<=threshold+1e-12]
        return max(eligible,key=lambda ar:ar[0])[0]  # stronger regularization

    def fit(self,pairs):
        pairs=pairs.reset_index(drop=True).copy(); self.selected_alpha=self._choose_alpha(pairs)
        self._ridge,self._med,self._scale=self._fit_alpha(pairs,self.selected_alpha); return self
    def predict(self,pairs):
        if self._ridge is None: raise RuntimeError('candidate not fitted')
        p=self._pred_model(pairs,self._ridge,self._med,self._scale)
        return self.validate_predictions(p,len(pairs))
    def candidate_diagnostics(self): return {'selected_alpha':float(self.selected_alpha)}

@dataclass
class QuadraticDeltaRidgeCandidate(CandidatePredictor):
    """Low-capacity nonlinear source/time interaction model in composition space."""
    states: tuple|list
    alpha: float=10.0
    alpha_grid: tuple|list|None=(1.0,10.0,100.0)
    cv_folds:int=5
    one_se:bool=True
    patient_balanced:bool=True
    name:str='QuadraticDeltaRidge'
    _ridge:object=None; _med:np.ndarray|None=None; _scale:np.ndarray|None=None; selected_alpha:float|None=None
    def fresh(self): return self.__class__(states=list(self.states),alpha=self.alpha,alpha_grid=self.alpha_grid,cv_folds=self.cv_folds,one_se=self.one_se,patient_balanced=self.patient_balanced,name=self.name)
    def _fit_alpha(self,pairs,a):
        _,S,T=feature_matrix(pairs,states=self.states); Z,_=_source_design(pairs,self.states,quadratic=True)
        Zs,med,scale=_standardize_fit(Z); w=patient_sample_weights(pairs.patient_id.astype(str).to_numpy()) if self.patient_balanced else None
        r=Ridge(alpha=float(a)).fit(Zs,T-S,sample_weight=w); return r,med,scale
    def _pred_model(self,pairs,r,med,scale):
        Z,S=_source_design(pairs,self.states,quadratic=True); return normalize_rows(S+r.predict(_standardize_apply(Z,med,scale)))
    def _choose_alpha(self,pairs):
        grid=[float(x) for x in (self.alpha_grid or [self.alpha])]; P=pairs.patient_id.astype(str).to_numpy(); up=np.unique(P)
        if len(grid)==1 or len(up)<4:return grid[0]
        fmap,k=_deterministic_fold_map(P,self.cv_folds); risks=[]; ses=[]
        for a in grid:
            pred=np.zeros((len(pairs),len(self.states)))
            for fold in range(k):
                te=np.asarray([fmap[x]==fold for x in P]);tr=~te
                if not te.any():continue
                r,m,s=self._fit_alpha(pairs.loc[tr].reset_index(drop=True),a);pred[te]=self._pred_model(pairs.loc[te].reset_index(drop=True),r,m,s)
            T=pairs[[f'target__{s}' for s in self.states]].to_numpy(float);pl=np.array([mae(T[P==p],pred[P==p]).mean() for p in up])
            risks.append(pl.mean());ses.append(pl.std(ddof=1)/np.sqrt(len(pl)) if len(pl)>1 else np.inf)
        b=int(np.argmin(risks));thr=risks[b]+ses[b]
        if not self.one_se:return grid[b]
        return max([a for a,r in zip(grid,risks) if r<=thr+1e-12])
    def fit(self,pairs):
        pairs=pairs.reset_index(drop=True).copy();self.selected_alpha=self._choose_alpha(pairs);self._ridge,self._med,self._scale=self._fit_alpha(pairs,self.selected_alpha);return self
    def predict(self,pairs):
        if self._ridge is None:raise RuntimeError('candidate not fitted')
        return self.validate_predictions(self._pred_model(pairs,self._ridge,self._med,self._scale),len(pairs))
    def candidate_diagnostics(self): return {'selected_alpha':float(self.selected_alpha)}


def _simplex_grid(m:int,step:float):
    q=int(round(1.0/step))
    if abs(q*step-1)>1e-8: raise ValueError('step must divide 1')
    out=[]
    def rec(prefix,left,depth):
        if depth==m-1:
            out.append(np.array(prefix+[left],float)/q);return
        for x in range(left+1):rec(prefix+[x],left-x,depth+1)
    rec([],q,0);return out

class CrossFittedSimplexEnsembleCandidate(CandidatePredictor):
    """Training-only stacked ensemble over a *pre-specified* candidate library.

    Base predictions are generated by patient-split OOF fitting.  A convex
    weight is selected by patient-equal MAE with a one-standard-error rule that
    prefers the designated low-capacity anchor and fewer active components.
    The outer held-out target is never used for weight selection.
    """
    def __init__(self,states,base_candidates,*,anchor_index=0,cv_folds=5,grid_step=0.25,one_se=True,name='CrossFittedSimplexEnsemble'):
        self.states=list(states);self.base_candidates=list(base_candidates);self.anchor_index=int(anchor_index);self.cv_folds=int(cv_folds);self.grid_step=float(grid_step);self.one_se=bool(one_se);self.name=name
        if len(self.base_candidates)<2:raise ValueError('need at least two base candidates')
        if not 0<=self.anchor_index<len(self.base_candidates):raise ValueError('invalid anchor')
        self.selected_weights=None;self._models=None;self._selection_diag=None
    def fresh(self): return CrossFittedSimplexEnsembleCandidate(self.states,[c.fresh() for c in self.base_candidates],anchor_index=self.anchor_index,cv_folds=self.cv_folds,grid_step=self.grid_step,one_se=self.one_se,name=self.name)
    def fit(self,pairs):
        pairs=pairs.reset_index(drop=True).copy();P=pairs.patient_id.astype(str).to_numpy();up=np.unique(P)
        if len(up)<3:raise ValueError('ensemble requires at least 3 patients')
        fmap,k=_deterministic_fold_map(P,self.cv_folds);M=len(self.base_candidates);oof=np.zeros((len(pairs),M,len(self.states)))
        for j,base in enumerate(self.base_candidates):
            for fold in range(k):
                te=np.asarray([fmap[x]==fold for x in P]);tr=~te
                if not te.any():continue
                c=base.fresh().fit(pairs.loc[tr].reset_index(drop=True));oof[te,j]=c.predict(pairs.loc[te].reset_index(drop=True))
        if np.any(oof.sum(axis=2)==0):raise RuntimeError('incomplete ensemble OOF predictions')
        T=pairs[[f'target__{s}' for s in self.states]].to_numpy(float);cands=[]
        for w in _simplex_grid(M,self.grid_step):
            pred=np.einsum('m,rms->rs',w,oof); pl=np.array([mae(T[P==p],pred[P==p]).mean() for p in up],float)
            cands.append((w,float(pl.mean()),float(pl.std(ddof=1)/np.sqrt(len(pl))) if len(pl)>1 else np.inf))
        best=min(cands,key=lambda x:x[1]);thr=best[1]+best[2] if self.one_se else best[1]+1e-15
        eligible=[x for x in cands if x[1]<=thr+1e-12]
        # Conservative deterministic tie break: more anchor, then fewer active, then risk, then lexical weights.
        chosen=min(eligible,key=lambda x:(-x[0][self.anchor_index],int(np.sum(x[0]>1e-12)),x[1],tuple(x[0])))
        self.selected_weights=chosen[0].copy(); self._selection_diag={'oof_risk':chosen[1],'best_oof_risk':best[1],'one_se_threshold':thr,'base_names':[c.name for c in self.base_candidates],'weights':self.selected_weights.tolist()}
        self._models=[c.fresh().fit(pairs) for c in self.base_candidates];return self
    def predict(self,pairs):
        if self._models is None:raise RuntimeError('candidate not fitted')
        pred=np.stack([c.predict(pairs) for c in self._models],axis=1);out=np.einsum('m,rms->rs',self.selected_weights,pred)
        return self.validate_predictions(out,len(pairs))
    def candidate_diagnostics(self):return copy.deepcopy(self._selection_diag)

class NestedRobustBlendCandidate(CandidatePredictor):
    """Nested patient-level tuning of a two-member low-variance blend.

    The candidate family is fixed in advance: patient-balanced MeanDelta and
    DirectDeltaRidge. Ridge alpha and the convex blend are selected *inside the
    training patients only* using deterministic patient-split OOF predictions.
    This turns the previously fixed RobustBlend into a leakage-safe adaptive
    candidate without adding a new evidence gate.
    """
    def __init__(self,states,*,alpha_grid=(1.0,10.0,100.0),blend_grid=(0.0,.25,.5,.75,1.0),cv_folds=5,one_se=False,name='NestedRobustBlend'):
        self.states=list(states);self.alpha_grid=tuple(map(float,alpha_grid));self.blend_grid=tuple(map(float,blend_grid));self.cv_folds=int(cv_folds);self.one_se=bool(one_se);self.name=name
        self.selected_alpha=None;self.selected_blend=None;self._model=None;self._diag=None
    def fresh(self):return type(self)(self.states,alpha_grid=self.alpha_grid,blend_grid=self.blend_grid,cv_folds=self.cv_folds,one_se=self.one_se,name=self.name)
    def fit(self,pairs):
        from .candidate_api import MeanDeltaCandidate
        from .candidates import DirectDeltaRidgeCandidate
        pairs=pairs.reset_index(drop=True).copy();P=pairs.patient_id.astype(str).to_numpy();up=np.unique(P);T=pairs[[f'target__{s}' for s in self.states]].to_numpy(float)
        fmap,k=_deterministic_fold_map(P,self.cv_folds)
        risks=[]
        for a in self.alpha_grid:
            pm=np.zeros_like(T);pr=np.zeros_like(T)
            for fold in range(k):
                te=np.asarray([fmap[x]==fold for x in P]);tr=~te
                if not te.any():continue
                trdf=pairs.loc[tr].reset_index(drop=True);tedf=pairs.loc[te].reset_index(drop=True)
                pm[te]=MeanDeltaCandidate(states=self.states,patient_balanced=True).fit(trdf).predict(tedf)
                pr[te]=DirectDeltaRidgeCandidate(states=self.states,alpha=a,patient_balanced=True).fit(trdf).predict(tedf)
            for b in self.blend_grid:
                pred=normalize_rows((1-b)*pm+b*pr);pl=np.array([mae(T[P==p],pred[P==p]).mean() for p in up],float)
                risks.append({'alpha':a,'blend':b,'risk':float(pl.mean()),'se':float(pl.std(ddof=1)/np.sqrt(len(pl))) if len(pl)>1 else np.inf})
        best=min(risks,key=lambda x:(x['risk'],x['alpha'],x['blend']))
        if self.one_se:
            thr=best['risk']+best['se']; eligible=[x for x in risks if x['risk']<=thr+1e-12]
            # stronger shrinkage first, then central blend, then risk
            sel=min(eligible,key=lambda x:(-x['alpha'],abs(x['blend']-.5),x['risk']))
        else:sel=best
        self.selected_alpha=float(sel['alpha']);self.selected_blend=float(sel['blend']);self._diag={'selected_alpha':self.selected_alpha,'selected_blend':self.selected_blend,'oof_risk':sel['risk'],'best_oof_risk':best['risk'],'one_se':self.one_se}
        self._model=RobustBlendCandidate(states=self.states,alpha=self.selected_alpha,blend=self.selected_blend,name=self.name).fit(pairs);return self
    def predict(self,pairs):
        if self._model is None:raise RuntimeError('candidate not fitted')
        return self._model.predict(pairs)
    def candidate_diagnostics(self):return copy.deepcopy(self._diag)

class ContextualDeltaRidgeCandidate(CandidatePredictor):
    """Patient-balanced delta ridge with explicitly declared predictor-side context.

    Only columns named in ``numeric_covariates`` / ``categorical_covariates`` are
    read. The class never auto-discovers columns, preventing outcome/target
    leakage through convenience feature selection. Categories and scaling are
    learned from the training rows only; unseen test categories map to all-zero
    indicators. Alpha may be selected within training patients by OOF risk.
    """
    def __init__(self,states,*,numeric_covariates=(),categorical_covariates=(),alpha_grid=(10.0,100.0,1000.0),cv_folds=5,one_se=True,patient_balanced=True,name='ContextualDeltaRidge'):
        self.states=list(states);self.numeric_covariates=tuple(numeric_covariates);self.categorical_covariates=tuple(categorical_covariates);self.alpha_grid=tuple(map(float,alpha_grid));self.cv_folds=int(cv_folds);self.one_se=bool(one_se);self.patient_balanced=bool(patient_balanced);self.name=name
        self._ridge=None;self._med=None;self._scale=None;self._levels=None;self.selected_alpha=None
    def fresh(self):return type(self)(self.states,numeric_covariates=self.numeric_covariates,categorical_covariates=self.categorical_covariates,alpha_grid=self.alpha_grid,cv_folds=self.cv_folds,one_se=self.one_se,patient_balanced=self.patient_balanced,name=self.name)
    def _design(self,pairs,fit=False):
        X,S=source_feature_matrix(pairs,states=self.states);base=np.c_[S,np.log1p(np.clip(X[:,-1],0,None))];parts=[base]
        if self.numeric_covariates:
            miss=[c for c in self.numeric_covariates if c not in pairs.columns]
            if miss:raise KeyError(f'missing numeric contextual covariates: {miss}')
            parts.append(pairs[list(self.numeric_covariates)].apply(pd.to_numeric,errors='coerce').to_numpy(float))
        if fit:self._levels={}
        for c in self.categorical_covariates:
            if c not in pairs.columns:raise KeyError(f'missing categorical contextual covariate: {c}')
            vals=pairs[c].astype(str).fillna('__NA__')
            if fit:self._levels[c]=sorted(vals.unique().tolist())
            lev=self._levels.get(c) if self._levels is not None else None
            if lev is None:raise RuntimeError('context levels not fitted')
            parts.append(np.column_stack([(vals==z).to_numpy(float) for z in lev]))
        return np.hstack(parts),S
    def _fit_alpha(self,pairs,a):
        _,S,T=feature_matrix(pairs,states=self.states);Z,_=self._design(pairs,fit=True);Zs,self._med,self._scale=_standardize_fit(Z);w=patient_sample_weights(pairs.patient_id.astype(str).to_numpy()) if self.patient_balanced else None;self._ridge=Ridge(alpha=float(a)).fit(Zs,T-S,sample_weight=w);return self
    def _pred(self,pairs):
        Z,S=self._design(pairs,fit=False);return normalize_rows(S+self._ridge.predict(_standardize_apply(Z,self._med,self._scale)))
    def _choose_alpha(self,pairs):
        P=pairs.patient_id.astype(str).to_numpy();up=np.unique(P);grid=self.alpha_grid
        if len(grid)==1 or len(up)<4:return grid[0]
        fmap,k=_deterministic_fold_map(P,self.cv_folds);res=[]
        for a in grid:
            pred=np.zeros((len(pairs),len(self.states)))
            for fold in range(k):
                te=np.asarray([fmap[x]==fold for x in P]);tr=~te
                if not te.any():continue
                m=self.fresh();m.alpha_grid=(a,);m._fit_alpha(pairs.loc[tr].reset_index(drop=True),a);pred[te]=m._pred(pairs.loc[te].reset_index(drop=True))
            T=pairs[[f'target__{s}' for s in self.states]].to_numpy(float);pl=np.array([mae(T[P==p],pred[P==p]).mean() for p in up]);res.append((a,float(pl.mean()),float(pl.std(ddof=1)/np.sqrt(len(pl))) if len(pl)>1 else np.inf))
        b=min(res,key=lambda x:x[1]);
        if not self.one_se:return b[0]
        return max([a for a,r,se in res if r<=b[1]+b[2]+1e-12])
    def fit(self,pairs):
        pairs=pairs.reset_index(drop=True).copy();self.selected_alpha=float(self._choose_alpha(pairs));return self._fit_alpha(pairs,self.selected_alpha)
    def predict(self,pairs):
        if self._ridge is None:raise RuntimeError('candidate not fitted')
        return self.validate_predictions(self._pred(pairs),len(pairs))
    def candidate_diagnostics(self):return {'selected_alpha':float(self.selected_alpha),'numeric_covariates':list(self.numeric_covariates),'categorical_covariates':list(self.categorical_covariates)}

class NestedCandidateSelector(CandidatePredictor):
    """Leakage-safe architecture selection from a pre-specified candidate library.

    Selection uses deterministic patient-split OOF risk *only inside fit()*.
    The selected architecture is then refitted on all supplied training patients.
    Outer QualifyOT scoring remains untouched. This is useful as a discovery
    candidate, but changing the library after target inspection is prohibited.
    """
    def __init__(self,states,candidates,*,cv_folds=5,one_se=False,complexity_order=None,name='NestedCandidateSelector'):
        self.states=list(states);self.candidates=list(candidates);self.cv_folds=int(cv_folds);self.one_se=bool(one_se);self.name=name
        if len(self.candidates)<2:raise ValueError('need >=2 candidates')
        self.complexity_order=list(range(len(candidates))) if complexity_order is None else list(complexity_order)
        if sorted(self.complexity_order)!=list(range(len(candidates))):raise ValueError('complexity_order must be a permutation of candidate indices')
        self.selected_index=None;self._model=None;self._diag=None
    def fresh(self):return type(self)(self.states,[c.fresh() for c in self.candidates],cv_folds=self.cv_folds,one_se=self.one_se,complexity_order=self.complexity_order,name=self.name)
    def fit(self,pairs):
        pairs=pairs.reset_index(drop=True).copy();P=pairs.patient_id.astype(str).to_numpy();up=np.unique(P);T=pairs[[f'target__{s}' for s in self.states]].to_numpy(float);fmap,k=_deterministic_fold_map(P,self.cv_folds);scores=[]
        for j,base in enumerate(self.candidates):
            pred=np.zeros((len(pairs),len(self.states)))
            for fold in range(k):
                te=np.asarray([fmap[x]==fold for x in P]);tr=~te
                if not te.any():continue
                m=base.fresh().fit(pairs.loc[tr].reset_index(drop=True));pred[te]=m.predict(pairs.loc[te].reset_index(drop=True))
            pl=np.array([mae(T[P==p],pred[P==p]).mean() for p in up]);scores.append({'index':j,'risk':float(pl.mean()),'se':float(pl.std(ddof=1)/np.sqrt(len(pl))) if len(pl)>1 else np.inf})
        best=min(scores,key=lambda x:x['risk'])
        if self.one_se:
            thr=best['risk']+best['se'];elig={x['index'] for x in scores if x['risk']<=thr+1e-12};idx=next(j for j in self.complexity_order if j in elig)
        else:idx=best['index']
        self.selected_index=int(idx);self._model=self.candidates[idx].fresh().fit(pairs);self._diag={'selected_index':self.selected_index,'selected_candidate':self.candidates[idx].name,'oof_scores':{self.candidates[x['index']].name:x['risk'] for x in scores},'one_se':self.one_se};return self
    def predict(self,pairs):
        if self._model is None:raise RuntimeError('candidate not fitted')
        return self._model.predict(pairs)
    def candidate_diagnostics(self):return copy.deepcopy(self._diag)

class FixedCandidateAverage(CandidatePredictor):
    """Fixed convex average of pre-specified candidates; no data-driven selection."""
    def __init__(self,states,candidates,weights,name='FixedCandidateAverage'):
        self.states=list(states);self.candidates=list(candidates);self.weights=np.asarray(weights,float);self.name=name;self._models=None
        if len(self.candidates)!=len(self.weights) or len(self.candidates)<2:raise ValueError('candidate/weight mismatch')
        if (self.weights<0).any() or not np.isfinite(self.weights).all() or abs(self.weights.sum()-1)>1e-10:raise ValueError('weights must be simplex')
    def fresh(self):return type(self)(self.states,[c.fresh() for c in self.candidates],self.weights.copy(),self.name)
    def fit(self,pairs):self._models=[c.fresh().fit(pairs) for c in self.candidates];return self
    def predict(self,pairs):
        if self._models is None:raise RuntimeError('candidate not fitted')
        P=np.stack([m.predict(pairs) for m in self._models],axis=1);return self.validate_predictions(np.einsum('m,rms->rs',self.weights,P),len(pairs))

class MultiScaleDeltaCandidate(FixedCandidateAverage):
    """Fixed multiscale shrinkage ensemble for low-variance composition prediction.

    Prediction = 0.50 patient-balanced MeanDelta + 0.25 DirectDeltaRidge(alpha=1)
                 + 0.25 DirectDeltaRidge(alpha=100).
    The two ridge scales deliberately bracket weak and strong shrinkage.  The
    weights are fixed, not selected from the evaluated cohort, so the candidate
    removes a single arbitrary ridge penalty without introducing outcome-driven
    hyperparameter search.
    """
    def __init__(self,states,name='MultiScaleDelta'):
        from .candidate_api import MeanDeltaCandidate
        from .candidates import DirectDeltaRidgeCandidate
        st=list(states)
        super().__init__(st,[MeanDeltaCandidate(states=st,patient_balanced=True,name='MeanDelta_PB'),DirectDeltaRidgeCandidate(st,alpha=1,patient_balanced=True,name='DeltaRidge_a1'),DirectDeltaRidgeCandidate(st,alpha=100,patient_balanced=True,name='DeltaRidge_a100')],[.50,.25,.25],name=name)
    def fresh(self):return type(self)(list(self.states),name=self.name)

# ---------------------------------------------------------------------------
# Patient-scale shrinkage integration (principled upgrade)
# ---------------------------------------------------------------------------

def patient_unit_total_weights(patients) -> np.ndarray:
    """Weights with total mass exactly one per physical patient.

    Unlike :func:`model.patient_sample_weights`, these weights are *not* rescaled
    to mean one.  Hence ``sum(weights) == number_of_patients``.  This matters for
    penalized estimators: multiplying every sample weight by a constant leaves a
    weighted mean unchanged but changes the effective Ridge penalty.  The unit-
    total convention makes the regularization scale invariant to the number of
    longitudinal pairs contributed by each patient.
    """
    p=np.asarray(patients).astype(str)
    if len(p)==0:
        return np.array([],float)
    u,c=np.unique(p,return_counts=True)
    counts=dict(zip(u,c))
    return np.asarray([1.0/counts[x] for x in p],float)


@dataclass
class PatientNormalizedDeltaRidgeCandidate(CandidatePredictor):
    """Patient-scale ridge predictor for longitudinal compositions.

    The fitted coefficient matrix minimizes, up to the implementation's common
    factor, ``sum_i mean_t ||Delta p_it - B x_it||^2 + alpha ||B||_F^2``.
    Each physical patient therefore contributes exactly one unit of data-fit
    mass, independent of visit/pair density.  This is distinct from the legacy
    mean-one sample-weight normalization and is intentionally exposed as a new
    candidate rather than silently changing historical results.
    """
    states: tuple|list
    alpha: float=10.0
    name: str='PatientNormalizedDeltaRidge'
    _ridge: object=None
    _median: np.ndarray|None=None
    _scale: np.ndarray|None=None

    def fresh(self):
        return type(self)(states=list(self.states),alpha=float(self.alpha),name=self.name)

    def fit(self,pairs):
        X,S,T=feature_matrix(pairs,states=self.states)
        w=patient_unit_total_weights(pairs.patient_id.astype(str).to_numpy())
        Xs,self._median,self._scale=_patient_weighted_standardize_fit(X,w)
        self._ridge=Ridge(alpha=float(self.alpha)).fit(Xs,T-S,sample_weight=w)
        return self

    def predict(self,pairs):
        if self._ridge is None:
            raise RuntimeError('candidate not fitted')
        X,S=source_feature_matrix(pairs,states=self.states)
        Xs=_standardize_apply(X,self._median,self._scale)
        return self.validate_predictions(normalize_rows(S+self._ridge.predict(Xs)),len(pairs))

    def candidate_diagnostics(self):
        return {'alpha':float(self.alpha),'patient_weight_total':'one_per_physical_patient'}


class FixedAitchisonAverageCandidate(CandidatePredictor):
    """Fixed weighted Aitchison barycenter of simplex-valued base predictions.

    For strictly positive compositions q_m and fixed weights w_m, the prediction
    is closure(exp(sum_m w_m log(q_m))).  The implementation uses an explicit
    epsilon only to define the barycenter on numerical simplex boundaries.
    No evaluated-cohort outcome is used to choose weights.
    """
    def __init__(self,states,candidates,weights,*,eps=1e-8,name='FixedAitchisonAverage'):
        self.states=list(states); self.candidates=list(candidates); self.weights=np.asarray(weights,float)
        self.eps=float(eps); self.name=name; self._models=None
        if len(self.candidates)!=len(self.weights) or len(self.candidates)<2:
            raise ValueError('candidate/weight mismatch')
        if (self.weights<0).any() or not np.isfinite(self.weights).all() or abs(self.weights.sum()-1)>1e-10:
            raise ValueError('weights must be simplex')
        if not np.isfinite(self.eps) or self.eps<=0:
            raise ValueError('eps must be positive')

    def fresh(self):
        return type(self)(self.states,[c.fresh() for c in self.candidates],self.weights.copy(),eps=self.eps,name=self.name)

    def fit(self,pairs):
        self._models=[c.fresh().fit(pairs) for c in self.candidates]
        return self

    def predict(self,pairs):
        if self._models is None:
            raise RuntimeError('candidate not fitted')
        P=np.stack([m.predict(pairs) for m in self._models],axis=1)
        logp=np.log(np.clip(P,self.eps,1.0))
        z=np.einsum('m,rms->rs',self.weights,logp)
        z=z-z.max(axis=1,keepdims=True)
        out=np.exp(z); out=out/out.sum(axis=1,keepdims=True)
        return self.validate_predictions(out,len(pairs))


class PatientScaleDeltaCandidate(FixedCandidateAverage):
    """Fixed patient-normalized shrinkage-path average.

    A low-variance MeanDelta anchor receives half of the mass.  The remaining
    half is spread uniformly over a pre-specified logarithmic Ridge path
    (alpha=1,10,100), fitted with one unit of total weight per physical patient.
    The model therefore integrates over regularization scale rather than
    selecting a cohort-favorable penalty.
    """
    def __init__(self,states,name='PatientScaleDelta'):
        from .candidate_api import MeanDeltaCandidate
        st=list(states)
        bases=[MeanDeltaCandidate(states=st,patient_balanced=True,name='MeanDelta_PB')]
        bases += [PatientNormalizedDeltaRidgeCandidate(st,alpha=a,name=f'PNDeltaRidge_a{a:g}') for a in (1.0,10.0,100.0)]
        super().__init__(st,bases,[.50,1/6,1/6,1/6],name=name)
    def fresh(self): return type(self)(list(self.states),name=self.name)


class AitchisonPatientScaleDeltaCandidate(FixedAitchisonAverageCandidate):
    """Aitchison-geometry counterpart of :class:`PatientScaleDeltaCandidate`."""
    def __init__(self,states,*,eps=1e-8,name='AitchisonPatientScaleDelta'):
        from .candidate_api import MeanDeltaCandidate
        st=list(states)
        bases=[MeanDeltaCandidate(states=st,patient_balanced=True,name='MeanDelta_PB')]
        bases += [PatientNormalizedDeltaRidgeCandidate(st,alpha=a,name=f'PNDeltaRidge_a{a:g}') for a in (1.0,10.0,100.0)]
        super().__init__(st,bases,[.50,1/6,1/6,1/6],eps=eps,name=name)
    def fresh(self): return type(self)(list(self.states),eps=self.eps,name=self.name)


class EvidenceTemperedDeltaCandidate(CandidatePredictor):
    """Patient-evidence-tempered multiscale delta ensemble.

    The candidate family and prior mixture are frozen in advance.  Inside each
    ``fit`` call, patient-split out-of-fold predictions define a patient-equal
    MAE surface over convex weights.  We first find the best training-only
    convex stack, then choose the mixture *closest to the fixed low-capacity
    prior* whose OOF risk remains within one patient-level standard error of
    that best stack.  Thus adaptation is permitted only to the extent justified
    by training-patient evidence; no outer held-out target can affect weights.

    This is a continuous analogue of a one-standard-error rule and deliberately
    avoids selecting one Ridge scale as a winner.
    """
    def __init__(self,states,*,cv_folds=5,prior_weights=(.50,1/6,1/6,1/6),name='EvidenceTemperedDelta'):
        from .candidate_api import MeanDeltaCandidate
        self.states=list(states); self.cv_folds=int(cv_folds); self.name=name
        self.prior_weights=np.asarray(prior_weights,float)
        if self.prior_weights.shape!=(4,) or (self.prior_weights<=0).any() or abs(self.prior_weights.sum()-1)>1e-10:
            raise ValueError('prior_weights must be four strictly-positive simplex weights')
        self.base_candidates=[MeanDeltaCandidate(states=self.states,patient_balanced=True,name='MeanDelta_PB')]
        self.base_candidates += [PatientNormalizedDeltaRidgeCandidate(self.states,alpha=a,name=f'PNDeltaRidge_a{a:g}') for a in (1.0,10.0,100.0)]
        self.selected_weights=None; self._models=None; self._diag=None

    def fresh(self):
        return type(self)(list(self.states),cv_folds=self.cv_folds,prior_weights=tuple(self.prior_weights),name=self.name)

    @staticmethod
    def _softmax_param(z):
        z=np.asarray(z,float); z=z-z.max(); e=np.exp(z); return e/e.sum()

    @staticmethod
    def _inv_softmax(w):
        w=np.asarray(w,float); return np.log(np.clip(w,1e-12,None))

    def fit(self,pairs):
        from scipy.optimize import minimize
        pairs=pairs.reset_index(drop=True).copy(); P=pairs.patient_id.astype(str).to_numpy(); up=np.unique(P)
        if len(up)<4:
            self.selected_weights=self.prior_weights.copy(); self._models=[c.fresh().fit(pairs) for c in self.base_candidates]
            self._diag={'weights':self.selected_weights.tolist(),'fallback':'too_few_patients','patients':int(len(up))}
            return self
        T=pairs[[f'target__{s}' for s in self.states]].to_numpy(float)
        fmap,k=_deterministic_fold_map(P,self.cv_folds); M=len(self.base_candidates); oof=np.zeros((len(pairs),M,len(self.states)),float)
        for j,base in enumerate(self.base_candidates):
            for fold in range(k):
                te=np.asarray([fmap[x]==fold for x in P]); tr=~te
                if not te.any(): continue
                m=base.fresh().fit(pairs.loc[tr].reset_index(drop=True))
                oof[te,j]=m.predict(pairs.loc[te].reset_index(drop=True))
        if np.any(oof.sum(axis=2)<=0):
            raise RuntimeError('incomplete OOF predictions')

        def patient_losses(w):
            pred=np.einsum('m,rms->rs',w,oof)
            row=mae(T,pred)
            return np.asarray([float(row[P==p].mean()) for p in up],float)
        def risk_z(z):
            return float(patient_losses(self._softmax_param(z)).mean())

        z0=self._inv_softmax(self.prior_weights)
        opt=minimize(risk_z,z0,method='BFGS',options={'maxiter':300,'gtol':1e-9})
        wbest=self._softmax_param(opt.x if opt.success and np.isfinite(opt.fun) else z0)
        plbest=patient_losses(wbest); rbest=float(plbest.mean())
        sebest=float(plbest.std(ddof=1)/np.sqrt(len(plbest))) if len(plbest)>1 else 0.0
        threshold=rbest+sebest

        pi=self.prior_weights.copy()
        def kl_z(z):
            w=self._softmax_param(z)
            return float(np.sum(w*np.log(np.clip(w/pi,1e-12,None))))
        def constraint(z):
            w=self._softmax_param(z)
            return float(threshold-patient_losses(w).mean())
        opt2=minimize(kl_z,z0,method='SLSQP',constraints=[{'type':'ineq','fun':constraint}],options={'maxiter':500,'ftol':1e-12})
        if opt2.success and constraint(opt2.x)>=-1e-8:
            chosen=self._softmax_param(opt2.x); status='evidence_tempered'
        else:
            # The fixed prior is a deterministic conservative fallback.
            chosen=pi; status='prior_fallback'
        self.selected_weights=chosen
        self._models=[c.fresh().fit(pairs) for c in self.base_candidates]
        self._diag={'weights':chosen.tolist(),'prior_weights':pi.tolist(),'best_weights':wbest.tolist(),
                    'best_oof_risk':rbest,'best_oof_se':sebest,'one_se_threshold':threshold,
                    'chosen_oof_risk':float(patient_losses(chosen).mean()),'selection_status':status,
                    'optimizer_best_success':bool(opt.success),'optimizer_tempered_success':bool(opt2.success)}
        return self

    def predict(self,pairs):
        if self._models is None or self.selected_weights is None:
            raise RuntimeError('candidate not fitted')
        P=np.stack([m.predict(pairs) for m in self._models],axis=1)
        out=np.einsum('m,rms->rs',self.selected_weights,P)
        return self.validate_predictions(out,len(pairs))

    def candidate_diagnostics(self):
        return copy.deepcopy(self._diag)

class PatientMultiScaleDeltaCandidate(FixedCandidateAverage):
    """Pair-density-invariant multiscale delta candidate.

    This is the patient-scale refinement of ``MultiScaleDeltaCandidate``:

        0.50 * MeanDelta
      + 0.25 * PatientNormalizedDeltaRidge(alpha=1)
      + 0.25 * PatientNormalizedDeltaRidge(alpha=100).

    With exactly one longitudinal pair per patient it is algebraically identical
    to the fixed MultiScaleDelta construction.  With unequal or duplicated pair
    counts, each physical patient retains one unit of total Ridge fitting mass,
    so predictions are invariant to uniform replication of within-patient rows.
    The construction therefore aligns candidate fitting with QualifyOT's
    patient-equal inferential unit without changing the evidence engine.
    """
    def __init__(self,states,name='PatientMultiScaleDelta'):
        from .candidate_api import MeanDeltaCandidate
        st=list(states)
        bases=[
            MeanDeltaCandidate(states=st,patient_balanced=True,name='MeanDelta_PB'),
            PatientNormalizedDeltaRidgeCandidate(st,alpha=1.0,name='PNDeltaRidge_a1'),
            PatientNormalizedDeltaRidgeCandidate(st,alpha=100.0,name='PNDeltaRidge_a100'),
        ]
        super().__init__(st,bases,[.50,.25,.25],name=name)
    def fresh(self): return type(self)(list(self.states),name=self.name)

class PatientScaleBridgeCandidate(CandidatePredictor):
    """Training-only safe bridge from PatientMultiScaleDelta to an adaptive ridge.

    The fixed PatientMultiScaleDelta predictor is the anchor. A more adaptive
    patient-normalized Ridge(alpha=1) supplies a residual direction. For
    lambda in a fixed grid, out-of-fold training-patient losses are compared
    with the anchor. Only weights whose simultaneous patient-bootstrap upper
    bound on harm is <= ``nonharm_margin`` are eligible. Among eligible weights
    the empirically lowest-risk weight is chosen. If no positive weight is
    supported, lambda=0 exactly recovers the anchor.

    This selection is entirely inside ``fit`` and is therefore nested again by
    QualifyOT's outer evidence engine. It is an algorithmic complexity-control
    device, not an additional qualification gate.
    """
    def __init__(self,states,*,cv_folds=5,grid_step=.10,selection_bootstrap=300,
                 seed=20260831,nonharm_margin=0.0,name='PatientScaleBridge'):
        self.states=list(states); self.cv_folds=int(cv_folds); self.grid_step=float(grid_step)
        self.selection_bootstrap=int(selection_bootstrap); self.seed=int(seed)
        self.nonharm_margin=float(nonharm_margin); self.name=name
        if self.grid_step<=0 or self.grid_step>1 or abs(round(1/self.grid_step)*self.grid_step-1)>1e-8:
            raise ValueError('grid_step must divide 1')
        self.selected_weight=None; self._anchor=None; self._adaptive=None; self._diag=None

    def fresh(self):
        return type(self)(self.states,cv_folds=self.cv_folds,grid_step=self.grid_step,
                          selection_bootstrap=self.selection_bootstrap,seed=self.seed,
                          nonharm_margin=self.nonharm_margin,name=self.name)

    def fit(self,pairs):
        from .weight_selection import patient_curves, one_se_msw
        pairs=pairs.reset_index(drop=True).copy(); P=pairs.patient_id.astype(str).to_numpy(); up=np.unique(P)
        anchor_proto=PatientMultiScaleDeltaCandidate(self.states)
        adaptive_proto=PatientNormalizedDeltaRidgeCandidate(self.states,alpha=1.0,name='PNDeltaRidge_a1')
        if len(up)<4:
            self.selected_weight=0.0; self._anchor=anchor_proto.fit(pairs); self._adaptive=adaptive_proto.fit(pairs)
            self._diag={'selected_weight':0.0,'fallback':'too_few_patients'}; return self
        fmap,k=_deterministic_fold_map(P,self.cv_folds)
        T=pairs[[f'target__{s}' for s in self.states]].to_numpy(float)
        A=np.zeros_like(T); B=np.zeros_like(T)
        for fold in range(k):
            te=np.asarray([fmap[x]==fold for x in P]); tr=~te
            if not te.any(): continue
            trdf=pairs.loc[tr].reset_index(drop=True); tedf=pairs.loc[te].reset_index(drop=True)
            A[te]=anchor_proto.fresh().fit(trdf).predict(tedf)
            B[te]=adaptive_proto.fresh().fit(trdf).predict(tedf)
        grid=np.round(np.arange(0,1.0000001,self.grid_step),10)
        _,D,R=patient_curves(T,A,B,P,grid)
        sel=one_se_msw(D,R,grid,alpha=.05,delta=self.nonharm_margin,
                       B=max(50,self.selection_bootstrap),seed=self.seed,lam=0.0)
        self.selected_weight=float(sel['weight'])
        self._anchor=anchor_proto.fit(pairs); self._adaptive=adaptive_proto.fit(pairs)
        self._diag={'selected_weight':self.selected_weight,'best_weight':float(sel['best_weight']),
                    'max_safe_weight':float(sel['max_eligible_weight']),'max_t_q':float(sel['max_t_q']),
                    'selection_bootstrap':int(max(50,self.selection_bootstrap)),
                    'nonharm_margin':self.nonharm_margin}
        return self

    def predict(self,pairs):
        if self._anchor is None or self._adaptive is None or self.selected_weight is None:
            raise RuntimeError('candidate not fitted')
        a=self._anchor.predict(pairs); b=self._adaptive.predict(pairs); w=self.selected_weight
        return self.validate_predictions((1-w)*a+w*b,len(pairs))

    def candidate_diagnostics(self): return copy.deepcopy(self._diag)
