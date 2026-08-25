from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import re
import numpy as np
from scipy.optimize import linprog, minimize, Bounds, LinearConstraint
from sklearn.linear_model import Ridge
from .utils import normalize_rows, patient_equal

STATES=['Memory','Effector','Exhausted','Regulatory','Other_T']
EDGES=[('Memory','Effector'),('Effector','Exhausted'),('Memory','Exhausted')]


def incidence(states=STATES, edges=EDGES):
    """Directed incidence matrix with -1 at edge tail and +1 at edge head."""
    B=np.zeros((len(states),len(edges)),float); idx={s:i for i,s in enumerate(states)}
    for j,(a,b) in enumerate(edges):
        if a not in idx or b not in idx:
            raise KeyError(f'edge {(a,b)} uses a state outside the configured state space')
        if a==b:
            raise ValueError('self-loops are not permitted in the pre-specified flow graph')
        B[idx[a],j]=-1.0; B[idx[b],j]=1.0
    return B

B=incidence()


def graph_hash(states=STATES,edges=EDGES):
    payload=json.dumps({'states':list(states),'edges':[list(e) for e in edges]},sort_keys=True,separators=(',',':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def mae(y,p): return np.abs(y-p).mean(1)
def hellinger(y,p): return np.sqrt(np.maximum(0,1-np.sqrt(np.clip(y,0,1)*np.clip(p,0,1)).sum(1)))
def tv(y,p): return .5*np.abs(y-p).sum(1)
def js(y,p):
    y=np.clip(y,1e-15,1); p=np.clip(p,1e-15,1); m=.5*(y+p)
    kl1=np.sum(y*np.log2(y/m),1); kl2=np.sum(p*np.log2(p/m),1); return np.sqrt(np.maximum(0,.5*(kl1+kl2)))


def patient_sample_weights(patients):
    """Pair weights whose total is exactly one per physical patient.

    The absolute normalization is immaterial to weighted Ridge/means; we scale
    the weights to have mean one so regularization magnitudes remain comparable
    to pair-equal fits while each physical patient contributes equal total mass.
    """
    p=np.asarray(patients).astype(str)
    if len(p)==0: return np.array([],float)
    u,c=np.unique(p,return_counts=True); n=dict(zip(u,c))
    w=np.array([1.0/n[x] for x in p],float)
    return w / w.mean()


def weighted_mean_rows(x, weights=None):
    x=np.asarray(x,float)
    if weights is None: return x.mean(0)
    w=np.asarray(weights,float)
    if len(w)!=len(x): raise ValueError('weights length mismatch')
    if not np.isfinite(w).all() or (w<0).any() or w.sum()<=0: raise ValueError('invalid weights')
    return np.average(x,axis=0,weights=w)


def residual_flow(source,target,tiny=1e-6,Bmat=None):
    """Residual graph-flow decomposition used by the primary external candidate.

    It solves min tiny*1^T f + ||r||_1 subject to B f + r = target-source,
    f>=0.  This is deliberately *not* an exact-flow assumption: states outside
    the directed graph may change through the residual term r.  The returned
    GCF is a compatibility diagnostic, not a biological-validity score.
    """
    Buse=B if Bmat is None else np.asarray(Bmat,float)
    source=np.asarray(source,float); target=np.asarray(target,float)
    d=target-source; k=len(source); m=Buse.shape[1]
    if Buse.shape[0]!=k: raise ValueError('B rows must match composition dimension')
    c=np.r_[tiny*np.ones(m),np.ones(k),np.ones(k)]
    A=np.c_[Buse,np.eye(k),-np.eye(k)]
    res=linprog(c,A_eq=A,b_eq=d,bounds=[(0,None)]*(m+2*k),method='highs')
    if not res.success: raise RuntimeError(res.message)
    f=np.maximum(res.x[:m],0); r=res.x[m:m+k]-res.x[m+k:]
    gcf=1-np.abs(r).sum()/max(np.abs(d).sum(),1e-15)
    return f,r,float(np.clip(gcf,0,1))


def project_flow(source,fraw,Bmat=None):
    """Euclidean projection in flow space with simplex feasibility enforced."""
    Buse=B if Bmat is None else np.asarray(Bmat,float)
    source=np.asarray(source,float); fraw=np.maximum(np.asarray(fraw,float),0)
    if Buse.shape[0]!=len(source) or Buse.shape[1]!=len(fraw): raise ValueError('flow/incidence shape mismatch')
    raw=source+Buse@fraw
    if raw.min()>=-1e-10: return fraw,raw
    def fun(f): return .5*np.sum((f-fraw)**2)
    res=minimize(fun,fraw,method='SLSQP',bounds=Bounds(0,np.inf),constraints=[LinearConstraint(Buse,-source,np.inf)],options={'ftol':1e-11,'maxiter':500})
    if not res.success: raise RuntimeError('flow projection failed '+res.message)
    f=np.maximum(res.x,0); p=source+Buse@f
    if p.min()<-1e-7 or abs(p.sum()-1)>1e-7: raise RuntimeError('simplex closure failed')
    # Numerical cleanup only; constraints already imply the simplex.
    p=np.maximum(p,0); p=p/p.sum(); return f,p


@dataclass
class GraphFlowModel:
    ridge: Ridge
    x_mean: np.ndarray
    x_scale: np.ndarray
    Bmat: np.ndarray
    edges: tuple
    patient_balanced: bool=False
    def predict(self,X,source):
        X=np.asarray(X,float); source=np.asarray(source,float)
        Z=(X-self.x_mean)/self.x_scale
        F=np.asarray(self.ridge.predict(Z),float)
        # sklearn returns shape (n_samples,) for a one-output Ridge; normalize
        # to (n_samples, 1) so single-edge graphs obey the same flow contract.
        if F.ndim==1: F=F[:,None]
        F=np.maximum(F,0); out=[]
        for s,f in zip(source,F): out.append(project_flow(s,f,self.Bmat)[1])
        return np.vstack(out)


def fit_graphflow(X,source,target,alpha=1.0,flow_labels=None,Bmat=None,edges=None,patients=None,patient_balanced=False):
    """Fit the predictive flow head to residual-flow labels.

    The label for a row is f* from the residual program B f + r = delta p.
    Held-out labels must never be passed to this routine.  If patient_balanced
    is True, every physical patient has equal total sample weight regardless
    of how many longitudinal pairs they contribute.
    """
    X=np.asarray(X,float); source=np.asarray(source,float); target=np.asarray(target,float)
    Buse=B if Bmat is None else np.asarray(Bmat,float)
    edges_tuple=tuple(EDGES if edges is None else tuple(tuple(e) for e in edges))
    flows=np.asarray(flow_labels,float) if flow_labels is not None else np.vstack([residual_flow(s,t,Bmat=Buse)[0] for s,t in zip(source,target)])
    if len(flows)!=len(X): raise ValueError('flow_labels length mismatch')
    if flows.shape[1]!=Buse.shape[1]: raise ValueError('flow_labels width mismatch')
    mu=np.nanmedian(X,0); X=np.where(np.isfinite(X),X,mu); scale=np.std(X,0); scale=np.where(scale>1e-8,scale,1)
    z=(X-mu)/scale
    sample_weight=None
    if patient_balanced:
        if patients is None: raise ValueError('patients are required for patient-balanced fitting')
        sample_weight=patient_sample_weights(patients)
    ridge=Ridge(alpha=alpha,fit_intercept=True).fit(z,flows,sample_weight=sample_weight)
    return GraphFlowModel(ridge,mu,scale,Buse.copy(),edges_tuple,bool(patient_balanced))


def _time_value(x):
    s=str(x).strip().lower().replace('_',' ').replace('-',' ')
    aliases={'pre':0.0,'baseline':0.0,'base':0.0,'i':0.0,'ii':1.0,'iii':2.0,'iv':3.0,'post':1.0}
    if s in aliases: return aliases[s]
    m=re.search(r'(?:follow\s*up|fu|time|visit|t)\s*(\d+)',s)
    if m: return float(m.group(1))
    try: return float(s)
    except Exception: return float('nan')


def feature_matrix(pairs,states=None):
    states=list(STATES if states is None else states)
    S=pairs[[f'source__{s}' for s in states]].to_numpy(float)
    t0=pairs.source_time.map(_time_value).astype(float)
    t1=pairs.target_time.map(_time_value).astype(float)
    t0=t0.fillna(0.0); fallback=t0+1.0; t1=t1.where(t1.notna(),fallback)
    dt=(t1.to_numpy(float)-t0.to_numpy(float))[:,None]
    if (dt < -1e-12).any(): raise ValueError('target_time precedes source_time in feature_matrix')
    return np.c_[S,dt],S,pairs[[f'target__{s}' for s in states]].to_numpy(float)


def fit_null(kind,source,target,patients=None,patient_balanced=False):
    source=np.asarray(source,float); target=np.asarray(target,float)
    weights=patient_sample_weights(patients) if patient_balanced else None
    if patient_balanced and patients is None: raise ValueError('patients are required for patient-balanced null fitting')
    if kind=='Persistence': return {'kind':kind,'patient_balanced':bool(patient_balanced)}
    if kind=='CohortMean': return {'kind':kind,'center':weighted_mean_rows(target,weights),'patient_balanced':bool(patient_balanced)}
    if kind=='MeanDelta': return {'kind':kind,'delta':weighted_mean_rows(target-source,weights),'patient_balanced':bool(patient_balanced)}
    raise KeyError(kind)


def predict_null(model,source):
    k=model['kind']
    if k=='Persistence': return source.copy()
    if k=='CohortMean': return np.repeat(model['center'][None,:],len(source),0)
    if k=='MeanDelta': return normalize_rows(source+model['delta'])
    raise KeyError(k)


def choose_null_loocv(source,target,patients,patient_balanced=False):
    kinds=['Persistence','CohortMean','MeanDelta']; scores={}; patients=np.asarray(patients); up=np.unique(patients.astype(str))
    for kind in kinds:
        vals=[]; pats=[]
        for p in up:
            te=patients.astype(str)==p; tr=~te
            m=fit_null(kind,source[tr],target[tr],patients=patients[tr],patient_balanced=patient_balanced)
            vals.extend(mae(target[te],predict_null(m,source[te]))); pats.extend([p]*te.sum())
        scores[kind]=patient_equal(np.asarray(vals),np.asarray(pats))
    best=min(kinds,key=lambda k:(round(scores[k],12),kinds.index(k)))
    return best,scores
