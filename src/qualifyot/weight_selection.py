from __future__ import annotations
import numpy as np
from .model import mae

def patient_curves(target,null,prior,patients,grid):
    pats=np.asarray(patients).astype(str); up=np.unique(pats); D=np.zeros((len(up),len(grid))); R=np.zeros_like(D)
    for j,weight in enumerate(grid):
        pred=(1-weight)*null+weight*prior; l=mae(target,pred); l0=mae(target,null)
        for i,p in enumerate(up): D[i,j]=(l[pats==p]-l0[pats==p]).mean(); R[i,j]=l[pats==p].mean()
    return up,D,R

def simultaneous_ucb(D,alpha=.05,B=5000,seed=0,batch=250):
    D=np.asarray(D,float); n,k=D.shape; mean=D.mean(0); se=D.std(0,ddof=1)/np.sqrt(max(n,1)); valid=se>1e-14
    rng=np.random.default_rng(seed); mx=[]
    for st in range(0,B,batch):
        m=min(batch,B-st); idx=rng.integers(0,n,size=(m,n)); mb=D[idx].mean(1); t=np.zeros_like(mb); t[:,valid]=(mb[:,valid]-mean[valid])/se[valid]; mx.extend(np.max(t[:,valid],axis=1) if valid.any() else np.zeros(m))
    q=float(np.quantile(mx,1-alpha)); u=mean+q*se; u[0]=0; return u,se,q

def one_se_msw(D,R,grid,alpha=.05,delta=0,B=5000,seed=0,lam=1):
    u,se,q=simultaneous_ucb(D,alpha,B,seed); feasible=u<=delta+1e-12; feasible[0]=True; ids=np.flatnonzero(feasible); mr=R.mean(0)
    best=ids[np.argmin(mr[ids])]; sebest=R[:,best].std(ddof=1)/np.sqrt(R.shape[0]); eligible=ids[mr[ids]<=mr[best]+lam*sebest+1e-15]; chosen=eligible[0]
    return {'weight':float(grid[chosen]),'best_weight':float(grid[best]),'max_eligible_weight':float(grid[ids[-1]]),'se_best':float(sebest),'max_t_q':q,'ucb':u,'risk':mr}
