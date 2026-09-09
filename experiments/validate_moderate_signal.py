from __future__ import annotations
import time
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import minimize
from qualifyot.inference import honest_confirm_predictions

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results'/'inference'; OUT.mkdir(parents=True,exist_ok=True)
STATES=['HSPC','Prog','Ery-early','Ery-late','Mega']; K=5
I=np.eye(K)
TBASE=np.array([[.55,.45,0,0,0],[0,.30,.66,0,.04],[0,0,.71,.29,0],[0,0,0,1,0],[0,0,0,0,1]],float)
M_EXACT=TBASE>0
M_EXTRA=M_EXACT.copy(); M_EXTRA[0,4]=True
M_MISS_MINOR=M_EXACT.copy(); M_MISS_MINOR[1,4]=False
M_MISS_MAJOR=M_EXACT.copy(); M_MISS_MAJOR[1,2]=False
M_REVERSE=M_MISS_MAJOR.copy(); M_REVERSE[2,1]=True
MASKS={'exact':M_EXACT,'extra':M_EXTRA,'missing_minor':M_MISS_MINOR,'missing_major':M_MISS_MAJOR,'reverse_major':M_REVERSE}

def perturb_T(rng,T,mask,conc=300):
    out=np.zeros_like(T)
    for r in range(K):
        idx=np.flatnonzero(mask[r]); vals=T[r,idx]
        if len(idx)==1: out[r,idx[0]]=1
        else: out[r,idx]=rng.dirichlet(np.maximum(vals*conc,.3))
    return out

def simulate(rng,n,signal,cells=1000):
    T=(1-signal)*I+signal*TBASE
    rows=[]
    for i in range(n):
        p=rng.dirichlet([20,5,1.5,.6,.8]); Ti=perturb_T(rng,T,T>0,conc=300)
        obs=[]
        for t in range(4):
            c=rng.multinomial(cells,p); obs.append(c/c.sum()); p=p@Ti
        for t in range(3): rows.append((i,obs[t],obs[t+1]))
    return rows

def fit_transition(rows,mask):
    S=np.vstack([r[1] for r in rows]); Y=np.vstack([r[2] for r in rows]); dims=[]
    for rr in range(K):
        idx=np.flatnonzero(mask[rr]);
        if len(idx)>1: dims.append((rr,idx))
    D=sum(len(idx) for _,idx in dims); z0=np.zeros(D)
    def unpack(z):
        T=np.zeros((K,K)); pos=0
        for rr in range(K):
            idx=np.flatnonzero(mask[rr])
            if len(idx)==1: T[rr,idx[0]]=1
            else:
                zz=z[pos:pos+len(idx)]; pos+=len(idx); zz=zz-zz.max(); e=np.exp(zz); T[rr,idx]=e/e.sum()
        return T
    def obj(z):
        T=unpack(z); P=S@T; return np.mean((P-Y)**2)+1e-5*np.mean(z*z)
    res=minimize(obj,z0,method='L-BFGS-B',options={'maxiter':120,'ftol':1e-10,'gtol':1e-7})
    if not np.isfinite(res.fun): raise RuntimeError('non-finite transition fit')
    return unpack(res.x)

def arrays(rows,T):
    pats=np.array([f'p{r[0]}' for r in rows]); S=np.vstack([r[1] for r in rows]); Y=np.vstack([r[2] for r in rows]); C=S@T; C=np.clip(C,0,None); C/=C.sum(1,keepdims=True); return pats,S,Y,C

def choose_lambda(rows,T):
    pats,S,Y,C=arrays(rows,T); grid=np.linspace(0,1,21); vals=[]
    for lam in grid:
        P=(1-lam)*S+lam*C; loss=np.abs(P-Y).mean(1); up=np.unique(pats); vals.append(np.mean([loss[pats==p].mean() for p in up]))
    return float(grid[int(np.argmin(vals))])

def run(reps=30,seed=20260907):
    rng=np.random.default_rng(seed); rows=[]
    for n in [15,30,60]:
      for signal in [0,.25,.5,.75,1.0]:
       for rep in range(reps):
        dev=simulate(rng,n,signal); conf=simulate(rng,n,signal)
        for name,mask in MASKS.items():
            T=fit_transition(dev,mask); lam=choose_lambda(dev,T); pats,S,Y,C=arrays(conf,T); R=S.copy(); Ret=(1-lam)*R+lam*C
            out,_=honest_confirm_predictions(Y,S,R,C,Ret,pats,movement_margin=.01,alpha=.05)
            rows.append(dict(n=n,signal=signal,graph=name,rep=rep,lambda_design=lam,qualified=out.qualified,movement_lcb=out.movement.lower_bound,utility_lcb=out.utility.lower_bound,retention_lcb=out.retention.lower_bound,utility=out.utility.estimate,retention=out.retention.estimate))
    d=pd.DataFrame(rows); d.to_csv(OUT/'E7_moderate_signal_runs.csv',index=False); s=d.groupby(['n','signal','graph']).agg(reps=('qualified','size'),qualification_rate=('qualified','mean'),mean_lambda=('lambda_design','mean'),mean_utility=('utility','mean'),mean_retention=('retention','mean'),movement_pass=('movement_lcb',lambda x:(x>0).mean()),utility_pass=('utility_lcb',lambda x:(x>0).mean()),retention_pass=('retention_lcb',lambda x:(x>0).mean())).reset_index(); s.to_csv(OUT/'E7_moderate_signal_summary.csv',index=False); return d,s
if __name__=='__main__':
 t=time.time(); d,s=run(); print('seconds',time.time()-t,'rows',len(d)); print(s.to_string(index=False))
