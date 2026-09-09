from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize

from qualifyot.candidate_api import CandidatePredictor
from qualifyot.model import feature_matrix, mae
from qualifyot.inference import honest_confirm_predictions

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'theory_validation'
OUT.mkdir(parents=True,exist_ok=True)
STATES=['HSPC','Prog','Ery-early','Ery-late','Mega']
MASK=np.array([[1,1,0,0,0],[0,1,1,0,1],[0,0,1,1,0],[0,0,0,1,0],[0,0,0,0,1]],bool)
TBASE=np.array([[.55,.45,0,0,0],[0,.30,.66,0,.04],[0,0,.71,.29,0],[0,0,0,1,0],[0,0,0,0,1]],float)

class GraphTransitionCandidate(CandidatePredictor):
    name='GraphTransition'; states=STATES
    def __init__(self,mask=MASK,reg=1e-4): self.mask=np.asarray(mask,bool); self.reg=float(reg); self.T_=None
    def fresh(self): return type(self)(self.mask.copy(),self.reg)
    def _unpack(self,z):
        T=np.zeros_like(self.mask,dtype=float); pos=0
        for r in range(len(STATES)):
            idx=np.flatnonzero(self.mask[r])
            if len(idx)==1: T[r,idx[0]]=1
            else:
                zz=z[pos:pos+len(idx)]; pos+=len(idx); zz-=zz.max(); e=np.exp(zz); T[r,idx]=e/e.sum()
        return T
    def fit(self,pairs):
        _,S,Y=feature_matrix(pairs,states=self.states)
        dim=sum(int(self.mask[r].sum()) for r in range(len(STATES)) if self.mask[r].sum()>1)
        def obj(z):
            T=self._unpack(z); pred=S@T
            return float(np.mean((pred-Y)**2)+self.reg*np.mean(z*z))
        res=minimize(obj,np.zeros(dim),method='L-BFGS-B',options={'maxiter':150,'ftol':1e-11,'gtol':1e-7})
        if not np.isfinite(res.fun): raise RuntimeError('transition fit failed')
        self.T_=self._unpack(res.x); return self
    def predict(self,pairs):
        if self.T_ is None: raise RuntimeError('not fitted')
        _,S,_=feature_matrix(pairs,states=self.states); p=S@self.T_; p=np.clip(p,0,None); p/=p.sum(1,keepdims=True)
        return self.validate_predictions(p,len(pairs))

def transition(signal):
    I=np.eye(5); return (1-signal)*I+signal*TBASE

def perturb_T(rng,base,conc=250):
    out=np.zeros_like(base)
    for r in range(5):
        idx=np.flatnonzero(MASK[r])
        if len(idx)==1: out[r,idx[0]]=1
        else:
            a=np.maximum(base[r,idx]*conc,.5); out[r,idx]=rng.dirichlet(a)
    return out

def simulate(seed,n,signal,cells=800):
    rng=np.random.default_rng(seed); base=transition(signal); rows=[]
    for i in range(n):
        p0=rng.dirichlet([20,5,1.5,.6,.8]); Ti=perturb_T(rng,base,250)
        latent=[p0]
        for _ in range(3): latent.append(latent[-1]@Ti)
        obs=[rng.multinomial(cells,p)/cells for p in latent]
        for t in range(3):
            r={'patient_id':f'P{i:03d}','pair_id':f'P{i:03d}_{t}','source_time':str(t),'target_time':str(t+1)}
            for k,s in enumerate(STATES): r[f'source__{s}']=obs[t][k]; r[f'target__{s}']=obs[t+1][k]
            rows.append(r)
    return pd.DataFrame(rows)

def split_ids(n):
    nd=int(round(.6*n)); return {f'P{i:03d}' for i in range(nd)}, {f'P{i:03d}' for i in range(nd,n)}

def patient_equal_risk(y,pred,pats):
    row=mae(y,pred); return float(np.mean([row[pats==q].mean() for q in np.unique(pats)]))

def run_one(seed,n,signal):
    df=simulate(seed,n,signal); dev_ids,conf_ids=split_ids(n)
    dev=df[df.patient_id.isin(dev_ids)].reset_index(drop=True); conf=df[df.patient_id.isin(conf_ids)].reset_index(drop=True)
    cand=GraphTransitionCandidate().fit(dev)
    _,Sd,Td=feature_matrix(dev,states=STATES); Pd=dev.patient_id.astype(str).to_numpy(); cp_dev=cand.predict(dev); ref_dev=Sd.copy()
    grid=np.linspace(0,1,101); risks=[]
    for lam in grid: risks.append(patient_equal_risk(Td,(1-lam)*ref_dev+lam*cp_dev,Pd))
    lam=float(grid[int(np.argmin(risks))])  # deliberately adaptive development-only choice
    _,Sc,Tc=feature_matrix(conf,states=STATES); Pc=conf.patient_id.astype(str).to_numpy(); cp=cand.predict(conf); ref=Sc.copy(); ret=(1-lam)*ref+lam*cp
    res,_=honest_confirm_predictions(Tc,Sc,ref,cp,ret,Pc,movement_margin=.01,alpha=.05)
    # Naive same-confirmation one-sided t IUT, for power/calibration comparison only.
    from scipy.stats import t as student_t
    from qualifyot.model import hellinger
    pats=np.unique(Pc)
    def collapse(v): return np.array([v[Pc==q].mean() for q in pats],float)
    gm=collapse(hellinger(cp,Sc)-.01*hellinger(Tc,Sc)); u=collapse(mae(Tc,ref)-mae(Tc,cp)); rr=collapse(mae(Tc,ref)-mae(Tc,ret))
    def tlcb(x):
        sd=x.std(ddof=1); return float(x.mean() if sd<1e-15 else x.mean()-student_t.ppf(.95,len(x)-1)*sd/np.sqrt(len(x)))
    naive=bool(tlcb(gm)>0 and tlcb(u)>0 and tlcb(rr)>0)
    return dict(n=n,signal=signal,lambda_selected=lam,confirmation_patients=len(conf_ids),honest_qualified=res.qualified,naive_t_qualified=naive,
                movement=res.movement.estimate,movement_lcb=res.movement.lower_bound,utility=res.utility.estimate,utility_lcb=res.utility.lower_bound,
                retention=res.retention.estimate,retention_lcb=res.retention.lower_bound)

def main(reps=100):
    rows=[]
    for n in [30,60]:
        for signal in [0.0,.25,.5]:
            for r in range(reps):
                rows.append(run_one(20261600+n*1000+int(signal*100)*100+r,n,signal))
            sub=pd.DataFrame(rows)[lambda x:(x.n==n)&(x.signal==signal)]
            print(n,signal,'honest',sub.honest_qualified.mean(),'naive',sub.naive_t_qualified.mean(),flush=True)
    d=pd.DataFrame(rows); d.to_csv(OUT/'E14_full_pipeline_honest_confirmation.csv',index=False)
    s=d.groupby(['n','signal']).agg(reps=('honest_qualified','size'),honest_qualification_rate=('honest_qualified','mean'),naive_t_qualification_rate=('naive_t_qualified','mean'),mean_lambda=('lambda_selected','mean'),mean_utility=('utility','mean'),mean_retention=('retention','mean')).reset_index()
    s.to_csv(OUT/'E14_full_pipeline_honest_confirmation_summary.csv',index=False)
if __name__=='__main__': main()
