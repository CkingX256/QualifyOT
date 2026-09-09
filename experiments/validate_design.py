from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np,pandas as pd
from qualifyot.safe_averaging import honest_safe_graph_average,safe_graph_average
from qualifyot.inference import nearest_grid_risk_regret_bound

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results'/'inference'; OUT.mkdir(parents=True,exist_ok=True)

def e5_lambda_grid(reps=2000,seed=20260905):
    rng=np.random.default_rng(seed); rows=[]
    Klist=[3,5,9]; hs=[.02,.01,.005,.0025]
    dense=np.linspace(0,1,4001)
    for K in Klist:
      for h in hs:
        bound=nearest_grid_risk_regret_bound(grid_step=h,n_states=K); violations=0; regrets=[]; gaps=[]
        grid=np.arange(0,1+h/2,h); grid[-1]=1.0
        for _ in range(reps):
            y=rng.dirichlet(np.ones(K)*2); ref=rng.dirichlet(np.ones(K)*2); cand=rng.dirichlet(np.ones(K)*2)
            # exact dense and coarse MAE risks for one patient; Lipschitz bound is deterministic and extends under averaging.
            pdense=(1-dense[:,None])*ref+dense[:,None]*cand; rd=np.abs(pdense-y).mean(1); j=np.argmin(rd); ro=rd[j]; lo=dense[j]
            pg=(1-grid[:,None])*ref+grid[:,None]*cand; rg=np.abs(pg-y).mean(1); k=np.argmin(rg); rr=rg[k]-ro; regrets.append(rr); gaps.append(abs(grid[k]-lo)); violations+=rr>bound+1e-10
        rows.append(dict(K=K,grid_step=h,reps=reps,theoretical_regret_bound=bound,max_observed_regret=max(regrets),mean_regret=np.mean(regrets),max_lambda_gap=max(gaps),bound_violations=violations))
    d=pd.DataFrame(rows); d.to_csv(OUT/'E5_lambda_grid_regret.csv',index=False); return d

def _targets(rng,base,n,conc=250):
    return np.vstack([rng.dirichlet(np.maximum(base*conc,.3)) for _ in range(n)])

def e8_honest_sga(reps=500,seed=20260908):
    rng=np.random.default_rng(seed); rows=[]
    q0=np.array([.65,.25,.10]); q1=np.array([.45,.40,.15]); q2=np.array([.75,.10,.15])
    lib=np.stack([q0,q1,q2])
    for scenario,true in [('true_structure',q1),('null_frozen',q0),('out_of_library',np.array([.25,.20,.55]))]:
      for n in [20,50,100]:
        metrics={'honest':[],'bootstrap':[]}
        for r in range(reps):
            cal=_targets(rng,true,n); pats=np.array([f'p{i}' for i in range(n)]); pred=np.tile(lib[None,:,:],(n,1,1))
            test=_targets(rng,true,2000); ptest=np.tile(lib[None,:,:],(len(test),1,1)); frozen_test=np.abs(test-q0).mean(1).mean()
            for method in ['honest','bootstrap']:
                if method=='honest': sel=honest_safe_graph_average(cal,pred,pats,frozen_index=0,step=.25,alpha=.05,safety_margin=0,one_se=True)
                else: sel=safe_graph_average(cal,pred,pats,frozen_index=0,step=.25,alpha=.05,safety_margin=0,bootstrap=300,seed=seed+r+n,one_se=True)
                w=sel.weights; mix=ptest@w if False else np.einsum('m,rms->rs',w,ptest); risk=np.abs(test-mix).mean(1).mean(); diff=risk-frozen_test
                metrics[method].append((diff,float(w[0]),sel.fallback_used))
        for method,vals in metrics.items():
            arr=np.array([[v[0],v[1],float(v[2])] for v in vals]); rows.append(dict(scenario=scenario,n=n,reps=reps,method=method,mean_test_risk_change=arr[:,0].mean(),p95_test_risk_change=np.quantile(arr[:,0],.95),harm_violation_rate=np.mean(arr[:,0]>1e-6),mean_frozen_weight=arr[:,1].mean(),fallback_rate=arr[:,2].mean()))
    d=pd.DataFrame(rows); d.to_csv(OUT/'E8_honest_sga_calibration.csv',index=False); return d

if __name__=='__main__':
 t=time.time(); a=e5_lambda_grid(); print('E5 done',time.time()-t); t=time.time(); b=e8_honest_sga(); print('E8 done',time.time()-t); print(a.to_string(index=False)); print(b.to_string(index=False))
