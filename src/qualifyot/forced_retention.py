from __future__ import annotations
import numpy as np
import pandas as pd
from .model import mae
from .utils import patient_equal

def audit_forced_retention(target,reference,candidate,patients,weights=(0,.1,.25,.5,.75,1.0),bootstrap=5000,seed=0):
    Y=np.asarray(target,float); R=np.asarray(reference,float); C=np.asarray(candidate,float); P=np.asarray(patients).astype(str); up=np.unique(P); rng=np.random.default_rng(seed)
    rows=[]
    for lam in weights:
        Q=(1-lam)*R+lam*C; reg=mae(Y,Q)-mae(Y,R); pr=np.array([reg[P==p].mean() for p in up])
        idx=rng.integers(0,len(up),size=(bootstrap,len(up))); boot=pr[idx].mean(1); k=max(1,int(np.ceil(.1*len(pr))))
        rows.append({'lambda':float(lam),'reference_risk':patient_equal(mae(Y,R),P),'forced_risk':patient_equal(mae(Y,Q),P),'mean_regret':float(pr.mean()),'regret_lo':float(np.quantile(boot,.025)),'regret_hi':float(np.quantile(boot,.975)),'fraction_patients_harmed':float((pr>0).mean()),'worst_decile_mean_regret':float(np.sort(pr)[::-1][:k].mean()),'max_patient_regret':float(pr.max())})
    return pd.DataFrame(rows)
