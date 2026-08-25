from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import t

def empirical_t_lcb_power(contributions,n_values=(20,30,40,50,60,80,100),mc=50000,confidence=.95,seed=0):
    """Conditional fixed-prediction replication planning.

    Treats the observed patient-level contrast distribution as an empirical
    population and estimates how often a future t interval would have lower
    bound >0. This is a planning diagnostic only: it does not model candidate
    refitting, distribution shift or changes in weight selection.
    """
    x=np.asarray(contributions,float); rng=np.random.default_rng(seed); rows=[]
    alpha=1-confidence
    for n in n_values:
        idx=rng.integers(0,len(x),size=(mc,int(n))); z=x[idx]; means=z.mean(1); sds=z.std(1,ddof=1); crit=t.ppf(1-alpha/2,int(n)-1); lcb=means-crit*sds/np.sqrt(n)
        rows.append({'n':int(n),'mc':int(mc),'confidence':float(confidence),'observed_mean':float(x.mean()),'observed_sd':float(x.std(ddof=1)),'prob_point_positive':float((means>0).mean()),'prob_lcb_positive':float((lcb>0).mean()),'median_lcb':float(np.median(lcb)),'lcb_q025':float(np.quantile(lcb,.025)),'lcb_q975':float(np.quantile(lcb,.975))})
    return pd.DataFrame(rows)
