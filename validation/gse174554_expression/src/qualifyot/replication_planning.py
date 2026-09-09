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


def bayesian_bootstrap_assurance(
    contributions,
    n_values=(20, 30, 40, 50, 60, 80, 100),
    *,
    posterior_draws=300,
    future_draws=300,
    confidence=.95,
    seed=0,
):
    """Bayesian-bootstrap predictive assurance for future positive LCB evidence.

    Unlike the fixed empirical plug-in planner, this integrates uncertainty in
    the patient-effect distribution by drawing Dirichlet weights over the
    observed patient contributions. It remains an assumption-dependent design
    aid rather than a universal prospective guarantee.
    """
    x=np.asarray(contributions,float)
    if x.ndim!=1 or len(x)<2: raise ValueError('contributions must contain at least two patients')
    if not np.isfinite(x).all(): raise ValueError('contributions contain non-finite values')
    rng=np.random.default_rng(seed); rows=[]; alpha=1-confidence
    post_w=rng.dirichlet(np.ones(len(x)),size=int(posterior_draws))
    for n in n_values:
        n=int(n); crit=t.ppf(1-alpha/2,n-1); probs=[]
        for w in post_w:
            idx=rng.choice(len(x),size=(int(future_draws),n),replace=True,p=w)
            z=x[idx]; means=z.mean(1); sds=z.std(1,ddof=1); lcb=means-crit*sds/np.sqrt(n)
            probs.append(float((lcb>0).mean()))
        probs=np.asarray(probs,float)
        rows.append({
            'n':n,'posterior_draws':int(posterior_draws),'future_draws':int(future_draws),
            'confidence':float(confidence),'assurance_mean':float(probs.mean()),
            'assurance_median':float(np.median(probs)),
            'assurance_lo':float(np.quantile(probs,.025)),
            'assurance_hi':float(np.quantile(probs,.975)),
            'observed_mean':float(x.mean()),'observed_sd':float(x.std(ddof=1)),
        })
    return pd.DataFrame(rows)


def assurance_target(plan, target=.80, criterion='mean'):
    """Return the first planned n reaching a requested assurance target."""
    col={'mean':'assurance_mean','median':'assurance_median','lower':'assurance_lo'}.get(criterion)
    if col is None: raise KeyError("criterion must be 'mean', 'median' or 'lower'")
    q=plan[plan[col]>=float(target)]
    return None if q.empty else int(q.iloc[0]['n'])
