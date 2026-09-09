import numpy as np
import pandas as pd

from qualifyot.bayesian_evidence import bayesian_bootstrap_mean, bayesian_bootstrap_graph_profile
from qualifyot.benchmarks import benchmark_patient_effects, signflip_test_one_sided
from qualifyot.robust_averaging import robust_graph_average
from qualifyot.edge_shrinkage import ContinuousEdgeShrinkageCandidate
from qualifyot.calibration import empirical_upper_critical_value


def _pairs(n=18):
    rng=np.random.default_rng(44)
    states=['A','B','C']
    rows=[]
    for i in range(n):
        s=rng.dirichlet([5,3,2]); t=s.copy(); mv=.10*s[0]
        t[0]-=mv; t[1]+=mv; t=t/t.sum()
        r={'patient_id':f'p{i}','pair_id':f'p{i}:0->1','source_time':0,'target_time':1}
        for j,k in enumerate(states):
            r[f'source__{k}']=s[j]; r[f'target__{k}']=t[j]
        rows.append(r)
    return pd.DataFrame(rows),states


def test_bayesian_bootstrap_mean_detects_clear_positive_effect():
    x=np.array([.02,.03,.01,.04,.025,.015,.03,.02])
    a=bayesian_bootstrap_mean(x,draws=2000,seed=7)
    b=bayesian_bootstrap_mean(x,draws=2000,seed=7)
    assert a==b
    assert a.probability_above_margin > .99
    assert a.credible_interval[0] > 0


def test_bayesian_graph_profile_is_predictive_not_empty():
    n=20; pats=np.array([f'p{i}' for i in range(n)])
    base=np.linspace(.05,.08,n)
    losses=np.column_stack([base+.02,base,base+.05])
    edges={'g0':[('A','B'),('B','C')],'g1':[('A','B')],'g2':[('B','C')]}
    p=bayesian_bootstrap_graph_profile(losses,pats,['g0','g1','g2'],graph_edges=edges,draws=2000,seed=3)
    best=p.model_table.iloc[0]
    assert best['model']=='g1'
    assert best['bb_probability_best'] > .99
    e=p.edge_table.set_index(['edge_from','edge_to'])
    assert e.loc[('A','B'),'bb_predictive_inclusion'] > .99


def test_benchmarks_and_signflip_are_deterministic():
    x=np.array([.01,.02,.03,.015,.025,.018,.012,.021,.017])
    a=benchmark_patient_effects(x,bootstrap=300,permutations=500,bayes_draws=1000,seed=9)
    b=benchmark_patient_effects(x,bootstrap=300,permutations=500,bayes_draws=1000,seed=9)
    assert a==b
    assert a.t_pass and a.bayesian_bootstrap_pass
    assert signflip_test_one_sided(x,seed=1)==signflip_test_one_sided(x,seed=99)  # exact at n=9


def test_robust_graph_average_rejects_tail_harming_candidate():
    n=30; y=np.tile([.7,.3],(n,1)); frozen=np.tile([.6,.4],(n,1)); cand=y.copy()
    # Six patients (20%) follow the reverse mechanism; candidate is much worse there.
    y[-6:]=[.3,.7]
    pred=np.stack([frozen,cand],axis=1); pats=np.array([f'p{i}' for i in range(n)])
    r=robust_graph_average(y,pred,pats,frozen_index=0,step=.5,bootstrap=400,seed=5,tail_fraction=.2,tail_harm_margin=0)
    assert r.tail_harm_ucb <= 1e-12
    assert r.fallback_used


def test_continuous_edge_shrinkage_obeys_candidate_api():
    df,states=_pairs(15)
    c=ContinuousEdgeShrinkageCandidate(states,[('A','B'),('B','C'),('A','C')],gamma_grid=(0,.003,.01),cv_folds=3,seed=7)
    c.fit(df.iloc[:12].reset_index(drop=True))
    q=c.predict(df.iloc[12:].reset_index(drop=True))
    assert q.shape==(3,3)
    assert np.all(q>=-1e-10) and np.allclose(q.sum(1),1)
    assert len(c.regularization_path())==3
    assert set(c.edge_weights())=={'A->B','B->C','A->C'}


def test_empirical_calibration_is_conservative_finite_sample_quantile():
    x=np.linspace(0,1,1000)
    r=empirical_upper_critical_value(x,n=20,statistic='PDR',alpha=.05)
    assert .94 <= r.critical_value <= .96
    assert r.empirical_exceedance <= .05
