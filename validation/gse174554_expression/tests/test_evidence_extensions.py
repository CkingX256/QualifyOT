import numpy as np
from qualifyot.evidence_state import classify
from qualifyot.forced_retention import audit_forced_retention
from qualifyot.replication_planning import empirical_t_lcb_power
from qualifyot.measurement import dirichlet_posterior_draw

def test_evidence_state_promising_has_limiting_gate():
    s=classify(.5,.3,.01,.03,-.001,1.0,.09,.10)
    assert s.label=='Promising' and 'NPI lower bound' in s.limiting_gates

def test_forced_retention_identity_zero():
    y=np.array([[.7,.3],[.4,.6]]); p=np.array(['a','b']); d=audit_forced_retention(y,y,y,p,weights=(0,1),bootstrap=50,seed=1)
    assert np.allclose(d.mean_regret,0)

def test_replication_planning_increases_for_strong_positive():
    x=np.array([.01,.02,.015,.025,.012,.018]); d=empirical_t_lcb_power(x,[10,100],mc=2000,seed=2)
    assert d.prob_lcb_positive.iloc[1] >= d.prob_lcb_positive.iloc[0]

def test_dirichlet_draw_simplex():
    q=dirichlet_posterior_draw([10,5,0],np.random.default_rng(3),size=5)
    assert q.shape==(5,3) and np.all(q>0) and np.allclose(q.sum(1),1)
