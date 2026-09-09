import numpy as np
import pandas as pd
import pytest

from qualifyot.decision import IUTConfig, retained_risk_improvement
from qualifyot.inference import mean_interval, one_sided_lcb, patient_equal_values
from qualifyot.candidate_api import GraphFlowCandidate
from qualifyot.generic_engine import run_generic_lopo
from qualifyot.partial_identification import graph_confidence_set
from qualifyot.safe_averaging import safe_graph_average


def _toy_pairs(n=9):
    rng=np.random.default_rng(123)
    states=['A','B','C']
    rows=[]
    for i in range(n):
        s=rng.dirichlet([4,3,2]); t=s.copy(); mv=.12*s[0]; t[0]-=mv; t[1]+=mv
        t=np.maximum(t,1e-8); t=t/t.sum()
        r={'patient_id':f'p{i}','pair_id':f'p{i}:0->1','source_time':0,'target_time':1}
        for j,k in enumerate(states): r[f'source__{k}']=s[j]; r[f'target__{k}']=t[j]
        rows.append(r)
    return pd.DataFrame(rows),states


def test_npi_identity_makes_final_risk_gate_redundant():
    assert retained_risk_improvement(.02,.10,.08)
    assert not retained_risk_improvement(0.0,.10,.10)
    with pytest.raises(ValueError):
        retained_risk_improvement(.02,.10,.09)


def test_iut_config_validation_and_mapping():
    c=IUTConfig.from_mapping({'movement_margin':.02,'component_alpha':.025})
    assert c.movement_margin==.02 and c.component_alpha==.025
    with pytest.raises(ValueError): IUTConfig(movement_margin=-.1)


def test_patient_intervals_are_deterministic_and_finite():
    x=np.array([-.02,.01,.03,.04,.00,.02,.01,.05,.03])
    for method in ['t','percentile','bootstrap_t']:
        a=mean_interval(x,method=method,bootstrap=300,seed=7)
        b=mean_interval(x,method=method,bootstrap=300,seed=7)
        assert a==b and np.isfinite([a.lower,a.estimate,a.upper]).all()
        assert np.isfinite(one_sided_lcb(x,method=method,bootstrap=300,seed=9))


def test_patient_equal_collapse_ignores_pair_count_as_weight():
    vals=np.array([1.,3.,10.]); pats=np.array(['a','a','b'])
    up,x=patient_equal_values(vals,pats)
    assert list(up)==['a','b'] and np.allclose(x,[2,10])


def test_empty_graph_is_valid_persistence_candidate():
    df,states=_toy_pairs(7)
    c=GraphFlowCandidate(states=states,edges=[]).fit(df.iloc[:5].reset_index(drop=True))
    q=c.predict(df.iloc[5:].reset_index(drop=True))
    src=df.iloc[5:][[f'source__{s}' for s in states]].to_numpy(float)
    assert np.allclose(q,src)


def test_fixed_reference_rule_is_explicit_and_reproducible():
    df,states=_toy_pairs(8)
    c=GraphFlowCandidate(states=states,edges=[('A','B')],patient_balanced=True)
    a=run_generic_lopo(df,c,bootstrap=100,seed=44,patient_balanced=True,states=states,reference_rule='Persistence')
    b=run_generic_lopo(df,c,bootstrap=100,seed=44,patient_balanced=True,states=states,reference_rule='Persistence')
    assert a['reference_rule']=='Persistence'
    assert np.isclose(a['PUC'],b['PUC']) and np.isclose(a['NPI'],b['NPI'])
    assert set(a['folds'].reference_kind)=={'Persistence'}


def test_graph_confidence_set_handles_degenerate_duplicate_predictions():
    n=20; base=np.linspace(.05,.15,n)
    losses=np.column_stack([base,base,base+.08])
    pats=np.array([f'p{i}' for i in range(n)])
    out=graph_confidence_set(losses,pats,['g0','g1','g2'],bootstrap=300,seed=2)
    assert set(out.members)=={'g0','g1'} and 'g2' in out.excluded


def test_sga_seed_determinism_and_fallback():
    n=30; y=np.tile([.7,.3],(n,1)); f=np.tile([.65,.35],(n,1)); bad=np.tile([.4,.6],(n,1))
    pred=np.stack([f,bad],axis=1); pats=np.array([f'p{i}' for i in range(n)])
    a=safe_graph_average(y,pred,pats,frozen_index=0,step=.5,bootstrap=300,seed=5)
    b=safe_graph_average(y,pred,pats,frozen_index=0,step=.5,bootstrap=300,seed=5)
    assert np.allclose(a.weights,b.weights)
    assert a.fallback_used and a.certified_nonharm_on_selection_sample
