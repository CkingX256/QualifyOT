import numpy as np
import pandas as pd
from qualifyot.candidate_api import MeanDeltaCandidate, GraphFlowCandidate
from qualifyot.generic_engine import run_generic_lopo
from qualifyot.model import STATES

def toy_pairs(npat=6):
    rows=[]
    rng=np.random.default_rng(2)
    for i in range(npat):
        s=rng.dirichlet(np.ones(5)*3); t=.8*s+.2*np.array([.1,.3,.3,.1,.2]); t=t/t.sum()
        r={'cohort':'toy','patient_id':f'P{i}','pair_id':f'P{i}:0->1','source_time':'pre','target_time':'post'}
        for k,v in zip(STATES,s): r[f'source__{k}']=v
        for k,v in zip(STATES,t): r[f'target__{k}']=v
        rows.append(r)
    return pd.DataFrame(rows)

def test_mean_delta_candidate_simplex():
    p=toy_pairs(); c=MeanDeltaCandidate().fit(p.iloc[:4]); q=c.predict(p.iloc[4:])
    assert q.shape==(2,5); assert np.all(q>=-1e-9); assert np.allclose(q.sum(1),1)

def test_graphflow_candidate_simplex():
    p=toy_pairs(); c=GraphFlowCandidate().fit(p.iloc[:4]); q=c.predict(p.iloc[4:])
    assert q.shape==(2,5); assert np.all(q>=-1e-9); assert np.allclose(q.sum(1),1)

def test_generic_engine_runs_patient_lopo():
    p=toy_pairs(7); out=run_generic_lopo(p,MeanDeltaCandidate(),bootstrap=100,seed=7)
    assert out['patients']==7; assert len(out['folds'])==7; assert 0<=out['positive_weight_fold_fraction']<=1

def test_generic_engine_custom_state_space():
    states=['A','B','C']; rows=[]
    rng=np.random.default_rng(9)
    for i in range(7):
        s=rng.dirichlet(np.ones(3)*3); t=.8*s+.2*np.array([.2,.5,.3]); t=t/t.sum()
        r={'cohort':'toy3','patient_id':f'P{i}','pair_id':f'P{i}:0->1','source_time':0,'target_time':1}
        for k,v in zip(states,s): r[f'source__{k}']=v
        for k,v in zip(states,t): r[f'target__{k}']=v
        rows.append(r)
    p=pd.DataFrame(rows); c=MeanDeltaCandidate(states=states)
    out=run_generic_lopo(p,c,bootstrap=100,seed=11,states=states)
    assert out['patients']==7 and out['states']=='A|B|C'

def test_single_edge_graphflow_candidate_shape():
    states=['A','B','C']; rows=[]
    rng=np.random.default_rng(10)
    for i in range(6):
        s=rng.dirichlet(np.ones(3)*3); t=s.copy(); move=min(.05,t[0]); t[0]-=move; t[1]+=move
        r={'cohort':'toy3','patient_id':f'P{i}','pair_id':f'P{i}:0->1','source_time':0,'target_time':1}
        for k,v in zip(states,s): r[f'source__{k}']=v
        for k,v in zip(states,t): r[f'target__{k}']=v
        rows.append(r)
    p=pd.DataFrame(rows); c=GraphFlowCandidate(states=states,edges=[('A','B')]).fit(p.iloc[:4]); q=c.predict(p.iloc[4:])
    assert q.shape==(2,3) and np.allclose(q.sum(1),1)

def test_generic_engine_reports_core_and_legacy_fields():
    import pandas as pd
    from pathlib import Path
    from qualifyot.candidates import RobustBlendCandidate
    from qualifyot.generic_engine import run_generic_lopo
    root=Path(__file__).resolve().parents[1]
    states=['Memory','Effector','Exhausted','Regulatory','Other_T']
    df=pd.read_csv(root/'data'/'processed_pairs'/'GSE123813_pairs.csv')
    out=run_generic_lopo(df,RobustBlendCandidate(states=states),bootstrap=30,seed=17,patient_balanced=True,states=states)
    assert 'core_qualified' in out and 'legacy_qualified' in out
    assert set(out['core_axes'])=={'Movement','Utility','Retention'}
    assert out['retention_stability']==out['positive_weight_fold_fraction']


def test_native_predictors_do_not_require_heldout_target_columns():
    from qualifyot.candidates import DirectDeltaRidgeCandidate, TargetRidgeCandidate, RobustBlendCandidate
    p=toy_pairs(7); train=p.iloc[:5].copy(); test=p.iloc[5:].copy()
    target_cols=[f"target__{s}" for s in STATES]
    target_free=test.drop(columns=target_cols)
    candidates=[
        MeanDeltaCandidate().fit(train),
        GraphFlowCandidate().fit(train),
        DirectDeltaRidgeCandidate(STATES).fit(train),
        TargetRidgeCandidate(STATES).fit(train),
        RobustBlendCandidate(STATES).fit(train),
    ]
    for c in candidates:
        assert np.allclose(c.predict(test),c.predict(target_free),atol=1e-12)
