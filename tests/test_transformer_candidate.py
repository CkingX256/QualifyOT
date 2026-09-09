import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from qualifyot.candidates import RobustBlendCandidate
from qualifyot.transformer_candidate import ResidualStateTransformerCandidate, GraphAwareTransformerCandidate, TransformerGraphFlowCandidate
from qualifyot.generic_engine import run_generic_lopo
from qualifyot.cli import main as cli_main

ST=["A","B","C","D"]


def toy(n=10,seed=4):
    rng=np.random.default_rng(seed); rows=[]
    for i in range(n):
        s=rng.dirichlet(np.array([4.,3.,2.,3.]))
        # nonlinear interaction: A->B strengthens when C is abundant; B->C
        # depends on A*D.  This is intentionally not exactly linear in source.
        m1=min(float(s[0]),0.03+0.12*s[0]/(0.08+s[0])*s[2])
        m2=min(float(s[1]+m1),0.02+0.10*s[0]*s[3])
        t=s.copy(); t[0]-=m1; t[1]+=m1-m2; t[2]+=m2
        t=np.clip(t,1e-8,None); t=t/t.sum()
        r={"patient_id":f"P{i}","pair_id":f"P{i}:0->1","source_time":0.0,"target_time":1.0}
        for k,v in zip(ST,s): r[f"source__{k}"]=v
        for k,v in zip(ST,t): r[f"target__{k}"]=v
        rows.append(r)
    return pd.DataFrame(rows)


def make_candidate(seed=11,graph=False):
    kw=dict(states=ST,d_model=8,nhead=2,num_layers=1,dim_feedforward=16,
            epochs=8,learning_rate=3e-3,residual_scale=.25,dropout=0.0,
            seed=seed,patient_balanced=True)
    if graph:
        return GraphAwareTransformerCandidate(edges=[("A","B"),("B","C")],**kw)
    return ResidualStateTransformerCandidate(**kw)


def test_transformer_simplex_and_diagnostics():
    df=toy(); c=make_candidate().fit(df.iloc[:8])
    q=c.predict(df.iloc[8:])
    assert q.shape==(2,4)
    assert np.all(q>=0)
    assert np.allclose(q.sum(1),1.0,atol=1e-6)
    d=c.candidate_diagnostics()
    assert d["training_patients"]==8
    assert d["parameter_count"]>0
    assert d["mean_abs_logratio_correction"]>=0


def test_transformer_is_deterministic_for_fixed_seed():
    df=toy()
    a=make_candidate(seed=123).fit(df.iloc[:8]).predict(df.iloc[8:])
    b=make_candidate(seed=123).fit(df.iloc[:8]).predict(df.iloc[8:])
    assert np.allclose(a,b,atol=1e-8)


def test_predict_does_not_read_target_columns():
    df=toy(); train=df.iloc[:8].copy(); test=df.iloc[8:].copy()
    c=make_candidate().fit(train)
    q1=c.predict(test)
    target_cols=[f"target__{s}" for s in ST]
    target_free=test.drop(columns=target_cols)
    q2=c.predict(target_free)
    assert np.allclose(q1,q2,atol=1e-8)
    # The baseline candidate received the same predictor-side refactor.
    b=RobustBlendCandidate(states=ST).fit(train)
    assert np.allclose(b.predict(test),b.predict(target_free),atol=1e-12)


def test_graph_aware_attention_requires_valid_graph_and_runs():
    df=toy()
    c=make_candidate(graph=True).fit(df.iloc[:8])
    q=c.predict(df.iloc[8:])
    assert q.shape==(2,4) and np.allclose(q.sum(1),1,atol=1e-6)
    with pytest.raises(KeyError):
        GraphAwareTransformerCandidate(states=ST,edges=[("A","Z")],d_model=8,nhead=2,epochs=2).fit(df.iloc[:8])


def test_fresh_has_no_learned_state():
    df=toy(); c=make_candidate().fit(df.iloc[:8]); f=c.fresh()
    assert f._model is None and f._base is None
    with pytest.raises(RuntimeError): f.predict(df.iloc[8:])


def test_generic_lopo_collects_transformer_diagnostics():
    df=toy(7)
    c=ResidualStateTransformerCandidate(states=ST,d_model=8,nhead=2,dim_feedforward=16,epochs=3,seed=7,residual_scale=.2)
    out=run_generic_lopo(df,c,bootstrap=20,seed=7,patient_balanced=True,states=ST)
    assert len(out["folds"])==7
    assert "candidate_diagnostics" in out["folds"].iloc[0]
    assert out["core_evidence_state"] in {"Qualified","Promising","Equivocal","Adverse","Negligible"}


def test_cli_candidate_run_transformer(tmp_path,capsys):
    df=toy(7); csv=tmp_path/"toy.csv"; out=tmp_path/"out.json"; df.to_csv(csv,index=False)
    rc=cli_main(["candidate-run",str(csv),"--states",",".join(ST),"--candidate","transformer",
                 "--bootstrap","20","--epochs","2","--d-model","8","--nhead","2","--ff","16",
                 "--seed","9","--out",str(out)])
    assert rc==0 and out.exists()
    obj=json.loads(out.read_text())
    assert obj["schema"]=="qualifyot-candidate-run"
    assert obj["candidate"]=="ResidualStateTransformer"


def test_transformer_graphflow_preserves_graph_and_no_target_read():
    df=toy(10)
    c=TransformerGraphFlowCandidate(states=ST,edges=[("A","B"),("B","C")],d_model=8,nhead=2,
                                    dim_feedforward=16,epochs=6,seed=33,flow_correction_scale=.05).fit(df.iloc[:8])
    q1=c.predict(df.iloc[8:])
    q2=c.predict(df.iloc[8:].drop(columns=[f"target__{s}" for s in ST]))
    assert q1.shape==(2,4) and np.allclose(q1.sum(1),1,atol=1e-8)
    assert np.allclose(q1,q2,atol=1e-8)
    d=c.candidate_diagnostics()
    assert d["transformer_flow_mae"]>=0 and d["base_flow_mae"]>=0
    assert d["parameter_count"]>0


def test_cli_transformer_graphflow_runs(tmp_path):
    df=toy(7); csv=tmp_path/"toy_graph.csv"; out=tmp_path/"graphflow.json"; df.to_csv(csv,index=False)
    rc=cli_main(["candidate-run",str(csv),"--states",",".join(ST),"--candidate","transformer-graphflow",
                 "--graph","A->B,B->C","--bootstrap","10","--epochs","1","--d-model","8",
                 "--nhead","2","--ff","16","--seed","12","--out",str(out)])
    assert rc==0 and out.exists()
    obj=json.loads(out.read_text())
    assert obj["candidate"]=="TransformerGraphFlow"
    assert len(obj["folds"])==7
