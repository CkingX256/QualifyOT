import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from qualifyot.candidate_api import GraphFlowCandidate
from qualifyot.model import incidence, source_feature_matrix
from qualifyot.pipeline import PipelineConfig, QualifyOTPipeline
from qualifyot.transformer_candidate import GraphResidualTransformerCandidate

ST = ["A", "B", "C", "D"]
ED = [("A", "B"), ("B", "C")]


def toy(n=9, seed=91):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        s = rng.dirichlet([4, 3, 2, 3])
        f1 = min(.8*s[0], .015 + .12*s[0]*s[2] + .03*s[3])
        f2 = min(.7*(s[1]+f1), .01 + .09*s[0]*s[1])
        t = s.copy(); t[0] -= f1; t[1] += f1-f2; t[2] += f2
        t = np.clip(t, 1e-8, None); t /= t.sum()
        row = {"patient_id": f"P{i}", "pair_id": f"P{i}:0->1", "source_time": 0., "target_time": 1.}
        for k, v in zip(ST, s): row[f"source__{k}"] = v
        for k, v in zip(ST, t): row[f"target__{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def fused(seed=7, residual=.2, epochs=4):
    return GraphResidualTransformerCandidate(
        states=ST, edges=ED, d_model=8, nhead=2, num_layers=1,
        dim_feedforward=16, epochs=epochs, seed=seed,
        residual_fraction=residual, anchor_strength=.15, patient_balanced=True,
    )


def test_zero_residual_exactly_nests_graphflow():
    df = toy()
    tr, te = df.iloc[:7].copy(), df.iloc[7:].copy()
    a = GraphFlowCandidate(states=ST, edges=ED, alpha=1., patient_balanced=True).fit(tr)
    b = fused(residual=0., epochs=2).fit(tr)
    assert np.allclose(a.predict(te), b.predict(te), atol=2e-7)


def test_prediction_is_graph_constrained_and_simplex_closed():
    df = toy(); tr, te = df.iloc[:7], df.iloc[7:]
    c = fused(epochs=5).fit(tr)
    q = c.predict(te); f = c.predict_flows(te)
    _, s = source_feature_matrix(te, states=ST)
    B = incidence(ST, ED)
    reconstructed = np.vstack([x + B @ z for x, z in zip(s, f)])
    assert np.all(f >= -1e-12)
    assert np.allclose(q, reconstructed, atol=2e-6)
    assert np.all(q >= -1e-12)
    assert np.allclose(q.sum(1), 1., atol=1e-8)


def test_unified_transformer_prediction_never_reads_target():
    df = toy(); tr, te = df.iloc[:7].copy(), df.iloc[7:].copy()
    c = fused(epochs=3).fit(tr)
    q1 = c.predict(te)
    q2 = c.predict(te.drop(columns=[f"target__{s}" for s in ST]))
    assert np.allclose(q1, q2, atol=1e-8)


def test_unified_transformer_deterministic_seed():
    df = toy(); tr, te = df.iloc[:7], df.iloc[7:]
    a = fused(seed=44, epochs=4).fit(tr).predict(te)
    b = fused(seed=44, epochs=4).fit(tr).predict(te)
    assert np.allclose(a, b, atol=1e-8)


def test_diagnostics_make_anchor_and_feasibility_explicit():
    c = fused(epochs=3).fit(toy().iloc[:7])
    d = c.candidate_diagnostics()
    assert d["method"] == "graph-constrained residual Transformer"
    assert d["graphflow_train_mae"] >= 0
    assert d["fused_train_mae"] >= 0
    assert 0 <= d["feasibility_scaling_fraction"] <= 1
    assert 0 <= d["mean_feasibility_scale"] <= 1
    assert d["parameter_count"] > 0


def test_single_entry_pipeline_uses_same_iut_and_no_extra_neural_gate():
    df = toy(7)
    pipe = QualifyOTPipeline(
        ST, ED,
        candidate="graph-transformer",
        config=PipelineConfig(bootstrap=20, seed=23, orthogonal_profiles=False),
        transformer_kwargs={"epochs": 2, "d_model": 8, "nhead": 2, "dim_feedforward": 16},
    )
    out = pipe.run(df)
    assert out["core_evidence_state"] in {"Qualified", "Promising", "Equivocal", "Adverse", "Negligible"}
    assert set(out["core_axes"]) == {"Movement", "Utility", "Retention"}
    assert out["methodology"]["heldout_target_used_for_tuning"] is False
    assert out["methodology"]["orthogonal_profiles_are_core_gates"] is False


def test_primary_cli_graph_transformer(tmp_path):
    from qualifyot.cli import main as cli_main
    import json
    df=toy(7); csv=tmp_path/'toy.csv'; out=tmp_path/'out.json'; df.to_csv(csv,index=False)
    rc=cli_main(['candidate-run',str(csv),'--states',','.join(ST),'--candidate','graph-transformer',
                 '--graph','A->B,B->C','--bootstrap','10','--epochs','2','--d-model','8',
                 '--nhead','2','--ff','16','--seed','19','--out',str(out)])
    assert rc==0
    obj=json.loads(out.read_text())
    assert obj['candidate']=='GraphResidualTransformer'
    assert set(obj['core_axes'])=={'Movement','Utility','Retention'}
