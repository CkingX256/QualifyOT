from __future__ import annotations

"""Executed validation for the simplified GraphResidualTransformer method.

The script intentionally tests only the predictive candidate layer.  It does
not change the Movement–Utility–Retention IUT.  Real-data LOPO runs use the
same generic evidence engine and frozen patient sets as the rest of QualifyOT.
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd

from qualifyot.candidate_api import GraphFlowCandidate
from qualifyot.transformer_candidate import GraphResidualTransformerCandidate
from qualifyot.generic_engine import run_generic_lopo
from qualifyot.model import mae

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "fused_methodology"
OUT.mkdir(parents=True, exist_ok=True)

ST = ["A", "B", "C", "D", "E"]
ED = [("A", "B"), ("B", "C"), ("D", "C")]


def _row(pid, s, t):
    r = {"patient_id": str(pid), "pair_id": f"{pid}:0->1", "source_time": 0., "target_time": 1.}
    for k, v in zip(ST, s): r[f"source__{k}"] = float(v)
    for k, v in zip(ST, t): r[f"target__{k}"] = float(v)
    return r


def simulate(n, seed, regime="nonlinear", noise=.012):
    rng = np.random.default_rng(seed); rows = []
    for i in range(n):
        s = rng.dirichlet(np.array([4., 3., 2.5, 3.5, 4.]))
        if regime == "null":
            t = s.copy()
        else:
            if regime == "linear":
                f1, f2, f3 = .045*s[0], .035*s[1], .030*s[3]
            elif regime == "nonlinear":
                f1 = .02 + .16*s[0]*s[2] + .08*s[0]/(.08+s[0])*s[3]
                f2 = .01 + .14*s[1]*s[0]
                f3 = .01 + .15*s[3]*s[4]
            else:
                raise KeyError(regime)
            f1 = min(f1, .80*s[0]); f2 = min(f2, .70*(s[1]+f1)); f3 = min(f3, .70*s[3])
            t = s.copy(); t[0] -= f1; t[1] += f1-f2; t[3] -= f3; t[2] += f2+f3
        if noise:
            z = rng.normal(0, noise, len(ST)); z -= z.mean()
            t = np.clip(t + z, 1e-6, None); t /= t.sum()
        rows.append(_row(f"P{i}", s, t))
    return pd.DataFrame(rows)


def fused(seed, *, residual_fraction=.20, anchor_strength=.15, epochs=20):
    return GraphResidualTransformerCandidate(
        states=ST, edges=ED, alpha=1., d_model=8, nhead=2, num_layers=1,
        dim_feedforward=16, dropout=0., residual_fraction=residual_fraction,
        anchor_strength=anchor_strength, epochs=epochs, learning_rate=2e-3,
        seed=seed, patient_balanced=True,
    )


def synthetic(reps=10):
    rows = []
    for regime in ("null", "linear", "nonlinear"):
        for rep in range(reps):
            seed = 99000 + 1000*("null", "linear", "nonlinear").index(regime) + rep
            tr = simulate(40, seed, regime); te = simulate(500, seed+500000, regime)
            y = te[[f"target__{s}" for s in ST]].to_numpy(float)
            methods = {
                "GraphFlow": GraphFlowCandidate(states=ST, edges=ED, alpha=1., patient_balanced=True),
                "GraphResidualTransformer": fused(seed),
            }
            for name, c in methods.items():
                t0 = time.perf_counter(); c.fit(tr); p = c.predict(te)
                rows.append({"regime": regime, "replicate": rep, "candidate": name,
                             "mae": float(mae(y,p).mean()), "seconds": time.perf_counter()-t0})
    df = pd.DataFrame(rows); df.to_csv(OUT/"synthetic_fused_runs.csv", index=False)
    sm = df.groupby(["regime","candidate"],as_index=False).agg(
        mae_mean=("mae","mean"), mae_sd=("mae","std"), runtime_mean=("seconds","mean"))
    sm.to_csv(OUT/"synthetic_fused_summary.csv", index=False)
    return sm


def residual_sensitivity(reps=5):
    rows=[]
    for residual in (0., .10, .20, .35):
        for rep in range(reps):
            seed=120000+rep
            tr=simulate(40,seed,"nonlinear"); te=simulate(500,seed+500000,"nonlinear")
            y=te[[f"target__{s}" for s in ST]].to_numpy(float)
            c=fused(seed,residual_fraction=residual)
            c.fit(tr); p=c.predict(te)
            rows.append({"residual_fraction":residual,"replicate":rep,"mae":float(mae(y,p).mean())})
    df=pd.DataFrame(rows); df.to_csv(OUT/"residual_fraction_sensitivity.csv",index=False)
    df.groupby("residual_fraction",as_index=False).agg(mae_mean=("mae","mean"),mae_sd=("mae","std")).to_csv(
        OUT/"residual_fraction_sensitivity_summary.csv",index=False)


def real_one(label, path, states, edges, *, epochs, bootstrap):
    df=pd.read_csv(ROOT/path)
    c=GraphResidualTransformerCandidate(states=states,edges=edges,alpha=1.,d_model=8,nhead=2,
        num_layers=1,dim_feedforward=16,epochs=epochs,residual_fraction=.20,anchor_strength=.15,
        seed=20260831,patient_balanced=True)
    t0=time.perf_counter(); r=run_generic_lopo(df,c,bootstrap=bootstrap,seed=20260831,
        patient_balanced=True,states=states,reference_rule="nested",orthogonal_profiles=False)
    return {"dataset":label,"candidate":"GraphResidualTransformer","epochs":epochs,"bootstrap":bootstrap,
        "patients":r["patients"],"pairs":r["pairs"],"PDR":r["PDR"],"PDR_lo":r["PDR_lo"],
        "PUC":r["PUC"],"PUC_lo":r["PUC_lo"],"NPI":r["NPI"],"NPI_lo":r["NPI_lo"],
        "retention_stability":r["retention_stability"],"state":r["core_evidence_state"],
        "qualified":r["core_qualified"],"runtime_seconds":time.perf_counter()-t0}


def main():
    print(synthetic().to_string(index=False))
    residual_sensitivity()
    five=["Memory","Effector","Exhausted","Regulatory","Other_T"]
    fed=[("Memory","Effector"),("Effector","Exhausted"),("Memory","Exhausted")]
    aml=["HSC","Progenitor","GMP","Monocytic","Other"]
    aed=[("Monocytic","GMP"),("GMP","Progenitor"),("Progenitor","HSC")]
    real=[
      real_one("GSE123813","data/processed_pairs/GSE123813_pairs.csv",five,fed,epochs=20,bootstrap=500),
      real_one("GSE235063_AML","data/processed_pairs/GSE235063_AML_primary_pairs.csv",aml,aed,epochs=8,bootstrap=500),
    ]
    pd.DataFrame(real).to_csv(OUT/"real_fused_lopo_executed.csv",index=False)
    print(pd.DataFrame(real).to_string(index=False))


if __name__ == "__main__":
    main()
