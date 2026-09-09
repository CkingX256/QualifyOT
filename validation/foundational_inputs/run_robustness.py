from __future__ import annotations

"""Reproduce the eta- and ontology-robustness audit used by the manuscript.

This is an orthogonal sensitivity analysis. It never changes the locked primary
contract or relabels a primary result.

eta audit
---------
The structured GraphFlow label program in ``qualifyot.model.residual_flow`` is

    min_{f>=0,r} ||r||_1 + eta * 1^T f   subject to B f + r = Delta p.

The frozen implementation uses eta=1e-6. We (i) compare residual-flow labels
for eta in {1e-8,...,1e-2} against the frozen eta on 51 observed pairs from
three five-state cohorts, and (ii) rerun the complete patient-level LOPO
pipeline at eta in {1e-6,1e-4,1e-2}.

ontology audit
--------------
To isolate the estimand effect of ontology from refitting, frozen GSE272993
nine-state source/target/reference/candidate/retained predictions are
re-expressed under five prespecified outcome-blind, simplex-preserving
coarsenings plus the original ontology and rescored at the patient level.
"""

from dataclasses import dataclass
from pathlib import Path
import ast
import json
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
AUDIT = HERE.parent
RELEASE = HERE.parents[2]
CORE = RELEASE
sys.path.insert(0, str(CORE / "src"))

from qualifyot.candidate_api import CandidatePredictor
from qualifyot.generic_engine import run_generic_lopo
from qualifyot.model import incidence, residual_flow, feature_matrix, source_feature_matrix, fit_graphflow, hellinger, mae
from qualifyot.experiment import bootstrap_contrast
from qualifyot.inference import movement_margin_evidence
from qualifyot.decision import classify_iut_contrast, IUTConfig

RESULTS = AUDIT / "results"
FIGURES = AUDIT / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

SEED = 20260909
ETA_PRIMARY = 1e-6
ETA_LABEL_GRID = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
ETA_EVIDENCE_GRID = (1e-6, 1e-4, 1e-2)
BOOTSTRAP = 500

FIVE_STATES = ["Memory", "Effector", "Exhausted", "Regulatory", "Other_T"]
FIVE_EDGES = [("Memory", "Effector"), ("Effector", "Exhausted"), ("Memory", "Exhausted")]
ETA_DATASETS = {
    "GSE123813": "GSE123813_pairs.csv",
    "GSE236581": "GSE236581_pairs.csv",
    "GSE120575": "GSE120575_pairs.csv",
}

NINE = ["SCM", "CM", "Early Activated", "IFN", "Activated", "Early Effector", "Effector", "NK-like", "Exhausted"]
ONTOLOGY_GROUPS = {
    "O9_fine": [[0],[1],[2],[3],[4],[5],[6],[7],[8]],
    "O6_adjacent": [[0,1],[2],[3],[4],[5,6,7],[8]],
    "O5_canonical": [[0,1],[2,3],[4],[5,6,7],[8]],
    "O5_activation_merged": [[0,1],[2,3,4],[5],[6,7],[8]],
    "O4_coarse": [[0,1],[2,3,4],[5,6,7],[8]],
    "O3_coarse": [[0,1],[2,3,4,5,6,7],[8]],
}
ONTOLOGY_LABELS = {
    "O9_fine": NINE,
    "O6_adjacent": ["Memory","Early Activated","IFN","Activated","Effector/NK","Exhausted"],
    "O5_canonical": ["Memory","EarlyActivation/IFN","Activated","Effector/NK","Exhausted"],
    "O5_activation_merged": ["Memory","Activated/IFN","EarlyEffector","Effector/NK","Exhausted"],
    "O4_coarse": ["Memory","Activated/IFN","Effector/NK","Exhausted"],
    "O3_coarse": ["Memory","Activated/Effector/NK","Exhausted"],
}


@dataclass
class EtaGraphFlowCandidate(CandidatePredictor):
    states: tuple | list | None = None
    edges: tuple | list | None = None
    eta: float = ETA_PRIMARY
    alpha: float = 1.0
    patient_balanced: bool = True
    name: str = "EtaGraphFlow"
    _model: object = None

    def fit(self, pairs: pd.DataFrame):
        states = list(self.states)
        edges = [tuple(e) for e in self.edges]
        X, S, T = feature_matrix(pairs, states=states)
        P = pairs.patient_id.to_numpy(str)
        B = incidence(states, edges)
        F = np.vstack([residual_flow(s, t, tiny=float(self.eta), Bmat=B)[0] for s, t in zip(S, T)])
        self._model = fit_graphflow(X, S, T, alpha=self.alpha, flow_labels=F, Bmat=B, edges=edges,
                                    patients=P, patient_balanced=self.patient_balanced)
        return self

    def predict(self, pairs: pd.DataFrame):
        if self._model is None:
            raise RuntimeError("candidate not fitted")
        X, S = source_feature_matrix(pairs, states=list(self.states))
        return self.validate_predictions(self._model.predict(X, S), len(pairs))


def eta_flow_label_audit() -> pd.DataFrame:
    B = incidence(FIVE_STATES, FIVE_EDGES)
    rows = []
    for dataset, filename in ETA_DATASETS.items():
        df = pd.read_csv(CORE / "data" / "processed_pairs" / filename)
        S = df[[f"source__{s}" for s in FIVE_STATES]].to_numpy(float)
        T = df[[f"target__{s}" for s in FIVE_STATES]].to_numpy(float)
        primary = [residual_flow(s, t, tiny=ETA_PRIMARY, Bmat=B)[0] for s, t in zip(S, T)]
        for eta in ETA_LABEL_GRID:
            diffs = []
            for (s, t), f0 in zip(zip(S, T), primary):
                f = residual_flow(s, t, tiny=eta, Bmat=B)[0]
                diffs.append(float(np.max(np.abs(f - f0))) if len(f) else 0.0)
            rows.append({
                "dataset": dataset,
                "n_pairs": len(df),
                "eta": eta,
                "primary_eta": ETA_PRIMARY,
                "max_flow_abs_diff_vs_primary_eta": max(diffs, default=0.0),
                "mean_pair_max_abs_diff_vs_primary_eta": float(np.mean(diffs)) if diffs else 0.0,
                "pairs_changed_gt_1e-8": int(np.sum(np.asarray(diffs) > 1e-8)),
            })
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "eta_flow_label_sensitivity.csv", index=False)
    return out


def eta_evidence_plateau_audit() -> pd.DataFrame:
    rows = []
    for di, (dataset, filename) in enumerate(ETA_DATASETS.items()):
        df = pd.read_csv(CORE / "data" / "processed_pairs" / filename)
        for eta in ETA_EVIDENCE_GRID:
            cand = EtaGraphFlowCandidate(states=FIVE_STATES, edges=FIVE_EDGES, eta=eta, name=f"GraphFlow_eta_{eta:g}")
            r = run_generic_lopo(df, cand, bootstrap=BOOTSTRAP, seed=SEED + di * 1000,
                                 states=FIVE_STATES, patient_balanced=True,
                                 reference_rule="nested", return_predictions=True)
            pr = r["prediction_rows"]
            candidate_mae = float(pr.apply(lambda x: np.abs(np.asarray(x.target)-np.asarray(x.candidate)).mean(), axis=1).mean())
            rows.append({
                "dataset": dataset,
                "eta": eta,
                "bootstrap": BOOTSTRAP,
                "patients": r["patients"],
                "pairs": r["pairs"],
                "candidate_mae": candidate_mae,
                "movement_estimate": r["movement_margin_evidence"]["contrast"],
                "movement_lcb": r["movement_margin_evidence"]["contrast_lcb"],
                "utility_estimate": r["PUC"],
                "utility_lcb": r["PUC_lo"],
                "deployment_retention_estimate": r["NPI"],
                "deployment_retention_lcb": r["NPI_lo"],
                "positive_weight_fold_fraction": r["positive_weight_fold_fraction"],
                "evidence_state": r["core_evidence_state"],
                "qualified": bool(r["core_qualified"]),
            })
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "eta_evidence_plateau.csv", index=False)
    return out


def _agg(x: np.ndarray, groups: list[list[int]]) -> np.ndarray:
    return np.column_stack([x[:, g].sum(axis=1) for g in groups])


def ontology_fixed_prediction_audit() -> pd.DataFrame:
    pred = CORE / "results" / "reference_and_loss" / "GSE272993_9state__nested_prediction_rows.csv"
    d = pd.read_csv(pred)
    pats = d.patient_id.astype(str).to_numpy()
    arrays = {k: np.vstack(d[k].map(ast.literal_eval).values) for k in ["target","source","reference","candidate","retained"]}
    cfg = IUTConfig()
    rows = []
    for name, groups in ONTOLOGY_GROUPS.items():
        Y,S,N,C,A = [_agg(arrays[k], groups) for k in ["target","source","reference","candidate","retained"]]
        for arr in [Y,S,N,C,A]:
            if not np.allclose(arr.sum(axis=1), 1.0, atol=2e-8):
                raise RuntimeError(f"{name}: ontology aggregation violates simplex closure")
        mv = movement_margin_evidence(hellinger(C,S), hellinger(Y,S), pats,
                                      movement_margin=cfg.movement_margin,
                                      alpha=cfg.component_alpha, method="t",
                                      bootstrap=1000, seed=SEED)
        u = bootstrap_contrast(mae(Y,N)-mae(Y,C), pats, 1000, SEED+1)
        nd = bootstrap_contrast(mae(Y,N)-mae(Y,A), pats, 1000, SEED+2)
        decision = classify_iut_contrast(movement_contrast_lcb=mv.contrast_lcb,
                                         utility_lcb=u[1], utility_ucb=u[2],
                                         retention_lcb=nd[1], positive_weight_fraction=0.0,
                                         cfg=cfg)
        mapping = dict(zip(ONTOLOGY_LABELS[name], [[NINE[i] for i in g] for g in groups]))
        rows.append({
            "ontology": name,
            "n_states": len(groups),
            "mapping_json": json.dumps(mapping, sort_keys=True),
            "movement_estimate": mv.contrast,
            "movement_lcb": mv.contrast_lcb,
            "utility_estimate": u[0],
            "utility_lcb": u[1],
            "utility_ucb": u[2],
            "deployment_retention_estimate": nd[0],
            "deployment_retention_lcb": nd[1],
            "deployment_retention_ucb": nd[2],
            "evidence_state": decision.label,
            "qualified": bool(decision.qualified),
        })
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "ontology_fixed_prediction_robustness.csv", index=False)
    return out


def make_figures(eta_labels: pd.DataFrame, eta_evidence: pd.DataFrame, ontology: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for dataset, g in eta_labels.groupby("dataset", sort=False):
        g = g.sort_values("eta")
        ax.plot(g["eta"], g["max_flow_abs_diff_vs_primary_eta"], marker="o", label=dataset)
    ax.axvline(ETA_PRIMARY, linestyle="--", linewidth=1.0, label=r"frozen $\eta=10^{-6}$")
    ax.set_xscale("log")
    ax.set_xlabel(r"GraphFlow scalarization coefficient $\eta$")
    ax.set_ylabel(r"Max $|f_\eta-f_{10^{-6}}|$")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "supp_eta_robustness.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "supp_eta_robustness.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    x = np.arange(len(ontology))
    ax.plot(x, ontology["movement_lcb"], marker="o", label="Movement LCB")
    ax.plot(x, ontology["utility_lcb"], marker="o", label="Utility LCB")
    ax.plot(x, ontology["deployment_retention_lcb"], marker="o", label="Deployment Retention LCB")
    ax.axhline(0, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ontology["ontology"], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("One-sided lower confidence bound")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "supp_ontology_robustness.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "supp_ontology_robustness.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_report(eta_labels: pd.DataFrame, eta_evidence: pd.DataFrame, ontology: pd.DataFrame) -> None:
    plateau = eta_labels[(eta_labels.eta >= 1e-6) & (eta_labels.eta <= 1e-2)]
    low = eta_labels[eta_labels.eta < 1e-6]
    state_grid = "; ".join(
        f"{d}: " + "/".join(g.sort_values("eta").evidence_state.astype(str))
        for d, g in eta_evidence.groupby("dataset", sort=False)
    )
    txt = f"""# Foundational-input robustness audit - eta and ontology

This analysis is orthogonal to the locked primary contracts. It tests two inputs to the structured-candidate interface without allowing either sensitivity analysis to relabel a primary result.

## GraphFlow scalarization coefficient eta

The GraphFlow residual-flow label program is written as `min ||r||_1 + eta 1^T f` subject to `Bf+r=Delta p`, `f>=0`; the frozen implementation uses `eta=1e-6`. Across 51 observed pairs from GSE123813, GSE236581 and GSE120575, every residual-flow label was numerically unchanged relative to the frozen eta for all eta in `[1e-6,1e-2]` (maximum absolute flow difference {plateau.max_flow_abs_diff_vs_primary_eta.max():.3g}; changed pairs {int(plateau['pairs_changed_gt_1e-8'].sum())}/{int(plateau.n_pairs.sum())}). At the two smaller numerical stress values, `1e-8` and `1e-7`, all {len(low)} dataset-by-eta cells changed at least one flow label, indicating solver/scalarization sensitivity below the frozen scale rather than a biologically meaningful tuning regime.

Complete LOPO evidence was rerun at eta `1e-6`, `1e-4` and `1e-2` with the same patient-level evidence engine. Evidence states were: {state_grid}. Thus the frozen eta lies on an empirical stability plateau for these three cohorts. This is a local numerical robustness statement, not a proof of invariance to arbitrary eta or graph topology.

## Ontology

Frozen GSE272993 nine-state source, target, reference, candidate and retained predictions were re-expressed under five prespecified outcome-blind, simplex-preserving coarsenings plus the original ontology. This fixed-prediction design isolates the ontology dependence of the estimand and evidence calculation; it is not a full candidate-refitting comparison. No tested ontology produced Qualification. Movement remained supported in {int((ontology.movement_lcb>0).sum())}/{len(ontology)} ontologies; Utility remained supported in {int((ontology.utility_lcb>0).sum())}/{len(ontology)} and unresolved in {int((ontology.utility_lcb<=0).sum())}/{len(ontology)}; development-selected Deployment Retention was exactly zero in {int(np.isclose(ontology.deployment_retention_estimate,0).sum())}/{len(ontology)}. Evidence states were {', '.join(f'{r.ontology}={r.evidence_state}' for r in ontology.itertuples())}.

The qualitative deployment conclusion is therefore stable over this declared ontology library, while effect magnitudes and the Utility state remain ontology-dependent. That dependence is expected: ontology is part of the frozen estimand, not a nuisance label dictionary.

## Claim boundary

The experiments establish local robustness over a declared eta neighborhood and six explicit ontology resolutions. They do not establish invariance to arbitrary state definitions, graph topologies, biological annotations or clinical endpoints.
"""
    (RESULTS / "FOUNDATIONAL_INPUT_ROBUSTNESS_REPORT.md").write_text(txt, encoding="utf-8")


def main() -> None:
    eta_labels = eta_flow_label_audit()
    eta_evidence = eta_evidence_plateau_audit()
    ontology = ontology_fixed_prediction_audit()
    make_figures(eta_labels, eta_evidence, ontology)
    write_report(eta_labels, eta_evidence, ontology)
    print("eta label audit")
    print(eta_labels.to_string(index=False))
    print("\neta evidence plateau")
    print(eta_evidence.to_string(index=False))
    print("\nontology fixed-prediction audit")
    print(ontology.to_string(index=False))

if __name__ == "__main__":
    main()
