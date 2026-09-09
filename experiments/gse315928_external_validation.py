from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from qualifyot.adaptive_candidates import PatientMultiScaleDeltaCandidate
from qualifyot.candidate_family import holm_rejections, iut_pvalue, one_sided_t_pvalue
from qualifyot.candidates import DirectDeltaRidgeCandidate, RobustBlendCandidate
from qualifyot.comparison_candidates import CompositionEntropicOTCandidate
from qualifyot.generic_engine import _inner_oof_generic, _resolve_reference, run_generic_lopo
from qualifyot.inference import honest_confirm_predictions, honest_confirmation_pvalues_localized
from qualifyot.model import feature_matrix, hellinger, mae, predict_null
from qualifyot.weight_selection import one_se_msw, patient_curves

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed_pairs" / "GSE315928_B_F1_pairs.csv"
SPLIT = ROOT / "configs" / "GSE315928_postspec" / "split.csv"
OUT = ROOT / "results" / "GSE315928_external_postspec" / "complete_validation"

CANDIDATES = (
    "RobustBlend",
    "DirectDeltaRidge_a10",
    "PatientMultiScaleDelta",
    "CompositionEntropicOT_e0.25",
)
SEEDS = (20260903, 20261003, 20261103, 20261203)


def states_of(df: pd.DataFrame) -> list[str]:
    return [c.split("__", 1)[1] for c in df.columns if c.startswith("source__")]


def make_candidate(name: str, states: list[str]):
    if name == "RobustBlend":
        return RobustBlendCandidate(states=states, alpha=10.0, blend=0.5, name=name)
    if name == "DirectDeltaRidge_a10":
        return DirectDeltaRidgeCandidate(states=states, alpha=10.0, patient_balanced=True, name=name)
    if name == "PatientMultiScaleDelta":
        return PatientMultiScaleDeltaCandidate(states=states, name=name)
    if name == "CompositionEntropicOT_e0.25":
        return CompositionEntropicOTCandidate(states=states, epsilon=0.25, patient_balanced=True, name=name)
    raise KeyError(name)


def diag_pvalues(out: dict) -> dict[str, float]:
    d = out["patient_influence"]
    p_m = one_sided_t_pvalue(d["movement_margin_contribution"].to_numpy(float), 0.0)
    p_u = one_sided_t_pvalue(d["PUC_contribution"].to_numpy(float), 0.0)
    p_n = one_sided_t_pvalue(d["NPI_contribution"].to_numpy(float), 0.0)
    return {"movement": p_m, "utility": p_u, "retention": p_n, "candidate_iut": iut_pvalue((p_m, p_u, p_n))}


def run_lopo_matrix(df: pd.DataFrame, states: list[str], bootstrap: int = 2000):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    patient_rows = []
    for seed in SEEDS:
        seed_iut = {}
        block = []
        for cand_name in CANDIDATES:
            t0 = time.time()
            cand = make_candidate(cand_name, states)
            out = run_generic_lopo(
                df,
                cand,
                bootstrap=bootstrap,
                seed=seed,
                patient_balanced=True,
                states=states,
                reference_rule="MeanDelta",
                orthogonal_profiles=False,
                return_predictions=False,
                movement_contrast_method="t",
                use_movement_contrast_for_core=True,
            )
            p = diag_pvalues(out)
            seed_iut[cand_name] = p["candidate_iut"]
            r = {
                "seed": seed,
                "bootstrap": bootstrap,
                "candidate": cand_name,
                "n_patients": int(out["patients"]),
                "candidate_risk": float(out["reference_risk"] - out["PUC"]),
                "reference_risk": float(out["reference_risk"]),
                "utility": float(out["PUC"]),
                "utility_lcb": float(out["PUC_lo"]),
                "utility_ucb": float(out["PUC_hi"]),
                "retention": float(out["NPI"]),
                "retention_lcb": float(out["NPI_lo"]),
                "retention_ucb": float(out["NPI_hi"]),
                "movement": float(out["movement_margin_evidence"]["contrast"]),
                "movement_lcb": float(out["movement_margin_evidence"]["contrast_lcb"]),
                "movement_ucb": float(out["movement_margin_evidence"].get("contrast_ucb", np.nan)),
                "state": str(out["core_evidence_state"]),
                "core_qualified": bool(out["core_qualified"]),
                "retention_stability": float(out["retention_stability"]),
                "p_m_tdiag": p["movement"],
                "p_u_tdiag": p["utility"],
                "p_n_tdiag": p["retention"],
                "p_candidate_iut_tdiag": p["candidate_iut"],
                "runtime_s": float(time.time() - t0),
            }
            block.append(r)
            q = out["patient_influence"].copy()
            q.insert(0, "candidate", cand_name)
            q.insert(0, "seed", seed)
            patient_rows.append(q)
        holm = holm_rejections(seed_iut, alpha=0.05)
        for r in block:
            rej, rank, thr = holm[r["candidate"]]
            r["holm_rank_tdiag"] = int(rank)
            r["holm_threshold_tdiag"] = float(thr)
            r["holm_rejected_tdiag"] = bool(rej)
            rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "lopo_4candidate_4seed_2000.csv", index=False)
    pd.concat(patient_rows, ignore_index=True).to_csv(OUT / "lopo_patient_influence.csv", index=False)


def honest_one(df: pd.DataFrame, split: pd.DataFrame, states: list[str], cand_name: str, *, selector_seed: int = 20260903, bootstrap: int = 2000):
    merge = df.merge(split, on="patient_id", how="left", validate="one_to_one")
    if merge["split"].isna().any():
        raise RuntimeError("split is incomplete")
    dev = merge[merge.split == "development"].drop(columns="split").reset_index(drop=True)
    con = merge[merge.split == "confirmation"].drop(columns="split").reset_index(drop=True)
    if dev.patient_id.nunique() != 20 or con.patient_id.nunique() != 13:
        raise RuntimeError("frozen 20/13 split changed")

    cand_proto = make_candidate(cand_name, states)
    # Development-only OOF predictions select the retained mixing weight.
    po, no = _inner_oof_generic(dev, cand_proto, patient_balanced=True, states=states, reference_rule="MeanDelta")
    _, Sdev, Tdev = feature_matrix(dev, states=states)
    Pdev = dev.patient_id.astype(str).to_numpy()
    grid = np.round(np.arange(0.0, 1.00001, 0.01), 2)
    _, D, R = patient_curves(Tdev, no, po, Pdev, grid)
    sel = one_se_msw(D, R, grid, alpha=0.05, delta=0.0, B=bootstrap, seed=selector_seed, lam=1.0)
    lam = float(sel["weight"])

    # Fit candidate/reference only on development patients.  Confirmation targets
    # are deliberately absent from the prediction dataframe.
    fitted = cand_proto.fresh().fit(dev)
    target_cols = [f"target__{s}" for s in states]
    con_source_only = con.drop(columns=target_cols).copy()
    cpred = fitted.predict(con_source_only)
    _, Sconf, Tconf = feature_matrix(con, states=states)
    _, ref_model = _resolve_reference("MeanDelta", Sdev, Tdev, Pdev, patient_balanced=True)
    rpred = predict_null(ref_model, Sconf)
    retained = (1.0 - lam) * rpred + lam * cpred

    hres, effects = honest_confirm_predictions(
        Tconf, Sconf, rpred, cpred, retained, con.patient_id.astype(str).to_numpy(), movement_margin=0.01, alpha=0.05
    )
    hp = honest_confirmation_pvalues_localized(
        effects["movement_contrast"], effects["movement_numerator"], effects["utility"], effects["utility_abs_bound"],
        effects["retention"], effects["retention_abs_bound"], movement_margin=0.01,
    )
    # Conventional confirmation performance, patient unit = row here.
    brier_c = np.sum((Tconf - cpred) ** 2, axis=1)
    brier_r = np.sum((Tconf - rpred) ** 2, axis=1)
    row = {
        "candidate": cand_name,
        "n_development": int(dev.patient_id.nunique()),
        "n_confirmation": int(con.patient_id.nunique()),
        "selector_seed": int(selector_seed),
        "selector_bootstrap": int(bootstrap),
        "selected_lambda": lam,
        "candidate_mae": float(mae(Tconf, cpred).mean()),
        "reference_mae": float(mae(Tconf, rpred).mean()),
        "candidate_hellinger": float(hellinger(Tconf, cpred).mean()),
        "reference_hellinger": float(hellinger(Tconf, rpred).mean()),
        "candidate_brier": float(brier_c.mean()),
        "reference_brier": float(brier_r.mean()),
        "movement": float(hres.movement.estimate),
        "movement_lcb": float(hres.movement.lower_bound),
        "utility": float(hres.utility.estimate),
        "utility_lcb": float(hres.utility.lower_bound),
        "retention": float(hres.retention.estimate),
        "retention_lcb": float(hres.retention.lower_bound),
        "honest_qualified": bool(hres.qualified),
        "p_m_honest": float(hp["movement"]),
        "p_u_honest": float(hp["utility"]),
        "p_n_honest": float(hp["retention"]),
        "p_candidate_iut_honest": float(hp["candidate_iut"]),
    }
    pred = pd.DataFrame({"patient_id": con.patient_id.astype(str)})
    for j, s in enumerate(states):
        pred[f"target__{s}"] = Tconf[:, j]
        pred[f"source__{s}"] = Sconf[:, j]
        pred[f"reference__{s}"] = rpred[:, j]
        pred[f"candidate__{s}"] = cpred[:, j]
        pred[f"retained__{s}"] = retained[:, j]
    for k, v in effects.items():
        pred[k] = np.asarray(v, float)
    pred.insert(1, "candidate", cand_name)
    return row, pred


def run_honest(df: pd.DataFrame, split: pd.DataFrame, states: list[str], bootstrap: int = 2000, selector_seed: int = 20260903):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    preds = []
    for cand in CANDIDATES:
        t0 = time.time()
        row, pred = honest_one(df, split, states, cand, selector_seed=selector_seed, bootstrap=bootstrap)
        row["runtime_s"] = float(time.time() - t0)
        rows.append(row)
        preds.append(pred)
    holm = holm_rejections({r["candidate"]: r["p_candidate_iut_honest"] for r in rows}, alpha=0.05)
    for r in rows:
        rej, rank, thr = holm[r["candidate"]]
        r["holm_rank"] = int(rank)
        r["holm_threshold"] = float(thr)
        r["holm_rejected"] = bool(rej)
        r["family_qualified"] = bool(r["honest_qualified"] and rej)
    pd.DataFrame(rows).to_csv(OUT / "honest_20dev_13confirmation.csv", index=False)
    pd.concat(preds, ignore_index=True).to_csv(OUT / "honest_confirmation_predictions_and_effects.csv", index=False)


def summarize():
    out = {}
    lp = OUT / "lopo_4candidate_4seed_2000.csv"
    hp = OUT / "honest_20dev_13confirmation.csv"
    if lp.exists():
        d = pd.read_csv(lp)
        out["lopo"] = {
            "runs": int(len(d)),
            "state_counts_by_candidate": {c: g.state.value_counts().to_dict() for c, g in d.groupby("candidate")},
            "holm_rejections_by_candidate": {c: int(g.holm_rejected_tdiag.sum()) for c, g in d.groupby("candidate")},
            "risk_rank": d.groupby("candidate").candidate_risk.mean().sort_values().to_dict(),
        }
    if hp.exists():
        d = pd.read_csv(hp)
        out["honest"] = {
            "n_candidates": int(len(d)),
            "honest_qualified": int(d.honest_qualified.sum()),
            "family_qualified": int(d.family_qualified.sum()),
            "rows": d.to_dict(orient="records"),
        }
    (OUT / "summary.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["lopo", "honest", "all", "summary"], default="all")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    df = pd.read_csv(DATA)
    split = pd.read_csv(SPLIT)
    states = states_of(df)
    if args.mode in ("lopo", "all"):
        run_lopo_matrix(df, states, bootstrap=args.bootstrap)
    if args.mode in ("honest", "all"):
        run_honest(df, split, states, bootstrap=args.bootstrap)
    summarize()
    print((OUT / "summary.json").read_text())


if __name__ == "__main__":
    main()
