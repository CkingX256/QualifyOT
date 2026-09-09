from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
RESULTS = ROOT / "results" / "honest_confirmation"
AUDIT = ROOT / "audit"
sys.path.insert(0, str(ROOT / "src"))

from qualifyot.inference import honest_confirm_predictions, honest_confirmation_pvalues_localized
from qualifyot.model import feature_matrix, fit_null, hellinger, mae, predict_null
from qualifyot.weight_selection import one_se_msw, patient_curves
from qualifyot_gse174554.pipeline import CANDIDATE_ORDER, STATES, candidate_family, load_processed_inputs


SEED = 20260903
BOOTSTRAP = 5000
GRID = np.round(np.arange(0.0, 1.00001, 0.01), 2)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=path.parent) as f:
        f.write(text)
        tmp = Path(f.name)
    os.replace(tmp, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=path.parent) as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        tmp = Path(f.name)
    os.replace(tmp, path)


def source_only(frame):
    return frame.drop(columns=[column for column in frame.columns if column.startswith("target__")])


def development_oof(frame, candidate):
    _, source, target = feature_matrix(frame, states=STATES)
    patients = frame.patient_id.astype(str).to_numpy()
    candidate_predictions = np.zeros_like(target)
    reference_predictions = np.zeros_like(target)
    diagnostics = []
    for fold, heldout in enumerate(sorted(set(patients))):
        test_mask = patients == heldout
        train_mask = ~test_mask
        train = frame.loc[train_mask].reset_index(drop=True)
        test = frame.loc[test_mask].reset_index(drop=True)
        fitted = candidate.fresh().fit(train)
        candidate_predictions[test_mask] = fitted.predict(source_only(test))
        reference_model = fit_null("MeanDelta", source[train_mask], target[train_mask], patients=patients[train_mask], patient_balanced=True)
        reference_predictions[test_mask] = predict_null(reference_model, source[test_mask])
        diagnostics.append({"fold": fold, "heldout_patient": heldout, "candidate_diagnostics": fitted.candidate_diagnostics() if hasattr(fitted, "candidate_diagnostics") else {}})
    return candidate_predictions, reference_predictions, diagnostics


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict[str, Any]]:
    ordered = sorted(pvalues.items(), key=lambda item: (item[1], CANDIDATE_ORDER.index(item[0])))
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    rejection_open = True
    rejected: dict[str, bool] = {}
    for index, (name, pvalue) in enumerate(ordered):
        running = max(running, (m - index) * pvalue)
        adjusted[name] = min(1.0, running)
        threshold = alpha / (m - index)
        reject = rejection_open and pvalue <= threshold
        rejected[name] = reject
        if not reject:
            rejection_open = False
    return {name: {"holm_adjusted_p": adjusted[name], "holm_reject": rejected[name]} for name in pvalues}


def main() -> int:
    started = now()
    leakage = json.loads((LOGS / "phase06_leakage_gate.json").read_text(encoding="utf-8"))
    if leakage.get("status") != "SUCCESS":
        raise RuntimeError("Leakage gate is not SUCCESS")
    pairs, store = load_processed_inputs(ROOT)
    development = pairs.loc[pairs.analysis_partition == "development"].reset_index(drop=True)
    confirmation = pairs.loc[pairs.analysis_partition == "confirmation"].reset_index(drop=True)
    _, source_dev, target_dev = feature_matrix(development, states=STATES)
    _, source_conf, target_conf = feature_matrix(confirmation, states=STATES)
    dev_patients = development.patient_id.astype(str).to_numpy()
    conf_patients = confirmation.patient_id.astype(str).to_numpy()
    summary_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    pvalues: dict[str, float] = {}
    for order, candidate in enumerate(candidate_family(store)):
        t0 = time.perf_counter()
        try:
            candidate_oof, reference_oof, fold_diagnostics = development_oof(development, candidate)
            _, harm_curves, risk_curves = patient_curves(target_dev, reference_oof, candidate_oof, dev_patients, GRID)
            selection = one_se_msw(harm_curves, risk_curves, GRID, alpha=0.05, delta=0.0, B=BOOTSTRAP, seed=SEED, lam=1.0)
            weight = float(selection["weight"])
            full_candidate = candidate.fresh().fit(development)
            full_reference = fit_null("MeanDelta", source_dev, target_dev, patients=dev_patients, patient_balanced=True)
            candidate_conf = full_candidate.predict(source_only(confirmation))
            reference_conf = predict_null(full_reference, source_conf)
            retained_conf = (1.0 - weight) * reference_conf + weight * candidate_conf
            honest, axes = honest_confirm_predictions(target_conf, source_conf, reference_conf, candidate_conf, retained_conf, conf_patients, movement_margin=0.01, alpha=0.05)
            pvalue_result = honest_confirmation_pvalues_localized(
                axes["movement_contrast"], axes["movement_numerator"], axes["utility"], axes["utility_abs_bound"], axes["retention"], axes["retention_abs_bound"], movement_margin=0.01
            )
            pvalues[candidate.name] = float(pvalue_result["candidate_iut"])
            result = honest.to_dict()
            movement_num = hellinger(candidate_conf, source_conf)
            movement_den = hellinger(target_conf, source_conf)
            reference_loss = mae(target_conf, reference_conf)
            candidate_loss = mae(target_conf, candidate_conf)
            retained_loss = mae(target_conf, retained_conf)
            summary_rows.append(
                {
                    "candidate_order": order,
                    "candidate": candidate.name,
                    "status": "SUCCESS",
                    "development_patients": len(development),
                    "confirmation_patients": len(confirmation),
                    "selected_lambda": weight,
                    "best_safe_lambda": selection["best_weight"],
                    "max_safe_lambda": selection["max_eligible_weight"],
                    "max_t_q": selection["max_t_q"],
                    "movement_estimate": result["movement"]["estimate"],
                    "movement_lcb": result["movement"]["lower_bound"],
                    "utility_estimate": result["utility"]["estimate"],
                    "utility_lcb": result["utility"]["lower_bound"],
                    "retention_estimate": result["retention"]["estimate"],
                    "retention_lcb": result["retention"]["lower_bound"],
                    "candidate_iut_p": pvalue_result["candidate_iut"],
                    "honest_qualified_unadjusted": honest.qualified,
                    "reference_mae": float(reference_loss.mean()),
                    "candidate_mae": float(candidate_loss.mean()),
                    "retained_mae": float(retained_loss.mean()),
                    "movement_ratio_descriptive": float(movement_num.mean() / movement_den.mean()) if movement_den.mean() > 0 else float("nan"),
                    "runtime_seconds": time.perf_counter() - t0,
                    "full_fit_diagnostics_json": json.dumps(full_candidate.candidate_diagnostics() if hasattr(full_candidate, "candidate_diagnostics") else {}, sort_keys=True),
                    "fold_diagnostics_sha256": hashlib.sha256(json.dumps(fold_diagnostics, sort_keys=True).encode()).hexdigest(),
                }
            )
            for i, patient_id in enumerate(conf_patients):
                row = {"candidate": candidate.name, "patient_id": patient_id, "pair_id": confirmation.pair_id.iloc[i], "selected_lambda": weight, "movement_numerator": axes["movement_numerator"][i], "movement_contrast": axes["movement_contrast"][i], "utility": axes["utility"][i], "retention": axes["retention"][i], "utility_abs_bound": axes["utility_abs_bound"][i], "retention_abs_bound": axes["retention_abs_bound"][i]}
                for j, state in enumerate(STATES):
                    row.update({f"source__{state}": source_conf[i, j], f"target__{state}": target_conf[i, j], f"reference__{state}": reference_conf[i, j], f"candidate__{state}": candidate_conf[i, j], f"retained__{state}": retained_conf[i, j]})
                prediction_rows.append(row)
            for i, patient_id in enumerate(dev_patients):
                row = {"candidate": candidate.name, "patient_id": patient_id}
                for j, state in enumerate(STATES):
                    row.update({f"target__{state}": target_dev[i, j], f"reference_oof__{state}": reference_oof[i, j], f"candidate_oof__{state}": candidate_oof[i, j]})
                oof_rows.append(row)
            for i, weight_grid in enumerate(GRID):
                grid_rows.append({"candidate": candidate.name, "lambda": weight_grid, "harm_ucb": selection["ucb"][i], "risk": selection["risk"][i], "selected": abs(weight_grid - weight) <= 1e-12})
            atomic_json(RESULTS / f"{order+1:02d}_{candidate.name}_diagnostics.json", {"folds": fold_diagnostics, "full_fit": full_candidate.candidate_diagnostics() if hasattr(full_candidate, "candidate_diagnostics") else {}, "selection": {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in selection.items()}, "honest": result, "pvalues": pvalue_result})
        except Exception:
            failure_rows.append({"candidate_order": order, "candidate": candidate.name, "status": "FAILED", "runtime_seconds": time.perf_counter() - t0, "traceback": traceback.format_exc()})
    corrections = holm(pvalues)
    for row in summary_rows:
        row.update(corrections[row["candidate"]])
        row["honest_qualified_holm"] = bool(row["honest_qualified_unadjusted"] and row["holm_reject"])
        row["evidence_state"] = "QUALIFIED_HOLM" if row["honest_qualified_holm"] else ("QUALIFIED_UNADJUSTED_ONLY" if row["honest_qualified_unadjusted"] else "NOT_QUALIFIED")
    summary_fields = ["candidate_order", "candidate", "status", "development_patients", "confirmation_patients", "selected_lambda", "best_safe_lambda", "max_safe_lambda", "max_t_q", "movement_estimate", "movement_lcb", "utility_estimate", "utility_lcb", "retention_estimate", "retention_lcb", "candidate_iut_p", "holm_adjusted_p", "holm_reject", "honest_qualified_unadjusted", "honest_qualified_holm", "evidence_state", "reference_mae", "candidate_mae", "retained_mae", "movement_ratio_descriptive", "runtime_seconds", "full_fit_diagnostics_json", "fold_diagnostics_sha256"]
    write_csv(RESULTS / "honest_candidate_summary.csv", summary_rows, summary_fields)
    pred_fields = ["candidate", "patient_id", "pair_id", "selected_lambda", "movement_numerator", "movement_contrast", "utility", "retention", "utility_abs_bound", "retention_abs_bound"] + [f"{kind}__{state}" for state in STATES for kind in ("source", "target", "reference", "candidate", "retained")]
    write_csv(RESULTS / "honest_confirmation_predictions_long.csv", prediction_rows, pred_fields)
    oof_fields = ["candidate", "patient_id"] + [f"{kind}__{state}" for state in STATES for kind in ("target", "reference_oof", "candidate_oof")]
    write_csv(RESULTS / "development_oof_predictions.csv", oof_rows, oof_fields)
    write_csv(RESULTS / "development_retention_grid.csv", grid_rows, ["candidate", "lambda", "harm_ucb", "risk", "selected"])
    write_csv(RESULTS / "failed_candidates.csv", failure_rows, ["candidate_order", "candidate", "status", "runtime_seconds", "traceback"])
    complete = len(summary_rows) == 5 and not failure_rows
    status = "SUCCESS" if complete else "PARTIAL"
    payload = {"phase": "honest_confirmation", "status": status, "started_utc": started, "finished_utc": now(), "primary_seed": SEED, "bootstrap": BOOTSTRAP, "development_patients": len(development), "confirmation_patients": len(confirmation), "candidate_success": len(summary_rows), "candidate_failed": len(failure_rows), "qualified_unadjusted": sum(bool(r["honest_qualified_unadjusted"]) for r in summary_rows), "qualified_holm": sum(bool(r["honest_qualified_holm"]) for r in summary_rows)}
    atomic_json(LOGS / "phase07_honest_confirmation.json", payload)
    atomic_json(LOGS / "pipeline_checkpoint.json", {"last_completed_phase": "honest_confirmation", "status": status, "updated_utc": now()})
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if status == "SUCCESS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
