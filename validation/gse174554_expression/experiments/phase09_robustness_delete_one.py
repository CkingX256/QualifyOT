from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
AUDIT = ROOT / "audit"
RESULTS = ROOT / "results" / "robustness"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from phase07_honest_confirmation import development_oof, source_only
from phase08_lopo import compute_outer_fold, lopo_cache_path
from qualifyot.experiment import bootstrap_contrast, bootstrap_ratio
from qualifyot.inference import heterogeneous_hoeffding_lcb, honest_confirm_predictions
from qualifyot.model import feature_matrix, fit_null, hellinger, mae, predict_null
from qualifyot.weight_selection import one_se_msw, patient_curves
from qualifyot_gse174554.pipeline import CANDIDATE_ORDER, STATES, candidate_family, load_processed_inputs


PRIMARY_SEED = 20260903
SEEDS = [20260903, 20261003, 20261103, 20261203]
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


def get_candidate(store, name: str):
    return next(candidate for candidate in candidate_family(store) if candidate.name == name)


def fit_honest_once(pairs: pd.DataFrame, candidate, bootstrap: int, seed: int) -> dict[str, Any]:
    development = pairs.loc[pairs.analysis_partition == "development"].reset_index(drop=True)
    confirmation = pairs.loc[pairs.analysis_partition == "confirmation"].reset_index(drop=True)
    if len(development) < 3 or len(confirmation) < 2:
        raise RuntimeError("delete-one leaves insufficient development/confirmation patients")
    _, source_dev, target_dev = feature_matrix(development, states=STATES)
    _, source_conf, target_conf = feature_matrix(confirmation, states=STATES)
    dev_patients = development.patient_id.astype(str).to_numpy()
    conf_patients = confirmation.patient_id.astype(str).to_numpy()
    candidate_oof, reference_oof, _ = development_oof(development, candidate)
    _, harm, risk = patient_curves(target_dev, reference_oof, candidate_oof, dev_patients, GRID)
    selection = one_se_msw(harm, risk, GRID, alpha=0.05, delta=0.0, B=bootstrap, seed=seed, lam=1.0)
    fitted = candidate.fresh().fit(development)
    candidate_prediction = fitted.predict(source_only(confirmation))
    reference_model = fit_null("MeanDelta", source_dev, target_dev, patients=dev_patients, patient_balanced=True)
    reference = predict_null(reference_model, source_conf)
    retained = (1 - selection["weight"]) * reference + selection["weight"] * candidate_prediction
    honest, _axes = honest_confirm_predictions(target_conf, source_conf, reference, candidate_prediction, retained, conf_patients, movement_margin=0.01, alpha=0.05)
    result = honest.to_dict()
    return {
        "selected_lambda": float(selection["weight"]),
        "movement_estimate": result["movement"]["estimate"],
        "movement_lcb": result["movement"]["lower_bound"],
        "utility_estimate": result["utility"]["estimate"],
        "utility_lcb": result["utility"]["lower_bound"],
        "retention_estimate": result["retention"]["estimate"],
        "retention_lcb": result["retention"]["lower_bound"],
        "qualified": bool(honest.qualified),
        "confirmation_prediction_bytes": np.concatenate([reference.ravel(), candidate_prediction.ravel(), retained.ravel()]).astype(np.float64).tobytes(),
        "confirmation_patients": conf_patients.tolist(),
    }


def delete_one_task(pairs, store, candidate_name: str, deleted_patient: str) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        reduced = pairs.loc[pairs.patient_id.astype(str) != deleted_patient].reset_index(drop=True)
        result = fit_honest_once(reduced, get_candidate(store, candidate_name), bootstrap=1000, seed=PRIMARY_SEED)
        result.pop("confirmation_prediction_bytes")
        return {"candidate": candidate_name, "deleted_patient": deleted_patient, "deleted_partition": str(pairs.loc[pairs.patient_id.astype(str) == deleted_patient, "analysis_partition"].iloc[0]), "status": "SUCCESS", "runtime_seconds": time.perf_counter() - started, **result, "traceback": ""}
    except Exception:
        return {"candidate": candidate_name, "deleted_patient": deleted_patient, "deleted_partition": str(pairs.loc[pairs.patient_id.astype(str) == deleted_patient, "analysis_partition"].iloc[0]), "status": "FAILED", "runtime_seconds": time.perf_counter() - started, "selected_lambda": "", "movement_estimate": "", "movement_lcb": "", "utility_estimate": "", "utility_lcb": "", "retention_estimate": "", "retention_lcb": "", "qualified": "", "confirmation_patients": "", "traceback": traceback.format_exc()}


def bootstrap_stability(lopo: pd.DataFrame, candidates: list[str]) -> list[dict[str, Any]]:
    rows = []
    for candidate_name in candidates:
        for seed in SEEDS:
            block = lopo.loc[(lopo.candidate == candidate_name) & (lopo.seed == seed)].sort_values("patient_id")
            patients = block.patient_id.astype(str).to_numpy()
            source = block[[f"source__{s}" for s in STATES]].to_numpy(float)
            target = block[[f"target__{s}" for s in STATES]].to_numpy(float)
            reference = block[[f"reference__{s}" for s in STATES]].to_numpy(float)
            candidate = block[[f"candidate__{s}" for s in STATES]].to_numpy(float)
            retained = block[[f"retained__{s}" for s in STATES]].to_numpy(float)
            for bootstrap in (250, 500, 1000, 2000):
                pdr = bootstrap_ratio(hellinger(candidate, source), hellinger(target, source), patients, bootstrap, seed + 700)
                puc = bootstrap_contrast(mae(target, reference) - mae(target, candidate), patients, bootstrap, seed + 701)
                npi = bootstrap_contrast(mae(target, reference) - mae(target, retained), patients, bootstrap, seed + 703)
                rows.append({"candidate": candidate_name, "seed": seed, "bootstrap": bootstrap, "PDR": pdr[0], "PDR_lo": pdr[1], "PDR_hi": pdr[2], "PUC": puc[0], "PUC_lo": puc[1], "PUC_hi": puc[2], "NPI": npi[0], "NPI_lo": npi[1], "NPI_hi": npi[2], "state": "SUPPORTED" if pdr[1] > 0.01 and puc[1] > 0 and npi[1] > 0 else "NOT_SUPPORTED"})
    return rows


def confirmation_sensitivities(pairs: pd.DataFrame, honest_predictions: pd.DataFrame, candidates: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    development = pairs.loc[pairs.analysis_partition == "development"].reset_index(drop=True)
    _, source_dev, target_dev = feature_matrix(development, states=STATES)
    dev_patients = development.patient_id.astype(str).to_numpy()
    reference_models = {
        kind: fit_null(kind, source_dev, target_dev, patients=dev_patients, patient_balanced=True)
        for kind in ("Persistence", "CohortMean", "MeanDelta")
    }
    reference_rows, loss_rows, tail_rows = [], [], []
    for candidate_name in candidates:
        block = honest_predictions.loc[honest_predictions.candidate == candidate_name].sort_values("patient_id")
        patients = block.patient_id.astype(str).to_numpy()
        source = block[[f"source__{s}" for s in STATES]].to_numpy(float)
        target = block[[f"target__{s}" for s in STATES]].to_numpy(float)
        candidate = block[[f"candidate__{s}" for s in STATES]].to_numpy(float)
        retained = block[[f"retained__{s}" for s in STATES]].to_numpy(float)
        for kind, model in reference_models.items():
            reference = predict_null(model, source)
            result, _ = honest_confirm_predictions(target, source, reference, candidate, retained, patients, movement_margin=0.01, alpha=0.05)
            out = result.to_dict()
            reference_rows.append({"candidate": candidate_name, "reference": kind, "candidate_refit": False, "lambda_reselected": False, "retained_prediction_changed": False, "movement_lcb": out["movement"]["lower_bound"], "utility": out["utility"]["estimate"], "utility_lcb": out["utility"]["lower_bound"], "retention": out["retention"]["estimate"], "retention_lcb": out["retention"]["lower_bound"], "state": "SUPPORTED" if result.qualified else "NOT_SUPPORTED"})
        primary_reference = block[[f"reference__{s}" for s in STATES]].to_numpy(float)
        movement_lcb = float(next(iter(block.movement_contrast), float("nan")))
        for loss_name in ("MAE", "Brier", "Hellinger"):
            if loss_name == "MAE":
                loss = mae
            elif loss_name == "Brier":
                loss = lambda y, p: np.mean((y - p) ** 2, axis=1)
            else:
                loss = hellinger
            utility = loss(target, primary_reference) - loss(target, candidate)
            retention = loss(target, primary_reference) - loss(target, retained)
            lo = -np.ones(len(utility)); hi = np.ones(len(utility))
            utility_lcb = heterogeneous_hoeffding_lcb(utility, lo, hi, alpha=0.05)
            retention_lcb = heterogeneous_hoeffding_lcb(retention, lo, hi, alpha=0.05)
            loss_rows.append({"candidate": candidate_name, "loss": loss_name, "predictions_refit": False, "lambda_reselected": False, "utility": float(utility.mean()), "utility_lcb": utility_lcb, "retention": float(retention.mean()), "retention_lcb": retention_lcb, "state": "SUPPORTED" if utility_lcb > 0 and retention_lcb > 0 else "NOT_SUPPORTED", "CII_Q_descriptive_only": float(min(utility_lcb, retention_lcb))})
        harm = mae(target, retained) - mae(target, primary_reference)
        k = max(1, math.ceil(0.20 * len(harm)))
        observed = {"mean_harm": float(harm.mean()), "fraction_harmed": float(np.mean(harm > 0)), "worst_patient_harm": float(harm.max()), "cvar20_harm": float(np.sort(harm)[-k:].mean())}
        rng = np.random.default_rng(PRIMARY_SEED)
        draws = {key: [] for key in observed}
        for _ in range(5000):
            sampled = harm[rng.integers(0, len(harm), len(harm))]
            kk = max(1, math.ceil(0.20 * len(sampled)))
            values = {"mean_harm": sampled.mean(), "fraction_harmed": np.mean(sampled > 0), "worst_patient_harm": sampled.max(), "cvar20_harm": np.sort(sampled)[-kk:].mean()}
            for key, value in values.items(): draws[key].append(float(value))
        for metric, estimate in observed.items():
            tail_rows.append({"candidate": candidate_name, "seed": PRIMARY_SEED, "metric": metric, "estimate": estimate, "bootstrap_lo": float(np.quantile(draws[metric], 0.025)), "bootstrap_hi": float(np.quantile(draws[metric], 0.975)), "bootstrap": 5000, "role": "PRIMARY_TAIL_SAFETY"})
    return reference_rows, loss_rows, tail_rows


def main() -> int:
    started = now()
    lopo_phase = json.loads((LOGS / "phase08_lopo.json").read_text(encoding="utf-8"))
    if lopo_phase.get("status") not in {"SUCCESS", "PARTIAL"}:
        raise RuntimeError("LOPO phase has no usable output")
    pairs, store = load_processed_inputs(ROOT)
    winner = json.loads((ROOT / "results" / "dependent_lopo" / "accuracy_winner.json").read_text(encoding="utf-8"))["candidate"]
    key_candidates = list(dict.fromkeys(["ExpressionElasticNet", winner]))
    lopo = pd.read_csv(ROOT / "results" / "dependent_lopo" / "lopo_patient_predictions.csv")
    honest = pd.read_csv(ROOT / "results" / "honest_confirmation" / "honest_confirmation_predictions_long.csv")
    stability = bootstrap_stability(lopo, key_candidates)
    write_csv(RESULTS / "bootstrap_stability.csv", stability, ["candidate", "seed", "bootstrap", "PDR", "PDR_lo", "PDR_hi", "PUC", "PUC_lo", "PUC_hi", "NPI", "NPI_lo", "NPI_hi", "state"])
    reference_rows, loss_rows, tail_rows = confirmation_sensitivities(pairs, honest, key_candidates)
    write_csv(RESULTS / "reference_sensitivity.csv", reference_rows, list(reference_rows[0]))
    write_csv(RESULTS / "loss_sensitivity.csv", loss_rows, list(loss_rows[0]))
    write_csv(RESULTS / "tail_safety.csv", tail_rows, list(tail_rows[0]))

    checkpoint_path = RESULTS / "delete_one_checkpoint.csv"
    checkpoint_context_path = RESULTS / "delete_one_checkpoint_context.json"
    patient_ids = sorted(set(pairs.patient_id.astype(str)))
    patient_hash = hashlib.sha256("\n".join(patient_ids).encode("utf-8")).hexdigest()
    existing: list[dict[str, Any]] = []
    checkpoint_context = json.loads(checkpoint_context_path.read_text(encoding="utf-8")) if checkpoint_context_path.exists() else {}
    if checkpoint_path.exists() and checkpoint_context.get("patient_hash") == patient_hash and checkpoint_context.get("key_candidates") == key_candidates:
        existing = list(csv.DictReader(checkpoint_path.open(encoding="utf-8", newline="")))
    else:
        atomic_json(checkpoint_context_path, {"patient_hash": patient_hash, "patient_count": len(patient_ids), "key_candidates": key_candidates, "scientific_configuration_unchanged": True, "updated_utc": now()})
    done = {(row["candidate"], row["deleted_patient"]) for row in existing}
    tasks = [(candidate, patient) for candidate in key_candidates for patient in patient_ids if (candidate, patient) not in done]
    delete_rows = existing[:]
    fields = ["candidate", "deleted_patient", "deleted_partition", "status", "runtime_seconds", "selected_lambda", "movement_estimate", "movement_lcb", "utility_estimate", "utility_lcb", "retention_estimate", "retention_lcb", "qualified", "confirmation_patients", "traceback"]
    for start in range(0, len(tasks), 4):
        batch = tasks[start:start + 4]
        results = Parallel(n_jobs=4, backend="loky")(delayed(delete_one_task)(pairs, store, candidate, patient) for candidate, patient in batch)
        for result in results:
            if isinstance(result.get("confirmation_patients"), list):
                result["confirmation_patients"] = "|".join(result["confirmation_patients"])
            delete_rows.append(result)
            write_csv(checkpoint_path, delete_rows, fields)
        atomic_json(LOGS / "phase09_delete_one_progress.json", {"completed": len(delete_rows), "expected": len(key_candidates) * len(pairs), "failed": sum(row["status"] != "SUCCESS" for row in delete_rows), "updated_utc": now()})

    determinism_rows = []
    for candidate_name in key_candidates:
        original = fit_honest_once(pairs, get_candidate(store, candidate_name), bootstrap=5000, seed=PRIMARY_SEED)
        rerun = fit_honest_once(pairs, get_candidate(store, candidate_name), bootstrap=5000, seed=PRIMARY_SEED)
        numeric_keys = ["selected_lambda", "movement_estimate", "movement_lcb", "utility_estimate", "utility_lcb", "retention_estimate", "retention_lcb"]
        max_difference = max(abs(float(original[key]) - float(rerun[key])) for key in numeric_keys)
        bytes_equal = original["confirmation_prediction_bytes"] == rerun["confirmation_prediction_bytes"]
        determinism_rows.append({"candidate": candidate_name, "analysis": "Honest", "max_numeric_difference": max_difference, "prediction_bytes_identical": bytes_equal, "state_identical": original["qualified"] == rerun["qualified"], "status": "PASS" if max_difference <= 1e-12 and bytes_equal and original["qualified"] == rerun["qualified"] else "FAIL"})
        ordered_patients = sorted(set(pairs.patient_id.astype(str)))
        cache_path = lopo_cache_path(CANDIDATE_ORDER.index(candidate_name), candidate_name, ordered_patients)
        cached = joblib.load(cache_path)
        rerun_folds = Parallel(n_jobs=4, backend="loky")(delayed(compute_outer_fold)(pairs, get_candidate(store, candidate_name), patient) for patient in ordered_patients)
        cached_values = np.concatenate([np.concatenate([fold["candidate"], fold["reference"]]) for fold in cached])
        rerun_values = np.concatenate([np.concatenate([fold["candidate"], fold["reference"]]) for fold in rerun_folds])
        difference = float(np.max(np.abs(cached_values - rerun_values)))
        determinism_rows.append({"candidate": candidate_name, "analysis": "DEPENDENT-LOPO DIAGNOSTIC", "max_numeric_difference": difference, "prediction_bytes_identical": bool(np.array_equal(cached_values, rerun_values)), "state_identical": True, "status": "PASS" if difference <= 1e-12 else "FAIL"})
    write_csv(RESULTS / "determinism_reruns.csv", determinism_rows, ["candidate", "analysis", "max_numeric_difference", "prediction_bytes_identical", "state_identical", "status"])
    status = "SUCCESS" if all(row["status"] == "PASS" for row in determinism_rows) and all(row["status"] == "SUCCESS" for row in delete_rows) else "PARTIAL"
    payload = {"phase": "robustness_delete_one", "status": status, "started_utc": started, "finished_utc": now(), "key_candidates": key_candidates, "bootstrap_stability_rows": len(stability), "reference_sensitivity_rows": len(reference_rows), "loss_sensitivity_rows": len(loss_rows), "tail_safety_rows": len(tail_rows), "delete_one_success": sum(row["status"] == "SUCCESS" for row in delete_rows), "delete_one_failed": sum(row["status"] != "SUCCESS" for row in delete_rows), "determinism_pass": sum(row["status"] == "PASS" for row in determinism_rows), "determinism_fail": sum(row["status"] != "PASS" for row in determinism_rows)}
    atomic_json(LOGS / "phase09_robustness_delete_one.json", payload)
    atomic_json(LOGS / "pipeline_checkpoint.json", {"last_completed_phase": "robustness_delete_one", "status": status, "updated_utc": now()})
    atomic_text(AUDIT / "DETERMINISM_AUDIT.md", "\n".join(["# Determinism audit", "", f"- Status: `{status}`", f"- Honest and LOPO key-candidate checks passed: `{payload['determinism_pass']}`; failed: `{payload['determinism_fail']}`", "- Required numeric tolerance: `1e-12`."]) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if status == "SUCCESS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
