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

import joblib
import numpy as np
from joblib import Parallel, delayed


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
RESULTS = ROOT / "results" / "dependent_lopo"
CACHE = RESULTS / "cache"
sys.path.insert(0, str(ROOT / "src"))

from qualifyot.experiment import bootstrap_contrast, bootstrap_ratio
from qualifyot.model import feature_matrix, fit_null, hellinger, mae, predict_null
from qualifyot.weight_selection import one_se_msw, patient_curves
from qualifyot_gse174554.pipeline import CANDIDATE_ORDER, STATES, candidate_family, load_processed_inputs


SEEDS = [20260903, 20261003, 20261103, 20261203]
BOOTSTRAP = 2000
GRID = np.round(np.arange(0.0, 1.00001, 0.01), 2)


def lopo_cache_path(candidate_order: int, candidate_name: str, patient_ids: list[str]) -> Path:
    patient_hash = hashlib.sha256("\n".join(patient_ids).encode("utf-8")).hexdigest()[:16]
    return CACHE / f"{candidate_order+1:02d}_{candidate_name}_{patient_hash}_fold_cache.joblib"


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


def compute_outer_fold(pairs, candidate, heldout: str) -> dict[str, Any]:
    patients = pairs.patient_id.astype(str).to_numpy()
    test_mask = patients == heldout
    train_mask = ~test_mask
    train = pairs.loc[train_mask].reset_index(drop=True)
    test = pairs.loc[test_mask].reset_index(drop=True)
    _, source_all, target_all = feature_matrix(pairs, states=STATES)
    outer_candidate = candidate.fresh().fit(train)
    candidate_test = outer_candidate.predict(source_only(test))
    reference_model = fit_null("MeanDelta", source_all[train_mask], target_all[train_mask], patients=patients[train_mask], patient_balanced=True)
    reference_test = predict_null(reference_model, source_all[test_mask])
    _, source_train, target_train = feature_matrix(train, states=STATES)
    train_patients = train.patient_id.astype(str).to_numpy()
    candidate_oof = np.zeros_like(target_train)
    reference_oof = np.zeros_like(target_train)
    for inner_heldout in sorted(set(train_patients)):
        inner_test = train_patients == inner_heldout
        inner_train = ~inner_test
        inner_fit = candidate.fresh().fit(train.loc[inner_train].reset_index(drop=True))
        candidate_oof[inner_test] = inner_fit.predict(source_only(train.loc[inner_test].reset_index(drop=True)))
        inner_reference = fit_null("MeanDelta", source_train[inner_train], target_train[inner_train], patients=train_patients[inner_train], patient_balanced=True)
        reference_oof[inner_test] = predict_null(inner_reference, source_train[inner_test])
    _, harm, risk = patient_curves(target_train, reference_oof, candidate_oof, train_patients, GRID)
    return {
        "heldout_patient": heldout,
        "source": source_all[test_mask][0],
        "target": target_all[test_mask][0],
        "candidate": candidate_test[0],
        "reference": reference_test[0],
        "harm_curves": harm,
        "risk_curves": risk,
        "outer_diagnostics": outer_candidate.candidate_diagnostics() if hasattr(outer_candidate, "candidate_diagnostics") else {},
    }


def main() -> int:
    started = now()
    honest = json.loads((LOGS / "phase07_honest_confirmation.json").read_text(encoding="utf-8"))
    if honest.get("status") not in {"SUCCESS", "PARTIAL"}:
        raise RuntimeError("Honest phase has not produced a usable status")
    pairs, store = load_processed_inputs(ROOT)
    patient_ids = sorted(set(pairs.patient_id.astype(str)))
    summaries: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    completed_candidates: list[str] = []
    for candidate_order, candidate in enumerate(candidate_family(store)):
        t0 = time.perf_counter()
        cache_path = lopo_cache_path(candidate_order, candidate.name, patient_ids)
        try:
            if cache_path.exists():
                folds = joblib.load(cache_path)
                if [fold["heldout_patient"] for fold in folds] != patient_ids:
                    raise RuntimeError("LOPO cache patient ordering mismatch")
            else:
                folds = Parallel(n_jobs=4, backend="loky", verbose=5)(delayed(compute_outer_fold)(pairs, candidate, patient_id) for patient_id in patient_ids)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = cache_path.with_suffix(".tmp.joblib")
                joblib.dump(folds, tmp_path, compress=3)
                os.replace(tmp_path, cache_path)
            source = np.vstack([fold["source"] for fold in folds])
            target = np.vstack([fold["target"] for fold in folds])
            candidate_prediction = np.vstack([fold["candidate"] for fold in folds])
            reference = np.vstack([fold["reference"] for fold in folds])
            patient_array = np.asarray(patient_ids)
            for seed in SEEDS:
                weights = []
                retained = []
                for fold_index, fold in enumerate(folds):
                    selection = one_se_msw(fold["harm_curves"], fold["risk_curves"], GRID, alpha=0.05, delta=0.0, B=BOOTSTRAP, seed=seed + fold_index * 7919, lam=1.0)
                    weight = float(selection["weight"])
                    weights.append(weight)
                    retained.append((1.0 - weight) * fold["reference"] + weight * fold["candidate"])
                retained = np.vstack(retained)
                movement_num = hellinger(candidate_prediction, source)
                movement_den = hellinger(target, source)
                pdr = bootstrap_ratio(movement_num, movement_den, patient_array, BOOTSTRAP, seed + 700)
                puc = bootstrap_contrast(mae(target, reference) - mae(target, candidate_prediction), patient_array, BOOTSTRAP, seed + 701)
                npi = bootstrap_contrast(mae(target, reference) - mae(target, retained), patient_array, BOOTSTRAP, seed + 703)
                summaries.append(
                    {
                        "candidate_order": candidate_order,
                        "candidate": candidate.name,
                        "seed": seed,
                        "label": "DEPENDENT-LOPO DIAGNOSTIC",
                        "patients": len(patient_ids),
                        "bootstrap": BOOTSTRAP,
                        "PDR": pdr[0], "PDR_lo": pdr[1], "PDR_hi": pdr[2],
                        "PUC": puc[0], "PUC_lo": puc[1], "PUC_hi": puc[2],
                        "NPI": npi[0], "NPI_lo": npi[1], "NPI_hi": npi[2],
                        "candidate_mae": float(mae(target, candidate_prediction).mean()),
                        "reference_mae": float(mae(target, reference).mean()),
                        "retained_mae": float(mae(target, retained).mean()),
                        "positive_weight_fraction": float(np.mean(np.asarray(weights) > 0)),
                        "mean_lambda": float(np.mean(weights)),
                        "median_lambda": float(np.median(weights)),
                        "runtime_seconds_shared_fit": time.perf_counter() - t0,
                    }
                )
                for i, patient_id in enumerate(patient_ids):
                    row = {"candidate": candidate.name, "seed": seed, "patient_id": patient_id, "lambda": weights[i], "label": "DEPENDENT-LOPO DIAGNOSTIC", "reference_loss": mae(target[[i]], reference[[i]])[0], "candidate_loss": mae(target[[i]], candidate_prediction[[i]])[0], "retained_loss": mae(target[[i]], retained[[i]])[0]}
                    for j, state in enumerate(STATES):
                        row.update({f"source__{state}": source[i, j], f"target__{state}": target[i, j], f"reference__{state}": reference[i, j], f"candidate__{state}": candidate_prediction[i, j], f"retained__{state}": retained[i, j]})
                    predictions.append(row)
            completed_candidates.append(candidate.name)
            atomic_json(LOGS / "phase08_lopo_progress.json", {"status": "RUNNING", "completed_candidates": completed_candidates, "candidate_count": len(completed_candidates), "expected_candidates": 5, "updated_utc": now()})
        except Exception:
            failures.append({"candidate_order": candidate_order, "candidate": candidate.name, "status": "FAILED", "runtime_seconds": time.perf_counter() - t0, "traceback": traceback.format_exc()})
    primary_rows = [row for row in summaries if row["seed"] == SEEDS[0]]
    if primary_rows:
        winner = min(primary_rows, key=lambda row: (round(float(row["candidate_mae"]) / 1e-12) * 1e-12, CANDIDATE_ORDER.index(row["candidate"])))
        for row in summaries:
            row["accuracy_winner_primary_seed"] = row["candidate"] == winner["candidate"]
    else:
        winner = None
    summary_fields = ["candidate_order", "candidate", "seed", "label", "patients", "bootstrap", "PDR", "PDR_lo", "PDR_hi", "PUC", "PUC_lo", "PUC_hi", "NPI", "NPI_lo", "NPI_hi", "candidate_mae", "reference_mae", "retained_mae", "positive_weight_fraction", "mean_lambda", "median_lambda", "runtime_seconds_shared_fit", "accuracy_winner_primary_seed"]
    write_csv(RESULTS / "lopo_seed_summary.csv", summaries, summary_fields)
    pred_fields = ["candidate", "seed", "patient_id", "lambda", "label", "reference_loss", "candidate_loss", "retained_loss"] + [f"{kind}__{state}" for state in STATES for kind in ("source", "target", "reference", "candidate", "retained")]
    write_csv(RESULTS / "lopo_patient_predictions.csv", predictions, pred_fields)
    write_csv(RESULTS / "failed_candidates.csv", failures, ["candidate_order", "candidate", "status", "runtime_seconds", "traceback"])
    if winner is not None:
        atomic_json(RESULTS / "accuracy_winner.json", {"candidate": winner["candidate"], "candidate_mae": winner["candidate_mae"], "seed": SEEDS[0], "tie_tolerance": 1e-12, "tie_order": CANDIDATE_ORDER, "selection_role": "secondary_accuracy_winner_does_not_replace_prespecified_expression_candidate"})
    status = "SUCCESS" if len(completed_candidates) == 5 and not failures else "PARTIAL"
    payload = {"phase": "dependent_lopo", "status": status, "started_utc": started, "finished_utc": now(), "patients": len(patient_ids), "candidate_success": len(completed_candidates), "candidate_failed": len(failures), "seeds": SEEDS, "bootstrap_per_seed": BOOTSTRAP, "model_fits_shared_across_seeds": True, "accuracy_winner": winner["candidate"] if winner else None, "label": "DEPENDENT-LOPO DIAGNOSTIC"}
    atomic_json(LOGS / "phase08_lopo.json", payload)
    atomic_json(LOGS / "pipeline_checkpoint.json", {"last_completed_phase": "dependent_lopo", "status": status, "updated_utc": now()})
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if status == "SUCCESS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
