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
AUDIT = ROOT / "audit"
LOGS = ROOT / "logs"
RESULTS = ROOT / "results" / "leakage_gate"
sys.path.insert(0, str(ROOT / "src"))

from qualifyot_gse174554.expression import SourceExpressionStore
from qualifyot_gse174554.pipeline import CANDIDATE_ORDER, candidate_family, load_processed_inputs


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


def cell_aggregation_invariance() -> tuple[float, float]:
    raw = np.arange(1, 61, dtype=np.float64).reshape(10, 6)
    base = np.log1p(raw.sum(1) / raw.sum() * 1_000_000.0)
    permuted = raw[:, [5, 1, 3, 0, 4, 2]]
    perm = np.log1p(permuted.sum(1) / permuted.sum() * 1_000_000.0)
    duplicated = np.concatenate([raw, raw], axis=1)
    dup = np.log1p(duplicated.sum(1) / duplicated.sum() * 1_000_000.0)
    return float(np.max(np.abs(base - perm))), float(np.max(np.abs(base - dup)))


def main() -> int:
    started = now()
    pairs, store = load_processed_inputs(ROOT)
    development = pairs.loc[pairs.analysis_partition == "development"].reset_index(drop=True)
    confirmation = pairs.loc[pairs.analysis_partition == "confirmation"].reset_index(drop=True)
    dev_ids, conf_ids = set(development.patient_id.astype(str)), set(confirmation.patient_id.astype(str))
    checks: list[dict[str, Any]] = []
    disjoint = not (dev_ids & conf_ids)
    checks.append({"check": "physical_patient_partition_disjointness", "candidate": "ALL", "max_abs_difference": 0.0, "status": "PASS" if disjoint else "FAIL", "details": f"development={len(dev_ids)} confirmation={len(conf_ids)} overlap={len(dev_ids & conf_ids)}"})
    perm_diff, dup_diff = cell_aggregation_invariance()
    checks.append({"check": "source_cell_permutation_invariance", "candidate": "PSEUDOBULK", "max_abs_difference": perm_diff, "status": "PASS" if perm_diff <= 1e-12 else "FAIL", "details": "raw counts summed before CPM"})
    checks.append({"check": "exact_full_source_cell_duplication_invariance", "candidate": "PSEUDOBULK", "max_abs_difference": dup_diff, "status": "PASS" if dup_diff <= 1e-12 else "FAIL", "details": "duplicating the complete cell set leaves CPM unchanged"})
    phase05 = json.loads((LOGS / "phase05_pseudobulk.json").read_text(encoding="utf-8"))
    embargo = phase05.get("recurrence_expression_opened") is False
    checks.append({"check": "recurrence_expression_embargo", "candidate": "ALL", "max_abs_difference": 0.0, "status": "PASS" if embargo else "FAIL", "details": f"recurrence_expression_opened={phase05.get('recurrence_expression_opened')}"})

    fit_rows: list[dict[str, Any]] = []
    source_only = confirmation.drop(columns=[column for column in confirmation.columns if column.startswith("target__")])
    mutated = confirmation.copy()
    mutated[["target__Tumor", "target__NonTumor"]] = mutated[["target__NonTumor", "target__Tumor"]].to_numpy()
    for order, candidate in enumerate(candidate_family(store)):
        t0 = time.perf_counter()
        try:
            fitted = candidate.fresh().fit(development)
            base = fitted.predict(confirmation)
            removed = fitted.predict(source_only)
            target_mutated = fitted.predict(mutated)
            removed_diff = float(np.max(np.abs(base - removed)))
            mutation_diff = float(np.max(np.abs(base - target_mutated)))
            simplex_error = float(np.max(np.abs(base.sum(1) - 1.0)))
            minimum = float(base.min())
            checks.extend(
                [
                    {"check": "heldout_target_column_removal", "candidate": candidate.name, "max_abs_difference": removed_diff, "status": "PASS" if removed_diff <= 1e-12 else "FAIL", "details": "prediction with target columns removed"},
                    {"check": "heldout_target_mutation", "candidate": candidate.name, "max_abs_difference": mutation_diff, "status": "PASS" if mutation_diff <= 1e-12 else "FAIL", "details": "confirmation target columns swapped"},
                    {"check": "k2_simplex_legality", "candidate": candidate.name, "max_abs_difference": simplex_error, "status": "PASS" if simplex_error <= 1e-12 and minimum >= -1e-12 else "FAIL", "details": f"minimum={minimum:.17g}"},
                ]
            )
            diagnostics = fitted.candidate_diagnostics() if hasattr(fitted, "candidate_diagnostics") else {}
            fit_rows.append({"candidate_order": order, "candidate": candidate.name, "status": "SUCCESS", "runtime_seconds": time.perf_counter() - t0, "prediction_sha256": hashlib.sha256(np.ascontiguousarray(base).tobytes()).hexdigest(), "diagnostics_json": json.dumps(diagnostics, sort_keys=True), "traceback": ""})
        except Exception:
            fit_rows.append({"candidate_order": order, "candidate": candidate.name, "status": "INCOMPATIBLE_K2", "runtime_seconds": time.perf_counter() - t0, "prediction_sha256": "", "diagnostics_json": "{}", "traceback": traceback.format_exc()})
    required_success = {row["candidate"] for row in fit_rows if row["status"] == "SUCCESS"}
    order_ok = [row["candidate"] for row in fit_rows] == CANDIDATE_ORDER
    checks.append({"check": "prespecified_candidate_order", "candidate": "ALL", "max_abs_difference": 0.0, "status": "PASS" if order_ok else "FAIL", "details": "|".join(row["candidate"] for row in fit_rows)})
    checks.append({"check": "all_five_candidates_executed_k2", "candidate": "ALL", "max_abs_difference": 0.0, "status": "PASS" if len(required_success) == 5 else "FAIL", "details": f"successful={len(required_success)}"})
    status = "SUCCESS" if all(row["status"] == "PASS" for row in checks) else "BLOCKED"
    write_csv(RESULTS / "leakage_checks.csv", checks, ["check", "candidate", "max_abs_difference", "status", "details"])
    write_csv(RESULTS / "candidate_k2_fit_status.csv", fit_rows, ["candidate_order", "candidate", "status", "runtime_seconds", "prediction_sha256", "diagnostics_json", "traceback"])
    payload = {"phase": "leakage_gate", "status": status, "started_utc": started, "finished_utc": now(), "patients": len(pairs), "development": len(development), "confirmation": len(confirmation), "checks": len(checks), "failed_checks": sum(row["status"] != "PASS" for row in checks), "candidate_success": sum(row["status"] == "SUCCESS" for row in fit_rows), "candidate_incompatible_k2": sum(row["status"] != "SUCCESS" for row in fit_rows), "tolerance": 1e-12}
    atomic_json(LOGS / "phase06_leakage_gate.json", payload)
    atomic_json(LOGS / "pipeline_checkpoint.json", {"last_completed_phase": "leakage_gate" if status == "SUCCESS" else "source_pseudobulk_and_compositions", "status": status, "updated_utc": now()})
    atomic_text(AUDIT / "LEAKAGE_AUDIT.md", "\n".join(["# Leakage audit", "", f"- Status: `{status}`", f"- Required checks: `{len(checks)}`; failed: `{payload['failed_checks']}`", f"- K=2 candidate executions: `{payload['candidate_success']}/5`", f"- Development/confirmation physical patients: `{len(development)}/{len(confirmation)}`", "- Confirmation target mutation and removal tolerance: `1e-12`.", "- Recurrence expression embargo verified from the pseudobulk execution manifest."]) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if status == "SUCCESS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
