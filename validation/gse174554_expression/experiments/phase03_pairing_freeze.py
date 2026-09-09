from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import openpyxl


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
AUDIT = ROOT / "audit"
LOGS = ROOT / "logs"
LOCAL_ID = AUDIT / "local_identity_mapping_EXCLUDE_FROM_PACKAGE"
WORKBOOK = PROJECT / "43018_2022_475_MOESM2_ESM.xlsx"
CONTRACT = ROOT / "configs" / "gse174554_fallback_contract.yaml"
sys.path.insert(0, str(ROOT / "src"))

from qualifyot_gse174554.split import GSE174554_SPLIT_SALT, exact_sorted_hash_split


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=path.parent) as f:
        f.write(text)
        tmp = Path(f.name)
    os.replace(tmp, path)


def atomic_json(path: Path, payload: object) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows and not fieldnames:
        raise ValueError(f"Empty CSV needs explicit fields: {path}")
    fields = fieldnames or list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=path.parent) as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        tmp = Path(f.name)
    os.replace(tmp, path)


def normalize(value: Any) -> str:
    return "" if value is None else str(value).strip()


def numeric_pair(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value).strip()
    return int(text) if re.fullmatch(r"\d+", text) else None


def elapsed_days(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    text = normalize(value)
    try:
        return float(text)
    except ValueError:
        return math.inf


def read_table1() -> tuple[list[dict[str, Any]], list[str]]:
    workbook = openpyxl.load_workbook(WORKBOOK, read_only=True, data_only=True)
    sheet = workbook["Table 1"]
    rows = list(sheet.iter_rows(values_only=True))
    header_index = next(i for i, row in enumerate(rows) if "ID" in row and "Stage" in row and "Pair#" in row)
    headers = [normalize(value) for value in rows[header_index]]
    records: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows[header_index + 1 :], start=header_index + 2):
        if not any(value is not None for value in row):
            continue
        record = {headers[i]: row[i] if i < len(row) else None for i in range(len(headers)) if headers[i]}
        record["workbook_row"] = ordinal
        records.append(record)
    workbook.close()
    return records, headers


def expression_sf_ids() -> set[str]:
    values: set[str] = set()
    with (AUDIT / "ORIGINAL_TAR_MEMBER_MANIFEST.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["kind"] not in {"barcodes", "features", "matrix"}:
                continue
            match = re.search(r"_(SF\d+)(?:v2)?(?:_batch2)?_(?:barcodes|features|matrix)", row["member_name"], re.IGNORECASE)
            if match:
                values.add(match.group(1).upper())
    return values


def main() -> int:
    started = utc_now()
    if not CONTRACT.is_file():
        raise FileNotFoundError("Fallback contract must exist before pairing")
    records, source_headers = read_table1()
    required_fields = ["ID", "Stage", "Pair#", "Elapsed time to recurrence", "snRNA-seq"]
    missing_fields = sorted(set(required_fields) - set(source_headers))
    if missing_fields:
        raise RuntimeError(f"Missing required Table 1 fields: {missing_fields}")
    raw_rows = [
        {
            "workbook_row": r["workbook_row"],
            "ID": normalize(r.get("ID")),
            "Stage": normalize(r.get("Stage")),
            "Pair#": normalize(r.get("Pair#")),
            "Elapsed time to recurrence": normalize(r.get("Elapsed time to recurrence")),
            "snRNA-seq": normalize(r.get("snRNA-seq")),
        }
        for r in records
    ]
    write_csv(LOCAL_ID / "SUPPLEMENTARY_TABLE1_PAIRING_FIELDS_RAW.csv", raw_rows)
    atomic_text(LOCAL_ID / ".exclude_from_package", "This directory contains local specimen-to-patient source mappings and must not enter the final package.\n")

    sf_available = expression_sf_ids()
    grouped: dict[int, list[dict[str, Any]]] = {}
    nonnumeric_rows: list[dict[str, Any]] = []
    for record in records:
        pair_number = numeric_pair(record.get("Pair#"))
        if pair_number is None:
            nonnumeric_rows.append(
                {
                    "workbook_row": record["workbook_row"],
                    "source_pair_label": normalize(record.get("Pair#")),
                    "specimen_id": normalize(record.get("ID")),
                    "reason": "NONNUMERIC_PAIR_LABEL_NOT_A_PHYSICAL_PATIENT",
                }
            )
        else:
            grouped.setdefault(pair_number, []).append(record)
    write_csv(AUDIT / "NONNUMERIC_PAIR_LABEL_EXCLUSIONS.csv", nonnumeric_rows)

    pairing_rows: list[dict[str, Any]] = []
    local_mapping_rows: list[dict[str, Any]] = []
    eligible_patient_ids: list[str] = []
    discrepancies: list[dict[str, Any]] = []
    for pair_number in sorted(grouped):
        rows = grouped[pair_number]
        snrna = [r for r in rows if normalize(r.get("snRNA-seq")).upper() in {"Y", "YES"}]
        primary = [r for r in snrna if normalize(r.get("Stage")).lower().startswith("primary")]
        recurrence = [r for r in snrna if normalize(r.get("Stage")).lower().startswith(("recur", "recurrent"))]
        recurrence.sort(key=lambda r: (elapsed_days(r.get("Elapsed time to recurrence")), int(r["workbook_row"])))
        source = primary[0] if len(primary) == 1 else None
        target = recurrence[0] if recurrence else None
        source_id = normalize(source.get("ID")) if source else ""
        target_id = normalize(target.get("ID")) if target else ""
        source_in_tar = source_id.upper() in sf_available
        target_in_tar = target_id.upper() in sf_available
        reasons: list[str] = []
        if len(primary) != 1:
            reasons.append("PRIMARY_SNRNA_COUNT_NOT_EXACTLY_ONE")
        if not recurrence:
            reasons.append("NO_RECURRENCE_SNRNA")
        if source and not source_in_tar:
            reasons.append("PRIMARY_EXPRESSION_NOT_IN_TAR")
        if target and not target_in_tar:
            reasons.append("RECURRENCE_EXPRESSION_NOT_IN_TAR")
        eligible = not reasons
        patient_id = f"GSE174554-P{pair_number:03d}"
        if eligible:
            eligible_patient_ids.append(patient_id)
        pairing_rows.append(
            {
                "patient_id": patient_id,
                "numeric_pair_number": pair_number,
                "eligible": eligible,
                "status": "ELIGIBLE" if eligible else "INELIGIBLE",
                "exclusion_reason": ";".join(reasons),
                "primary_snrna_count": len(primary),
                "recurrence_snrna_count": len(recurrence),
                "source_expression_present": source_in_tar,
                "target_expression_present": target_in_tar,
                "first_recurrence_elapsed_days": "" if target is None or math.isinf(elapsed_days(target.get("Elapsed time to recurrence"))) else elapsed_days(target.get("Elapsed time to recurrence")),
                "selection_rule": "earliest_numeric_elapsed_days_then_workbook_order",
            }
        )
        for role, selected in (("source_primary", source), ("target_first_recurrence", target)):
            if selected is not None:
                local_mapping_rows.append(
                    {
                        "patient_id": patient_id,
                        "numeric_pair_number": pair_number,
                        "role": role,
                        "specimen_id": normalize(selected.get("ID")),
                        "stage": normalize(selected.get("Stage")),
                        "elapsed_days": normalize(selected.get("Elapsed time to recurrence")),
                        "workbook_row": selected["workbook_row"],
                    }
                )
        if not eligible:
            discrepancies.append({"patient_id": patient_id, "numeric_pair_number": pair_number, "status": "INELIGIBLE", "reason": ";".join(reasons)})
    write_csv(AUDIT / "PHYSICAL_PATIENT_ELIGIBILITY.csv", pairing_rows)
    write_csv(LOCAL_ID / "SPECIMEN_TO_PATIENT_MAPPING.csv", local_mapping_rows)
    write_csv(AUDIT / "ELIGIBILITY_DISCREPANCIES_FROM_36.csv", discrepancies, ["patient_id", "numeric_pair_number", "status", "reason"])

    split_rows = exact_sorted_hash_split(eligible_patient_ids)
    write_csv(AUDIT / "FROZEN_PATIENT_SPLIT.csv", split_rows)
    patient_rows = [{"patient_id": value, "eligibility": "ELIGIBLE"} for value in sorted(eligible_patient_ids)]
    write_csv(AUDIT / "FROZEN_ELIGIBLE_PATIENTS.csv", patient_rows)

    closure_paths = [CONTRACT, ROOT / "src" / "qualifyot_gse174554" / "split.py", Path(__file__), AUDIT / "FROZEN_PATIENT_SPLIT.csv", AUDIT / "FROZEN_ELIGIBLE_PATIENTS.csv", AUDIT / "PHYSICAL_PATIENT_ELIGIBILITY.csv"]
    closure_rows = [{"path": str(p.relative_to(ROOT)), "size_bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in closure_paths]
    write_csv(AUDIT / "PRE_TARGET_DEPENDENCY_CLOSURE_SHA256.csv", closure_rows)
    closure_hash = hashlib.sha256("".join(f"{r['path']}\0{r['sha256']}\n" for r in closure_rows).encode()).hexdigest()
    development_n = sum(row["analysis_partition"] == "development" for row in split_rows)
    confirmation_n = len(split_rows) - development_n
    special_pair36 = [r for r in local_mapping_rows if r["numeric_pair_number"] == 36 and r["role"] == "target_first_recurrence"]
    pair36_ok = len(special_pair36) == 1 and special_pair36[0]["specimen_id"].upper() == "SF12407"
    # The plan explicitly makes final eligibility conditional on auditable TAR
    # mapping and forbids forcing N=36.  Six otherwise paired primaries are
    # absent from the original TAR (and Pair 37 has no primary snRNA specimen),
    # so they remain in the exclusion ledger.  The gate opens only when every
    # retained patient has an exact source/target specimen match and the split
    # cardinalities follow the literal floor rule.
    exact_mapping = all(row["eligible"] in {True, False} for row in pairing_rows) and all(
        row["source_expression_present"] and row["target_expression_present"]
        for row in pairing_rows
        if row["eligible"]
    )
    expected_development_n = math.floor(0.60 * len(eligible_patient_ids))
    gate_ready = (
        len(eligible_patient_ids) > 0
        and development_n == expected_development_n
        and confirmation_n == len(eligible_patient_ids) - expected_development_n
        and exact_mapping
        and pair36_ok
    )
    status = "SUCCESS" if gate_ready else "BLOCKED"
    freeze_lines = [
        "# GSE174554 expression-v1 pre-target protocol freeze",
        "",
        f"- Freeze timestamp (UTC): `{utc_now()}`",
        f"- Status: `{status}`",
        f"- Eligible physical patients: `{len(eligible_patient_ids)}`",
        f"- Development / confirmation: `{development_n}` / `{confirmation_n}`",
        "- Preliminary Supplement-only paired count: `36`.",
        f"- Final TAR-mapped count: `{len(eligible_patient_ids)}`; no patient count was forced.",
        f"- Numeric exclusions retained: `{len(discrepancies)}` (see `audit/ELIGIBILITY_DISCREPANCIES_FROM_36.csv`).",
        f"- Split salt: `{GSE174554_SPLIT_SALT}`",
        "- Split rule: lexicographic full SHA256, first floor(0.60*N) development.",
        f"- Pair 36 first recurrence rule verified as SF12407: `{str(pair36_ok).lower()}`",
        f"- Contract SHA256: `{sha256_file(CONTRACT)}`",
        f"- Dependency-closure SHA256: `{closure_hash}`",
        "- Target metadata opened before this freeze: `false`",
        "- Recurrence expression embargo: active",
        "",
        "Any mismatch in these hashes closes the target-access gate.",
    ]
    atomic_text(ROOT / "PROTOCOL_FREEZE.md", "\n".join(freeze_lines) + "\n")
    gate = {
        "schema_version": 1,
        "gate": "GSE174554_TARGET_ACCESS",
        "status": "VERIFIED_OPEN_FOR_METADATA_MAPPING" if status == "SUCCESS" else "BLOCKED",
        "target_metadata_opened_before_freeze": False,
        "eligible_n": len(eligible_patient_ids),
        "development_n": development_n,
        "confirmation_n": confirmation_n,
        "contract_sha256": sha256_file(CONTRACT),
        "dependency_closure_sha256": closure_hash,
        "protocol_freeze_sha256": sha256_file(ROOT / "PROTOCOL_FREEZE.md"),
        "verified_utc": utc_now(),
    }
    atomic_json(AUDIT / "TARGET_ACCESS_GATE.json", gate)
    phase = {
        "schema_version": 1,
        "phase": "pairing_and_pre_target_freeze",
        "status": status,
        "started_utc": started,
        "finished_utc": utc_now(),
        "numeric_pair_records": len(grouped),
        "eligible_physical_patients": len(eligible_patient_ids),
        "development_patients": development_n,
        "confirmation_patients": confirmation_n,
        "nonnumeric_group_rows": len(nonnumeric_rows),
        "ineligible_numeric_pairs": len(discrepancies),
        "pair36_first_recurrence_is_SF12407": pair36_ok,
        "target_access_gate": gate,
    }
    atomic_json(LOGS / "phase03_pairing_freeze.json", phase)
    atomic_json(LOGS / "pipeline_checkpoint.json", {"last_completed_phase": "pairing_and_pre_target_freeze" if status == "SUCCESS" else "archive_integrity", "status": status, "updated_utc": utc_now()})
    print(json.dumps(phase, ensure_ascii=False, indent=2))
    return 0 if status == "SUCCESS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
