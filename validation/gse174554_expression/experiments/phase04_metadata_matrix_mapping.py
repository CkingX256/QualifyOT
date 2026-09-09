from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import re
import tarfile
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
AUDIT = ROOT / "audit"
LOGS = ROOT / "logs"
LOCAL_ID = AUDIT / "local_identity_mapping_EXCLUDE_FROM_PACKAGE"
TAR_PATH = PROJECT / "GSE174554_RAW.tar"
METADATA_PATH = PROJECT / "GSE174554_Tumor_normal_metadata.txt.gz"


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


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=path.parent) as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        tmp = Path(f.name)
    os.replace(tmp, path)


def read_gate() -> dict[str, Any]:
    gate = json.loads((AUDIT / "TARGET_ACCESS_GATE.json").read_text(encoding="utf-8"))
    if gate.get("status") != "VERIFIED_OPEN_FOR_METADATA_MAPPING":
        raise RuntimeError(f"Target-access gate is not open: {gate.get('status')}")
    freeze_hash = hashlib.sha256((ROOT / "PROTOCOL_FREEZE.md").read_bytes()).hexdigest()
    if freeze_hash != gate.get("protocol_freeze_sha256"):
        raise RuntimeError("PROTOCOL_FREEZE.md hash changed after gate verification")
    return gate


def scan_metadata() -> tuple[dict[str, Counter[str]], dict[str, set[str]], Counter[str], dict[str, Any]]:
    specimen_counts: dict[str, Counter[str]] = defaultdict(Counter)
    specimen_barcodes: dict[str, set[str]] = defaultdict(set)
    all_labels: Counter[str] = Counter()
    row_count = 0
    malformed = 0
    with gzip.open(METADATA_PATH, "rt", encoding="utf-8", errors="strict", newline="") as f:
        header = f.readline().rstrip("\r\n")
        columns = header.split()
        for line_number, line in enumerate(f, start=2):
            text = line.strip()
            if not text:
                continue
            parts = text.split()
            if len(parts) != 2 or "_" not in parts[0]:
                malformed += 1
                continue
            compound, label = parts
            specimen, barcode = compound.split("_", 1)
            if not re.fullmatch(r"SF[^\s_]+", specimen) or not barcode:
                malformed += 1
                continue
            specimen_counts[specimen.upper()][label] += 1
            specimen_barcodes[specimen.upper()].add(barcode)
            all_labels[label] += 1
            row_count += 1
    schema = {
        "path": str(METADATA_PATH),
        "header_verbatim": header,
        "columns": columns,
        "delimiter": "whitespace",
        "compound_key": "Sample#_Barcode split once at first underscore",
        "rows_valid": row_count,
        "rows_malformed": malformed,
        "specimens": len(specimen_counts),
        "unique_labels": sorted(all_labels),
    }
    return specimen_counts, specimen_barcodes, all_labels, schema


def selected_specimens() -> list[dict[str, str]]:
    eligible = {
        row["patient_id"]
        for row in csv.DictReader((AUDIT / "PHYSICAL_PATIENT_ELIGIBILITY.csv").open(encoding="utf-8", newline=""))
        if row["eligible"].lower() == "true"
    }
    rows = []
    with (LOCAL_ID / "SPECIMEN_TO_PATIENT_MAPPING.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["patient_id"] in eligible:
                rows.append(row)
    return rows


def tar_expression_groups(archive: tarfile.TarFile) -> dict[str, dict[str, dict[str, tarfile.TarInfo]]]:
    groups: dict[str, dict[str, dict[str, tarfile.TarInfo]]] = defaultdict(lambda: defaultdict(dict))
    pattern = re.compile(r"^(GSM\d+)_(SF\d+)(v2)?(_batch2)?_(barcodes\.tsv\.gz|features\.tsv\.gz|matrix\.mtx\.gz)$", re.IGNORECASE)
    for member in archive.getmembers():
        match = pattern.match(Path(member.name).name)
        if not match:
            continue
        gsm, specimen, v2, batch2, component = match.groups()
        group_id = f"{gsm}_{specimen}{v2 or ''}{batch2 or ''}"
        kind = component.split(".", 1)[0]
        groups[specimen.upper()][group_id][kind] = member
    return groups


def count_gzip_lines(archive: tarfile.TarFile, member: tarfile.TarInfo) -> tuple[int, str]:
    raw = archive.extractfile(member)
    if raw is None:
        raise RuntimeError(f"Cannot extract {member.name}")
    digest = hashlib.sha256()
    count = 0
    with gzip.GzipFile(fileobj=raw) as stream:
        for line in stream:
            digest.update(line)
            count += 1
    return count, digest.hexdigest()


def read_barcode_set(archive: tarfile.TarFile, member: tarfile.TarInfo) -> set[str]:
    raw = archive.extractfile(member)
    if raw is None:
        raise RuntimeError(f"Cannot extract {member.name}")
    with gzip.GzipFile(fileobj=raw) as stream:
        return {line.decode("utf-8").strip().split("-", 1)[0] for line in stream if line.strip()}


def discover_metadata_aliases(
    selected: list[dict[str, str]],
    metadata_barcodes: dict[str, set[str]],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Map TAR specimen IDs to metadata IDs using exact barcode-set identity.

    An exact ID is preferred.  For an exact specimen ID, three mechanically
    auditable barcode relations are accepted: exact identity; complete
    metadata coverage when the raw matrix contains additional unlabeled
    cells; and a bijective ``.1`` suffix normalization used by five legacy
    metadata specimens.  A non-identical specimen name is accepted only when
    its complete canonical non-batch2 barcode set equals exactly one metadata
    specimen set.  Zero matches remain MISSING and multiple matches are
    AMBIGUOUS; neither is guessed by string similarity.
    """

    selected_ids = sorted({row["specimen_id"].upper() for row in selected})
    aliases: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    with tarfile.open(TAR_PATH, mode="r:") as archive:
        groups = tar_expression_groups(archive)
        for specimen in selected_ids:
            specimen_groups = groups.get(specimen, {})
            complete_all = sorted(
                (group_id, members)
                for group_id, members in specimen_groups.items()
                if {"barcodes", "features", "matrix"}.issubset(members)
                and "_batch2" not in group_id.lower()
            )
            exact_named = [item for item in complete_all if re.search(rf"_{re.escape(specimen)}$", item[0], re.IGNORECASE)]
            complete = exact_named if len(exact_named) == 1 else complete_all
            if len(complete) != 1:
                rows.append({"tar_specimen_id": specimen, "metadata_specimen_id": "", "evidence": "MISSING_OR_MULTIPLE_CANONICAL_TAR_GROUPS", "barcode_count": 0, "metadata_barcode_count": 0, "barcode_overlap_count": 0, "barcode_normalization": "NONE", "exact_match_count": 0, "status": "AMBIGUOUS"})
                continue
            bars = read_barcode_set(archive, complete[0][1]["barcodes"])
            normalization = "IDENTITY"
            overlap_count = 0
            metadata_count = 0
            if specimen in metadata_barcodes:
                exact = metadata_barcodes[specimen]
                metadata_count = len(exact)
                overlap_count = len(bars & exact)
                dot1_bars = {f"{barcode}.1" for barcode in bars}
                if exact == bars:
                    candidates = [specimen]
                    evidence = "EXACT_SPECIMEN_ID_AND_EXACT_BARCODE_SET"
                elif bars.issubset(exact):
                    candidates = [specimen]
                    evidence = "EXACT_SPECIMEN_ID_AND_RAW_BARCODE_SUBSET"
                elif exact.issubset(bars):
                    candidates = [specimen]
                    evidence = "EXACT_SPECIMEN_ID_AND_METADATA_BARCODE_SUBSET"
                elif dot1_bars == exact:
                    candidates = [specimen]
                    evidence = "EXACT_SPECIMEN_ID_AND_BIJECTIVE_DOT1_NORMALIZATION"
                    normalization = "APPEND_DOT1"
                    overlap_count = len(exact)
                else:
                    candidates = []
                    evidence = "EXACT_ID_BARCODE_SET_MISMATCH"
            else:
                candidates = sorted(key for key, values in metadata_barcodes.items() if values == bars)
                evidence = "UNIQUE_EXACT_BARCODE_SET_IDENTITY" if len(candidates) == 1 else "NO_UNIQUE_EXACT_BARCODE_SET_IDENTITY"
                if len(candidates) == 1:
                    metadata_count = len(metadata_barcodes[candidates[0]])
                    overlap_count = len(bars)
            status = "PASS" if len(candidates) == 1 else ("MISSING" if not candidates else "AMBIGUOUS")
            metadata_id = candidates[0] if len(candidates) == 1 else ""
            if metadata_id:
                aliases[specimen] = metadata_id
            rows.append(
                {
                    "tar_specimen_id": specimen,
                    "metadata_specimen_id": metadata_id,
                    "evidence": evidence,
                    "barcode_count": len(bars),
                    "metadata_barcode_count": metadata_count,
                    "barcode_overlap_count": overlap_count,
                    "barcode_normalization": normalization,
                    "exact_match_count": len(candidates),
                    "status": status,
                }
            )
    return aliases, rows


def scan_matrix(archive: tarfile.TarFile, member: tarfile.TarInfo) -> dict[str, Any]:
    raw = archive.extractfile(member)
    if raw is None:
        raise RuntimeError(f"Cannot extract {member.name}")
    digest = hashlib.sha256()
    dimensions: tuple[int, int, int] | None = None
    entries = 0
    negative = 0
    noninteger = 0
    nul_padding_bytes = 0
    nul_affected_lines = 0
    parse_error_line = 0
    parse_error_excerpt = ""
    with gzip.GzipFile(fileobj=raw) as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            digest.update(raw_line)
            line_nuls = raw_line.count(b"\x00")
            if line_nuls:
                nul_padding_bytes += line_nuls
                nul_affected_lines += 1
            # Some deposited matrices contain large, auditable NUL padding
            # runs after otherwise valid coordinate records.  Removal is
            # accepted only because all remaining rows are still parsed under
            # the strict three-field grammar and total nnz is checked below.
            line = raw_line.replace(b"\x00", b"").strip()
            if not line or line.startswith(b"%"):
                continue
            parts = line.split()
            if dimensions is None:
                if len(parts) != 3:
                    parse_error_line = line_number
                    parse_error_excerpt = repr(line[:160])
                    break
                dimensions = tuple(int(value) for value in parts)
                continue
            if len(parts) != 3:
                parse_error_line = line_number
                parse_error_excerpt = repr(line[:160])
                break
            value = float(parts[2])
            entries += 1
            negative += int(value < 0)
            noninteger += int(abs(value - round(value)) > 1e-8)
    if dimensions is None:
        dimensions = (0, 0, 0)
        if not parse_error_line:
            parse_error_line = 1
            parse_error_excerpt = "MISSING_DIMENSIONS"
    return {
        "matrix_rows": dimensions[0],
        "matrix_cols": dimensions[1],
        "declared_nnz": dimensions[2],
        "observed_nnz_lines": entries,
        "negative_entries": negative,
        "noninteger_entries": noninteger,
        "nul_padding_bytes": nul_padding_bytes,
        "nul_affected_lines": nul_affected_lines,
        "parse_error_line": parse_error_line,
        "parse_error_excerpt": parse_error_excerpt,
        "decompressed_sha256": digest.hexdigest(),
    }


def main() -> int:
    started = utc_now()
    gate = read_gate()
    specimen_label_counts, metadata_barcodes, label_counts, schema = scan_metadata()
    atomic_json(AUDIT / "TARGET_METADATA_SCHEMA.json", schema)
    label_rows = [{"author_label": key, "cell_count": label_counts[key], "frozen_mapping": "Tumor" if key == "Tumor" else ("NonTumor" if key == "Normal" else "UNMAPPED")} for key in sorted(label_counts)]
    write_csv(AUDIT / "AUTHOR_LABEL_INVENTORY.csv", label_rows, ["author_label", "cell_count", "frozen_mapping"])
    selected = selected_specimens()
    selected_ids = {row["specimen_id"].upper() for row in selected}
    metadata_aliases, alias_rows = discover_metadata_aliases(selected, metadata_barcodes)
    write_csv(
        LOCAL_ID / "FROZEN_SPECIMEN_METADATA_ALIAS_MAP.csv",
        alias_rows,
        ["tar_specimen_id", "metadata_specimen_id", "evidence", "barcode_count", "metadata_barcode_count", "barcode_overlap_count", "barcode_normalization", "exact_match_count", "status"],
    )
    label_count_rows = []
    for specimen in sorted(selected_ids):
        metadata_specimen = metadata_aliases.get(specimen, "")
        counts = specimen_label_counts.get(metadata_specimen, Counter())
        label_count_rows.append(
            {
                "specimen_id": specimen,
                "metadata_specimen_id": metadata_specimen,
                "mapping_evidence": next((row["evidence"] for row in alias_rows if row["tar_specimen_id"] == specimen), "MISSING"),
                "tumor_cells": counts.get("Tumor", 0),
                "normal_cells": counts.get("Normal", 0),
                "unknown_label_cells": sum(value for key, value in counts.items() if key not in {"Tumor", "Normal"}),
                "total_metadata_cells": sum(counts.values()),
                "target_status": "DEFINED" if counts.get("Tumor", 0) + counts.get("Normal", 0) > 0 else "TARGET_UNDEFINED",
            }
        )
    write_csv(AUDIT / "SELECTED_SPECIMEN_AUTHOR_LABEL_COUNTS.csv", label_count_rows, ["specimen_id", "metadata_specimen_id", "mapping_evidence", "tumor_cells", "normal_cells", "unknown_label_cells", "total_metadata_cells", "target_status"])

    validation_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    with tarfile.open(TAR_PATH, mode="r:") as archive:
        groups = tar_expression_groups(archive)
        for selected_row in selected:
            specimen = selected_row["specimen_id"].upper()
            specimen_groups = groups.get(specimen, {})
            if not specimen_groups:
                mapping_rows.append({**selected_row, "specimen_id": specimen, "raw_group_count": 0, "mapping_status": "MISSING_EXPRESSION_GROUP", "raw_groups": ""})
                continue
            complete_group_ids = sorted(group_id for group_id, members in specimen_groups.items() if {"barcodes", "features", "matrix"}.issubset(members))
            non_batch = [group_id for group_id in complete_group_ids if "_batch2" not in group_id.lower()]
            exact_named = [group_id for group_id in non_batch if re.search(rf"_{re.escape(specimen)}$", group_id, re.IGNORECASE)]
            canonical_group_id = exact_named[0] if len(exact_named) == 1 else (non_batch[0] if len(non_batch) == 1 else "")
            mapping_rows.append(
                {
                    **selected_row,
                    "specimen_id": specimen,
                    "raw_group_count": len(complete_group_ids),
                    "mapping_status": "EXACT_SPECIMEN_ID" if complete_group_ids else "INCOMPLETE_EXPRESSION_GROUP",
                    "raw_groups": ";".join(complete_group_ids),
                    "canonical_group_id": canonical_group_id,
                }
            )
            main_barcodes: set[str] | None = None
            for group_id in complete_group_ids:
                members = specimen_groups[group_id]
                barcode_count, barcode_hash = count_gzip_lines(archive, members["barcodes"])
                feature_count, feature_hash = count_gzip_lines(archive, members["features"])
                matrix = scan_matrix(archive, members["matrix"])
                orientation = "genes_by_cells" if matrix["matrix_rows"] == feature_count and matrix["matrix_cols"] == barcode_count else ("cells_by_genes" if matrix["matrix_rows"] == barcode_count and matrix["matrix_cols"] == feature_count else "DIMENSION_MISMATCH")
                bars = read_barcode_set(archive, members["barcodes"])
                if "_batch2" not in group_id.lower() and main_barcodes is None:
                    main_barcodes = bars
                batch_overlap = "NOT_APPLICABLE"
                if "_batch2" in group_id.lower() and main_barcodes is not None:
                    batch_overlap = f"{len(bars & main_barcodes)}/{len(bars)}"
                valid = not matrix["parse_error_line"] and orientation == "genes_by_cells" and matrix["declared_nnz"] == matrix["observed_nnz_lines"] and matrix["negative_entries"] == 0 and matrix["noninteger_entries"] == 0
                status = ("PASS_WITH_NUL_PADDING" if matrix["nul_padding_bytes"] else "PASS") if valid else "FAIL"
                validation_rows.append(
                    {
                        "specimen_id": specimen,
                        "group_id": group_id,
                        "barcode_member": members["barcodes"].name,
                        "feature_member": members["features"].name,
                        "matrix_member": members["matrix"].name,
                        "barcode_count": barcode_count,
                        "feature_count": feature_count,
                        **matrix,
                        "orientation": orientation,
                        "batch2_barcode_overlap_with_main": batch_overlap,
                        "barcode_decompressed_sha256": barcode_hash,
                        "feature_decompressed_sha256": feature_hash,
                        "status": status,
                    }
                )

    map_fields = ["patient_id", "numeric_pair_number", "role", "specimen_id", "stage", "elapsed_days", "workbook_row", "raw_group_count", "mapping_status", "raw_groups", "canonical_group_id"]
    write_csv(LOCAL_ID / "FROZEN_SPECIMEN_EXPRESSION_FILE_MAP.csv", mapping_rows, map_fields)
    validation_fields = ["specimen_id", "group_id", "barcode_member", "feature_member", "matrix_member", "barcode_count", "feature_count", "matrix_rows", "matrix_cols", "declared_nnz", "observed_nnz_lines", "negative_entries", "noninteger_entries", "nul_padding_bytes", "nul_affected_lines", "parse_error_line", "parse_error_excerpt", "decompressed_sha256", "orientation", "batch2_barcode_overlap_with_main", "barcode_decompressed_sha256", "feature_decompressed_sha256", "status"]
    write_csv(AUDIT / "SELECTED_EXPRESSION_MATRIX_VALIDATION.csv", validation_rows, validation_fields)

    allowed_labels = set(label_counts) <= {"Tumor", "Normal"}
    all_metadata_present = all(row["specimen_id"] in metadata_aliases for row in mapping_rows)
    all_mapped = all(row["mapping_status"] == "EXACT_SPECIMEN_ID" and row.get("canonical_group_id") for row in mapping_rows)
    all_matrices_pass = bool(validation_rows) and all(str(row["status"]).startswith("PASS") for row in validation_rows)
    all_targets_defined = all(row["target_status"] == "DEFINED" for row in label_count_rows)
    invalid_specimens = {row["specimen_id"] for row in validation_rows if row["status"] == "FAIL"}
    split_map = {
        row["patient_id"]: row["analysis_partition"]
        for row in csv.DictReader((AUDIT / "FROZEN_PATIENT_SPLIT.csv").open(encoding="utf-8", newline=""))
    }
    reasons_by_patient: dict[str, list[str]] = defaultdict(list)
    specimens_by_patient: dict[str, list[str]] = defaultdict(list)
    roles_by_patient: dict[str, list[str]] = defaultdict(list)
    for row in selected:
        specimen = row["specimen_id"].upper()
        if specimen in invalid_specimens and row["role"] == "source_primary":
            reasons_by_patient[row["patient_id"]].append("SOURCE_MATRIX_INTEGRITY_FAILURE")
        if specimen not in metadata_aliases:
            reasons_by_patient[row["patient_id"]].append(
                "SOURCE_METADATA_MAPPING_MISSING" if row["role"] == "source_primary" else "TARGET_METADATA_MAPPING_MISSING"
            )
        specimens_by_patient[row["patient_id"]].append(specimen)
        roles_by_patient[row["patient_id"]].append(row["role"])
    post_freeze_exclusions = [
        {
            "patient_id": patient_id,
            "frozen_partition_preserved": split_map[patient_id],
            "specimen_id": ";".join(specimens_by_patient[patient_id]),
            "role": ";".join(roles_by_patient[patient_id]),
            "reason": ";".join(sorted(set(reasons))),
            "replacement": "NONE",
            "rehash": "FORBIDDEN_NOT_PERFORMED",
        }
        for patient_id, reasons in sorted(reasons_by_patient.items())
        if reasons
    ]
    write_csv(AUDIT / "POST_FREEZE_DATA_INTEGRITY_EXCLUSIONS.csv", post_freeze_exclusions, ["patient_id", "frozen_partition_preserved", "specimen_id", "role", "reason", "replacement", "rehash"])
    ambiguous_aliases = [row for row in alias_rows if row["status"] == "AMBIGUOUS"]
    base_mapping_pass = allowed_labels and all_mapped and not ambiguous_aliases
    status = "SUCCESS" if base_mapping_pass and all_matrices_pass and all_metadata_present and all_targets_defined else ("PARTIAL" if base_mapping_pass and post_freeze_exclusions else "BLOCKED")
    payload = {
        "schema_version": 1,
        "phase": "metadata_and_matrix_mapping",
        "status": status,
        "started_utc": started,
        "finished_utc": utc_now(),
        "pre_target_gate_hash": gate["protocol_freeze_sha256"],
        "metadata_rows": schema["rows_valid"],
        "metadata_specimens": schema["specimens"],
        "author_labels": dict(sorted(label_counts.items())),
        "frozen_label_mapping": {"Tumor": "Tumor", "Normal": "NonTumor"},
        "selected_physical_patients": len({row["patient_id"] for row in selected}),
        "selected_specimens": len(selected_ids),
        "expression_raw_groups_validated": len(validation_rows),
        "allowed_labels_only": allowed_labels,
        "all_selected_metadata_present": all_metadata_present,
        "all_selected_expression_mapped": all_mapped,
        "all_matrices_pass": all_matrices_pass,
        "matrices_with_nul_padding": sum(row["status"] == "PASS_WITH_NUL_PADDING" for row in validation_rows),
        "total_nul_padding_bytes": sum(int(row["nul_padding_bytes"]) for row in validation_rows),
        "all_targets_defined": all_targets_defined,
        "metadata_alias_exact_id": sum(row["evidence"] == "EXACT_SPECIMEN_ID_AND_EXACT_BARCODE_SET" for row in alias_rows),
        "metadata_alias_exact_id_subset": sum(row["evidence"] in {"EXACT_SPECIMEN_ID_AND_RAW_BARCODE_SUBSET", "EXACT_SPECIMEN_ID_AND_METADATA_BARCODE_SUBSET"} for row in alias_rows),
        "metadata_alias_dot1_normalized": sum(row["evidence"] == "EXACT_SPECIMEN_ID_AND_BIJECTIVE_DOT1_NORMALIZATION" for row in alias_rows),
        "metadata_alias_unique_barcode_identity": sum(row["evidence"] == "UNIQUE_EXACT_BARCODE_SET_IDENTITY" for row in alias_rows),
        "metadata_alias_missing": sum(row["status"] == "MISSING" for row in alias_rows),
        "metadata_alias_ambiguous": len(ambiguous_aliases),
        "invalid_matrix_specimens": sorted(invalid_specimens),
        "post_freeze_exclusions": len(post_freeze_exclusions),
        "effective_analysis_patients": gate["eligible_n"] - len({row["patient_id"] for row in post_freeze_exclusions}),
        "target_metadata_opened_after_freeze": True,
    }
    atomic_json(LOGS / "phase04_metadata_matrix_mapping.json", payload)
    atomic_json(LOGS / "pipeline_checkpoint.json", {"last_completed_phase": "metadata_and_matrix_mapping" if status in {"SUCCESS", "PARTIAL"} else "pairing_and_pre_target_freeze", "status": status, "updated_utc": utc_now()})
    report = [
        "# Stage 4: target metadata and expression matrix mapping",
        "",
        f"- Status: `{status}`",
        f"- Valid metadata rows: `{schema['rows_valid']}`; malformed: `{schema['rows_malformed']}`",
        f"- Unique author labels (printed before mapping): `{', '.join(sorted(label_counts))}`",
        "- Frozen label mapping: `Tumor -> Tumor`; `Normal -> NonTumor`.",
        f"- Selected patients/specimens: `{payload['selected_physical_patients']}` / `{payload['selected_specimens']}`",
        f"- Raw expression groups fully validated: `{len(validation_rows)}`",
        f"- All dimensions/orientations/count properties passed: `{str(all_matrices_pass).lower()}`",
        f"- Target metadata was opened only after freeze hash `{gate['protocol_freeze_sha256']}`.",
    ]
    atomic_text(AUDIT / "STAGE04_METADATA_MATRIX_MAPPING_REPORT.md", "\n".join(report) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if status == "SUCCESS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
