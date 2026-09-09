from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import re
import sys
import tarfile
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
AUDIT = ROOT / "audit"
LOGS = ROOT / "logs"
DATA = ROOT / "data" / "processed"
LOCAL_ID = AUDIT / "local_identity_mapping_EXCLUDE_FROM_PACKAGE"
TAR_PATH = PROJECT / "GSE174554_RAW.tar"
METADATA_PATH = PROJECT / "GSE174554_Tumor_normal_metadata.txt.gz"
sys.path.insert(0, str(ROOT / "src"))

from qualifyot_gse174554.expression import SourceExpressionStore


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_effective_mapping() -> tuple[list[dict[str, str]], dict[str, str]]:
    phase = json.loads((LOGS / "phase04_metadata_matrix_mapping.json").read_text(encoding="utf-8"))
    if phase.get("status") not in {"SUCCESS", "PARTIAL"}:
        raise RuntimeError("Stage 4 has not closed as SUCCESS/PARTIAL")
    excluded = set()
    exclusion_file = AUDIT / "POST_FREEZE_DATA_INTEGRITY_EXCLUSIONS.csv"
    if exclusion_file.exists():
        with exclusion_file.open(encoding="utf-8", newline="") as f:
            excluded = {row["patient_id"] for row in csv.DictReader(f)}
    split = {}
    with (AUDIT / "FROZEN_PATIENT_SPLIT.csv").open(encoding="utf-8", newline="") as f:
        split = {row["patient_id"]: row["analysis_partition"] for row in csv.DictReader(f)}
    with (LOCAL_ID / "FROZEN_SPECIMEN_METADATA_ALIAS_MAP.csv").open(encoding="utf-8", newline="") as f:
        aliases = {
            row["tar_specimen_id"]: row
            for row in csv.DictReader(f)
            if row["status"] == "PASS"
        }
    rows = []
    with (LOCAL_ID / "SPECIMEN_TO_PATIENT_MAPPING.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["patient_id"] in split and row["patient_id"] not in excluded:
                specimen = row["specimen_id"].upper()
                if specimen not in aliases:
                    raise RuntimeError(f"Effective specimen has no frozen metadata alias: {specimen}")
                row["metadata_specimen_id"] = aliases[specimen]["metadata_specimen_id"]
                row["barcode_normalization"] = aliases[specimen].get("barcode_normalization", "IDENTITY")
                rows.append(row)
    return rows, split


def load_file_map() -> dict[tuple[str, str], str]:
    mapping = {}
    with (LOCAL_ID / "FROZEN_SPECIMEN_EXPRESSION_FILE_MAP.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            canonical = row.get("canonical_group_id", "")
            if not canonical:
                raise RuntimeError(f"No frozen canonical expression group for {row['specimen_id']}")
            mapping[(row["patient_id"], row["role"])] = canonical
    return mapping


def load_metadata_labels(specimens: set[str]) -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = defaultdict(dict)
    with gzip.open(METADATA_PATH, "rt", encoding="utf-8", errors="strict") as f:
        next(f)
        for line in f:
            compound, label = line.split()
            specimen, barcode = compound.split("_", 1)
            specimen = specimen.upper()
            if specimen not in specimens:
                continue
            mapped = "Tumor" if label == "Tumor" else ("NonTumor" if label == "Normal" else "UNMAPPED")
            previous = labels[specimen].get(barcode)
            if previous is not None and previous != mapped:
                raise RuntimeError(f"Conflicting labels for {specimen}_{barcode}")
            labels[specimen][barcode] = mapped
    return labels


def tar_groups(archive: tarfile.TarFile) -> dict[str, dict[str, tarfile.TarInfo]]:
    groups: dict[str, dict[str, tarfile.TarInfo]] = defaultdict(dict)
    pattern = re.compile(r"^(GSM\d+_SF\d+(?:v2)?(?:_batch2)?)_(barcodes|features|matrix)\.(?:tsv|mtx)\.gz$", re.IGNORECASE)
    for member in archive.getmembers():
        match = pattern.match(Path(member.name).name)
        if match:
            groups[match.group(1)][match.group(2).lower()] = member
    return groups


def read_gzip_lines(archive: tarfile.TarFile, member: tarfile.TarInfo) -> list[str]:
    raw = archive.extractfile(member)
    if raw is None:
        raise RuntimeError(f"Cannot read {member.name}")
    with gzip.GzipFile(fileobj=raw) as stream:
        return [line.decode("utf-8").rstrip("\r\n") for line in stream]


def sum_matrix_by_gene(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    selected_columns: set[int],
    gene_count: int,
) -> tuple[np.ndarray, int, int]:
    sums = np.zeros(gene_count, dtype=np.float64)
    raw = archive.extractfile(member)
    if raw is None:
        raise RuntimeError(f"Cannot read {member.name}")
    dimensions_seen = False
    observed = 0
    declared = -1
    nul_bytes = 0
    with gzip.GzipFile(fileobj=raw) as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            nul_bytes += raw_line.count(b"\x00")
            line = raw_line.replace(b"\x00", b"").strip()
            if not line or line.startswith(b"%"):
                continue
            parts = line.split()
            if not dimensions_seen:
                if len(parts) != 3:
                    raise RuntimeError(f"Invalid dimensions in {member.name}")
                rows, _cols, declared = (int(value) for value in parts)
                if rows != gene_count:
                    raise RuntimeError(f"Gene dimension mismatch in {member.name}")
                dimensions_seen = True
                continue
            if len(parts) != 3:
                raise RuntimeError(f"Unrecoverable MatrixMarket row {line_number} in {member.name}")
            gene_index, column_index = int(parts[0]), int(parts[1])
            value = float(parts[2])
            observed += 1
            if column_index in selected_columns:
                sums[gene_index - 1] += value
    if observed != declared:
        raise RuntimeError(f"Observed nnz {observed} != declared {declared} in {member.name}")
    return sums, observed, nul_bytes


def main() -> int:
    started = utc_now()
    mapping, split = load_effective_mapping()
    file_map = load_file_map()
    by_patient: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in mapping:
        by_patient[row["patient_id"]][row["role"]] = row
    if any(set(roles) != {"source_primary", "target_first_recurrence"} for roles in by_patient.values()):
        raise RuntimeError("Every effective patient must have one source and one first recurrence")
    specimens = {row["metadata_specimen_id"].upper() for row in mapping}
    labels = load_metadata_labels(specimens)

    expression_raw: dict[str, tuple[list[str], np.ndarray]] = {}
    qc_rows: list[dict[str, Any]] = []
    all_genes: set[str] = set()
    with tarfile.open(TAR_PATH, mode="r:") as archive:
        groups = tar_groups(archive)
        for patient_id in sorted(by_patient):
            source = by_patient[patient_id]["source_primary"]
            specimen = source["specimen_id"].upper()
            group_id = file_map[(patient_id, "source_primary")]
            members = groups[group_id]
            feature_lines = read_gzip_lines(archive, members["features"])
            gene_ids = [line.split("\t")[0] for line in feature_lines]
            if len(gene_ids) != len(set(gene_ids)):
                raise RuntimeError(f"Duplicate gene IDs in {group_id}")
            barcodes_full = read_gzip_lines(archive, members["barcodes"])
            normalized_barcodes = [value.split("-", 1)[0] for value in barcodes_full]
            if len(normalized_barcodes) != len(set(normalized_barcodes)):
                raise RuntimeError(f"Duplicate normalized barcodes in canonical group {group_id}")
            specimen_labels = labels.get(source["metadata_specimen_id"].upper(), {})
            barcode_normalization = source.get("barcode_normalization", "IDENTITY")
            lookup_barcodes = [
                f"{barcode}.1" if barcode_normalization == "APPEND_DOT1" else barcode
                for barcode in normalized_barcodes
            ]
            valid_columns = {
                index + 1
                for index, barcode in enumerate(lookup_barcodes)
                if specimen_labels.get(barcode) in {"Tumor", "NonTumor"}
            }
            sums, observed_nnz, nul_bytes = sum_matrix_by_gene(archive, members["matrix"], valid_columns, len(gene_ids))
            if nul_bytes:
                raise RuntimeError(f"Included source matrix unexpectedly has {nul_bytes} NUL bytes: {group_id}")
            expression_raw[patient_id] = (gene_ids, sums)
            all_genes.update(gene_ids)
            qc_rows.append(
                {
                    "patient_id": patient_id,
                    "partition": split[patient_id],
                    "source_matrix_group_hash": hashlib.sha256(group_id.encode()).hexdigest(),
                    "matrix_barcodes": len(normalized_barcodes),
                    "metadata_matched_valid_cells": len(valid_columns),
                    "metadata_unmatched_matrix_cells": len(normalized_barcodes) - len(valid_columns),
                    "gene_count": len(gene_ids),
                    "observed_nnz": observed_nnz,
                    "raw_library_sum": float(sums.sum()),
                    "status": "PASS" if valid_columns and sums.sum() > 0 else "FAIL",
                }
            )
    if not all(row["status"] == "PASS" for row in qc_rows):
        raise RuntimeError("At least one retained source pseudobulk failed QC")

    gene_universe = tuple(sorted(all_genes))
    gene_index = {gene: i for i, gene in enumerate(gene_universe)}
    patient_ids = tuple(sorted(expression_raw))
    logcpm = np.zeros((len(patient_ids), len(gene_universe)), dtype=np.float64)
    for i, patient_id in enumerate(patient_ids):
        genes, sums = expression_raw[patient_id]
        indices = np.fromiter((gene_index[gene] for gene in genes), dtype=np.int64, count=len(genes))
        total = float(sums.sum())
        logcpm[i, indices] = np.log1p(sums / total * 1_000_000.0)
    store = SourceExpressionStore(patient_ids, gene_universe, logcpm)
    store_path = DATA / "source_expression_log1p_cpm.npz"
    store.save(store_path)

    pair_rows: list[dict[str, Any]] = []
    for patient_id in patient_ids:
        source = by_patient[patient_id]["source_primary"]
        target = by_patient[patient_id]["target_first_recurrence"]
        source_counts = labels[source["metadata_specimen_id"].upper()]
        target_counts = labels[target["metadata_specimen_id"].upper()]
        def comp(counts: dict[str, str]) -> tuple[float, float, int, int]:
            tumor = sum(value == "Tumor" for value in counts.values())
            nontumor = sum(value == "NonTumor" for value in counts.values())
            valid = tumor + nontumor
            if valid == 0:
                return math.nan, math.nan, 0, len(counts)
            return tumor / valid, nontumor / valid, valid, len(counts) - valid
        st, sn, sv, su = comp(source_counts)
        tt, tn, tv, tu = comp(target_counts)
        target_time = target["elapsed_days"] if target["elapsed_days"] else "NA"
        pair_rows.append(
            {
                "patient_id": patient_id,
                "pair_id": f"{patient_id}:primary-first-recurrence",
                "analysis_partition": split[patient_id],
                "source_time": 0,
                "target_time": target_time,
                "source__Tumor": st,
                "source__NonTumor": sn,
                "target__Tumor": tt,
                "target__NonTumor": tn,
                "source_valid_cells": sv,
                "source_unknown_cells": su,
                "target_valid_cells": tv,
                "target_unknown_cells": tu,
                "simplex_status": "PASS" if abs(st + sn - 1) <= 1e-12 and abs(tt + tn - 1) <= 1e-12 else "FAIL",
            }
        )
    pair_fields = ["patient_id", "pair_id", "analysis_partition", "source_time", "target_time", "source__Tumor", "source__NonTumor", "target__Tumor", "target__NonTumor", "source_valid_cells", "source_unknown_cells", "target_valid_cells", "target_unknown_cells", "simplex_status"]
    write_csv(DATA / "gse174554_processed_pairs.csv", pair_rows, pair_fields)
    write_csv(AUDIT / "SOURCE_PSEUDOBULK_QC.csv", qc_rows, list(qc_rows[0]))
    gene_rows = [{"gene_index": i, "gene_id": gene} for i, gene in enumerate(gene_universe)]
    write_csv(DATA / "source_expression_gene_ids.csv", gene_rows, ["gene_index", "gene_id"])
    status = "SUCCESS" if all(row["simplex_status"] == "PASS" for row in pair_rows) else "FAILED"
    payload = {
        "schema_version": 1,
        "phase": "source_pseudobulk_and_compositions",
        "status": status,
        "started_utc": started,
        "finished_utc": utc_now(),
        "patients": len(patient_ids),
        "development_patients": sum(split[p] == "development" for p in patient_ids),
        "confirmation_patients": sum(split[p] == "confirmation" for p in patient_ids),
        "post_freeze_exclusions": len(split) - len(patient_ids),
        "genes": len(gene_universe),
        "source_expression_store": str(store_path),
        "source_expression_store_file_sha256": sha256_file(store_path),
        "source_expression_content_sha256": store.content_sha256(),
        "recurrence_expression_opened": False,
        "simplex_tolerance": 1e-12,
        "batch2_policy": "canonical_non_batch2_group_only; batch2 groups retained in validation audit",
    }
    atomic_json(LOGS / "phase05_pseudobulk.json", payload)
    atomic_json(LOGS / "pipeline_checkpoint.json", {"last_completed_phase": "source_pseudobulk_and_compositions" if status == "SUCCESS" else "metadata_and_matrix_mapping", "status": status, "updated_utc": utc_now()})
    report = [
        "# Stage 5: source-only pseudobulk and K=2 compositions",
        "",
        f"- Status: `{status}`",
        f"- Effective patients: `{len(patient_ids)}` (development `{payload['development_patients']}`, confirmation `{payload['confirmation_patients']}`)",
        f"- Frozen-post integrity exclusions: `{payload['post_freeze_exclusions']}`; no rehash or replacement.",
        f"- Source gene universe: `{len(gene_universe)}`",
        "- Expression transform: patient primary raw sums -> CPM -> log1p.",
        "- Recurrence expression opened: `false`.",
        "- Composition mapping: author `Tumor` and `Normal -> NonTumor`; unknown labels excluded from denominator.",
        "- Simplex tolerance: `1e-12`.",
    ]
    atomic_text(AUDIT / "STAGE05_PSEUDOBULK_REPORT.md", "\n".join(report) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if status == "SUCCESS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
