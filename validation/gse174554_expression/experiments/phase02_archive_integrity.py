from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
AUDIT = ROOT / "audit"
LOGS = ROOT / "logs"
RECON = AUDIT / "reconstructed"
ORIGINAL_TAR = PROJECT / "GSE174554_RAW.tar"
RAR_FIRST = PROJECT / "GSE174554_RAW.part1.rar"
SEVEN_ZIP = Path(r"C:\Program Files\7-Zip\7z.exe")


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


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def member_kind(name: str) -> str:
    low = name.lower()
    if low.endswith("barcodes.tsv.gz"):
        return "barcodes"
    if low.endswith("features.tsv.gz"):
        return "features"
    if low.endswith("matrix.mtx.gz"):
        return "matrix"
    if low.endswith(".zip"):
        return "proteomics_zip"
    if low.endswith(".gz"):
        return "other_gzip"
    return "other"


def sample_key(name: str, kind: str) -> str:
    filename = Path(name).name
    patterns = {
        "barcodes": r"_barcodes\.tsv\.gz$",
        "features": r"_features\.tsv\.gz$",
        "matrix": r"_matrix\.mtx\.gz$",
    }
    return re.sub(patterns[kind], "", filename, flags=re.IGNORECASE) if kind in patterns else ""


def build_tar_manifest() -> tuple[list[dict[str, object]], dict[str, set[str]]]:
    rows: list[dict[str, object]] = []
    groups: dict[str, set[str]] = {}
    with tarfile.open(ORIGINAL_TAR, mode="r:") as archive:
        for index, member in enumerate(archive.getmembers()):
            if not member.isfile():
                rows.append(
                    {
                        "index": index,
                        "member_name": member.name,
                        "size_bytes": member.size,
                        "mtime_utc": datetime.fromtimestamp(member.mtime, timezone.utc).isoformat(timespec="seconds"),
                        "kind": "non_regular",
                        "sample_key": "",
                        "sha256": "",
                        "gzip_magic": "NOT_APPLICABLE",
                    }
                )
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError(f"Cannot read TAR member: {member.name}")
            digest = hashlib.sha256()
            first = b""
            read_bytes = 0
            while chunk := stream.read(8 * 1024 * 1024):
                if not first:
                    first = chunk[:2]
                digest.update(chunk)
                read_bytes += len(chunk)
            if read_bytes != member.size:
                raise RuntimeError(f"Short TAR member read: {member.name}: {read_bytes} != {member.size}")
            kind = member_kind(member.name)
            key = sample_key(member.name, kind)
            if key:
                groups.setdefault(key, set()).add(kind)
            rows.append(
                {
                    "index": index,
                    "member_name": member.name,
                    "size_bytes": member.size,
                    "mtime_utc": datetime.fromtimestamp(member.mtime, timezone.utc).isoformat(timespec="seconds"),
                    "kind": kind,
                    "sample_key": key,
                    "sha256": digest.hexdigest(),
                    "gzip_magic": "PASS" if member.name.lower().endswith(".gz") and first == b"\x1f\x8b" else ("FAIL" if member.name.lower().endswith(".gz") else "NOT_APPLICABLE"),
                }
            )
    return rows, groups


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Refusing to write empty CSV")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=path.parent) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        tmp = Path(f.name)
    os.replace(tmp, path)


def reconstruct_rar() -> dict[str, object]:
    record: dict[str, object] = {"status": "NOT_EXECUTED", "tool": str(SEVEN_ZIP)}
    if not SEVEN_ZIP.exists():
        record["reason"] = "7-Zip not found"
        return record
    output_tar = RECON / "GSE174554_RAW.tar"
    if output_tar.exists():
        output_tar.unlink()
    command = [str(SEVEN_ZIP), "x", "-y", f"-o{RECON}", str(RAR_FIRST)]
    started = time.time()
    result = subprocess.run(command, text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    atomic_text(LOGS / "rar_reconstruction_raw.txt", result.stdout)
    record.update({"command": command, "returncode": result.returncode, "runtime_seconds": time.time() - started, "output": str(output_tar)})
    if result.returncode != 0 or not output_tar.is_file():
        record.update({"status": "FAIL", "reason": "7-Zip extraction failed or TAR absent"})
        return record
    original_hash = sha256_file(ORIGINAL_TAR)
    reconstructed_hash = sha256_file(output_tar)
    record.update(
        {
            "original_size_bytes": ORIGINAL_TAR.stat().st_size,
            "reconstructed_size_bytes": output_tar.stat().st_size,
            "original_sha256": original_hash,
            "reconstructed_sha256": reconstructed_hash,
            "status": "PASS" if ORIGINAL_TAR.stat().st_size == output_tar.stat().st_size and original_hash == reconstructed_hash else "FAIL",
        }
    )
    return record


def main() -> int:
    started = utc_now()
    rows, groups = build_tar_manifest()
    manifest_path = AUDIT / "ORIGINAL_TAR_MEMBER_MANIFEST.csv"
    write_csv(manifest_path, rows)
    required = {"barcodes", "features", "matrix"}
    complete = sorted(k for k, values in groups.items() if required.issubset(values))
    incomplete = sorted(k for k, values in groups.items() if not required.issubset(values))
    # A physical snRNA specimen may have a separate ``_batch2`` MTX triple and
    # some later GEO deposits use an ``SF####v2`` spelling.  The scientific
    # inventory is therefore the set of numeric SF specimen identifiers, not
    # the number of raw MTX triples.  snATAC groups intentionally lack a genes
    # feature file and are not expression inputs.
    expression_sf_ids = sorted(
        {
            match.group(1)
            for key in complete
            if (match := re.search(r"_(SF\d+)(?:v2)?(?:_batch2)?$", key, re.IGNORECASE))
        }
    )
    incomplete_snatac = sorted(k for k in incomplete if "_snATAC" in k)
    unexpected_incomplete = sorted(set(incomplete) - set(incomplete_snatac))
    group_rows = [
        {
            "sample_key": key,
            "has_barcodes": "barcodes" in groups[key],
            "has_features": "features" in groups[key],
            "has_matrix": "matrix" in groups[key],
            "status": "COMPLETE" if required.issubset(groups[key]) else "INCOMPLETE",
        }
        for key in sorted(groups)
    ]
    write_csv(AUDIT / "MTX_GROUP_INVENTORY.csv", group_rows)
    reconstruction = reconstruct_rar()
    regular_count = sum(row["kind"] != "non_regular" for row in rows)
    gzip_count = sum(str(row["member_name"]).lower().endswith(".gz") for row in rows)
    gzip_fail = sum(row["gzip_magic"] == "FAIL" for row in rows)
    proteomics_count = sum(row["kind"] == "proteomics_zip" for row in rows)
    original_structure_pass = (
        regular_count == 329
        and gzip_count == 323
        and proteomics_count == 6
        and len(complete) == 93
        and len(expression_sf_ids) == 78
        and len(incomplete_snatac) == 10
        and not unexpected_incomplete
        and gzip_fail == 0
    )
    status = "SUCCESS" if original_structure_pass and reconstruction["status"] == "PASS" else ("PARTIAL" if original_structure_pass else "FAILED")
    payload = {
        "schema_version": 1,
        "phase": "archive_integrity",
        "status": status,
        "started_utc": started,
        "finished_utc": utc_now(),
        "analysis_input": str(ORIGINAL_TAR),
        "analysis_input_sha256": sha256_file(ORIGINAL_TAR),
        "tar_member_count": len(rows),
        "regular_file_count": regular_count,
        "gzip_file_count": gzip_count,
        "gzip_magic_failures": gzip_fail,
        "proteomics_zip_count": proteomics_count,
        "complete_raw_mtx_groups": len(complete),
        "unique_expression_sf_specimens": len(expression_sf_ids),
        "incomplete_snatac_groups_excluded": len(incomplete_snatac),
        "unexpected_incomplete_mtx_groups": len(unexpected_incomplete),
        "original_tar_structural_check": "PASS" if original_structure_pass else "FAIL",
        "rar_reconstruction": reconstruction,
        "target_metadata_opened": False,
    }
    atomic_json(LOGS / "phase02_archive_integrity.json", payload)
    checkpoint = {"last_completed_phase": "archive_integrity" if status in {"SUCCESS", "PARTIAL"} else "provenance_baseline", "status": status, "updated_utc": utc_now()}
    atomic_json(LOGS / "pipeline_checkpoint.json", checkpoint)
    report = [
        "# Stage 2: archive integrity",
        "",
        f"- Status: `{status}`",
        f"- Original TAR structural check: `{'PASS' if original_structure_pass else 'FAIL'}`",
        f"- TAR members: `{len(rows)}` ({regular_count} regular files)",
        f"- Gzip members: `{gzip_count}`; gzip-magic failures: `{gzip_fail}`",
        f"- Proteomics ZIP members: `{proteomics_count}` (excluded from expression modeling)",
        f"- Complete raw MTX triples: `{len(complete)}`",
        f"- Unique numeric SF expression specimens after batch/v2 canonicalization: `{len(expression_sf_ids)}`",
        f"- Incomplete snATAC groups excluded: `{len(incomplete_snatac)}`; unexpected incomplete groups: `{len(unexpected_incomplete)}`",
        f"- Multipart RAR reconstruction: `{reconstruction['status']}`",
        f"- Analysis input remains: `{ORIGINAL_TAR}`",
        "- Target metadata opened: `false`",
    ]
    atomic_text(AUDIT / "STAGE02_ARCHIVE_INTEGRITY_REPORT.md", "\n".join(report) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if original_structure_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
