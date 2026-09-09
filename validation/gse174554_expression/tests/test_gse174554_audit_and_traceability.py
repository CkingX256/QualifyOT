from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from audit_claim_traceability import verify_claims


def _split_digest(patient_id: str) -> str:
    salt = "QualifyOT|GSE174554|primary-first-recurrence|expression-v1"
    return hashlib.sha256(f"{salt}|{patient_id}".encode()).hexdigest()


def test_frozen_split_uses_exact_sorted_hash_and_is_disjoint():
    path = ROOT / "audit" / "FROZEN_PATIENT_SPLIT.csv"
    if not path.exists():
        return
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ordered = sorted(rows, key=lambda row: row["split_sha256"])
    assert [row["patient_id"] for row in rows] == [row["patient_id"] for row in ordered]
    assert all(row["split_sha256"] == _split_digest(row["patient_id"]) for row in rows)
    cutoff = int(0.60 * len(rows))
    assert all(row["analysis_partition"] == "development" for row in rows[:cutoff])
    assert all(row["analysis_partition"] == "confirmation" for row in rows[cutoff:])
    assert len({row["patient_id"] for row in rows}) == len(rows)


def test_traceability_requires_exact_claim_markers(tmp_path: Path):
    claims = [{"claim_id": "n", "display_value": "26", "source_file": "x.csv", "source_locator": "row=2", "raw_value": "26"}]
    good = tmp_path / "good.tex"
    good.write_text("% CLAIM:n=26\n", encoding="utf-8")
    assert verify_claims(claims, [good]) == (True, [])
    good.write_text("% CLAIM:n=25\n", encoding="utf-8")
    passed, missing = verify_claims(claims, [good])
    assert not passed and missing == ["n"]


def test_packaging_excludes_identity_maps_and_raw_matrices():
    script = (ROOT / "experiments" / "phase11_package_verify.py").read_text(encoding="utf-8")
    assert "local_identity_mapping_EXCLUDE_FROM_PACKAGE" in script
    assert "FROZEN_SPECIMEN_METADATA_ALIAS_MAP.csv" in script
    assert "GEO expression matrices" in script

