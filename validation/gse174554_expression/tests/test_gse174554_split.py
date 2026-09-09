from __future__ import annotations

import hashlib
import math

from qualifyot_gse174554.split import GSE174554_SPLIT_SALT, exact_sorted_hash_split


def test_exact_sorted_hash_split_matches_literal_specification() -> None:
    patient_ids = [f"GSE174554-P{i:03d}" for i in range(1, 37)]
    observed = exact_sorted_hash_split(reversed(patient_ids))
    expected = sorted(
        (
            hashlib.sha256(f"{GSE174554_SPLIT_SALT}|{patient_id}".encode()).hexdigest(),
            patient_id,
        )
        for patient_id in patient_ids
    )
    assert [(r["split_sha256"], r["patient_id"]) for r in observed] == expected
    assert sum(r["analysis_partition"] == "development" for r in observed) == math.floor(0.60 * 36)
    assert sum(r["analysis_partition"] == "confirmation" for r in observed) == 15


def test_exact_sorted_hash_split_rejects_duplicate_patient() -> None:
    try:
        exact_sorted_hash_split(["GSE174554-P001", "GSE174554-P001"])
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate patients were accepted")
