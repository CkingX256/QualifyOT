from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable


GSE174554_SPLIT_SALT = "QualifyOT|GSE174554|primary-first-recurrence|expression-v1"


def exact_sorted_hash_split(patient_ids: Iterable[str]) -> list[dict[str, object]]:
    """Return the frozen exact 60/40 patient split.

    This deliberately does not use the generic QualifyOT split helper.  It
    hashes ``salt + '|' + canonical_patient_id``, sorts complete hex digests
    lexicographically, and assigns the first ``floor(0.60*N)`` to development.
    """

    ids = [str(value) for value in patient_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("Patient IDs must be unique")
    if any(not value for value in ids):
        raise ValueError("Patient IDs must be non-empty")
    rows = []
    for patient_id in ids:
        material = f"{GSE174554_SPLIT_SALT}|{patient_id}".encode("utf-8")
        rows.append({"patient_id": patient_id, "split_sha256": hashlib.sha256(material).hexdigest()})
    rows.sort(key=lambda row: (str(row["split_sha256"]), str(row["patient_id"])))
    development_n = math.floor(0.60 * len(rows))
    for rank, row in enumerate(rows, start=1):
        row["hash_rank"] = rank
        row["analysis_partition"] = "development" if rank <= development_n else "confirmation"
    return rows

