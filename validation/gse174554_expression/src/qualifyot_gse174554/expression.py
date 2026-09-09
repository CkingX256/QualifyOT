from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SourceExpressionStore:
    """Immutable patient-level source-only log1p(CPM) matrix."""

    patient_ids: tuple[str, ...]
    gene_ids: tuple[str, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.shape != (len(self.patient_ids), len(self.gene_ids)):
            raise ValueError("source expression shape does not match IDs")
        if len(set(self.patient_ids)) != len(self.patient_ids):
            raise ValueError("source patient IDs must be unique")
        if len(set(self.gene_ids)) != len(self.gene_ids):
            raise ValueError("source gene IDs must be unique")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("source logCPM must be finite and non-negative")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)

    def rows(self, patient_ids: list[str] | tuple[str, ...] | np.ndarray) -> np.ndarray:
        index = {patient: i for i, patient in enumerate(self.patient_ids)}
        missing = [str(patient) for patient in patient_ids if str(patient) not in index]
        if missing:
            raise KeyError(f"source expression missing patients: {missing}")
        return np.vstack([self.values[index[str(patient)]] for patient in patient_ids])

    def content_sha256(self) -> str:
        digest = hashlib.sha256()
        for patient in self.patient_ids:
            digest.update(patient.encode("utf-8") + b"\0")
        for gene in self.gene_ids:
            digest.update(gene.encode("utf-8") + b"\0")
        digest.update(np.ascontiguousarray(self.values).tobytes())
        return digest.hexdigest()

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            Path(path),
            patient_ids=np.asarray(self.patient_ids, dtype="U"),
            gene_ids=np.asarray(self.gene_ids, dtype="U"),
            values=np.asarray(self.values, dtype=np.float64),
            content_sha256=np.asarray([self.content_sha256()], dtype="U"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SourceExpressionStore":
        with np.load(Path(path), allow_pickle=False) as payload:
            store = cls(
                tuple(payload["patient_ids"].astype(str).tolist()),
                tuple(payload["gene_ids"].astype(str).tolist()),
                payload["values"].astype(np.float64),
            )
            expected = str(payload["content_sha256"][0])
        if store.content_sha256() != expected:
            raise ValueError("SourceExpressionStore content hash mismatch")
        return store

