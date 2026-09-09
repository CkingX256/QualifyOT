from __future__ import annotations

"""Utilities for pre-specified external-confirmation contracts.

The functions here are deliberately small and auditable. They provide:
1. canonical SHA256 hashing of YAML contracts;
2. fail-closed verification against a stored digest;
3. deterministic patient partitioning that can be frozen before patient IDs
   are revealed, then applied once metadata are available.

They do not inspect target outcomes.
"""

from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

import yaml


def _canonical_json_bytes(obj) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def load_yaml_contract(path: str | Path) -> dict:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("external confirmation contract must be a YAML mapping")
    return data


def contract_sha256(path: str | Path) -> str:
    """Hash contract semantics rather than YAML whitespace/comments."""
    return sha256(_canonical_json_bytes(load_yaml_contract(path))).hexdigest()


def dependency_sha256(path: str | Path) -> tuple[str, str]:
    """Return (digest_type, sha256) for an immutable gate dependency.

    YAML dependencies are hashed semantically so comments/whitespace do not
    invalidate a gate. Other files use their raw byte SHA256.
    """
    path = Path(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        return "yaml-semantic", contract_sha256(path)
    return "raw-bytes", sha256(path.read_bytes()).hexdigest()


def write_contract_digest(path: str | Path, digest_path: str | Path | None = None) -> Path:
    path = Path(path)
    digest_path = Path(digest_path) if digest_path else path.with_suffix(path.suffix + ".sha256")
    digest = contract_sha256(path)
    digest_path.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest_path


def verify_contract_digest(path: str | Path, digest_path: str | Path | None = None) -> bool:
    path = Path(path)
    digest_path = Path(digest_path) if digest_path else path.with_suffix(path.suffix + ".sha256")
    expected = digest_path.read_text(encoding="utf-8").strip().split()[0]
    actual = contract_sha256(path)
    if expected != actual:
        raise ValueError(f"contract SHA256 mismatch: expected {expected}, got {actual}")
    return True


def patient_set_sha256(patient_ids: Iterable[str]) -> str:
    ids = sorted({str(x) for x in patient_ids})
    if not ids:
        raise ValueError("patient set cannot be empty")
    return sha256(_canonical_json_bytes(ids)).hexdigest()


def deterministic_patient_split(
    patient_ids: Iterable[str],
    *,
    development_fraction: float,
    salt: str,
) -> dict[str, str]:
    """Deterministically assign physical patients without using outcomes.

    Each patient is mapped to U[0,1) by SHA256(salt || patient_id). The split
    therefore depends only on the frozen salt, the patient identity and the
    frozen development fraction.
    """
    if not (0.0 < float(development_fraction) < 1.0):
        raise ValueError("development_fraction must lie strictly between 0 and 1")
    ids = sorted({str(x) for x in patient_ids})
    if len(ids) < 2:
        raise ValueError("at least two physical patients are required")
    out: dict[str, str] = {}
    for pid in ids:
        raw = sha256(f"{salt}\0{pid}".encode("utf-8")).digest()
        u = int.from_bytes(raw[:8], "big") / 2**64
        out[pid] = "development" if u < development_fraction else "confirmation"
    # Fail closed if a tiny cohort happens to hash entirely to one arm.
    if len(set(out.values())) != 2:
        raise ValueError("deterministic split produced an empty arm; contract requires a new pre-specified salt")
    return out


def freeze_patient_set_and_split(
    patient_ids: Iterable[str],
    *,
    contract_path: str | Path,
    patient_file: str | Path,
    split_file: str | Path,
    gate_file: str | Path,
    dependency_paths: Iterable[str | Path] = (),
) -> dict:
    """Freeze an eligible physical-patient set before target extraction.

    The contract must contain ``inference.split_salt`` and
    ``inference.development_fraction``.  Outputs are deterministic and include a
    gate record that target-extraction code can verify before reading outcomes.
    Optional dependency files (for example the master contract and external
    method registry) are hashed into the gate so the entire analysis-contract
    dependency closure is fail-closed, not only the cohort YAML.
    """
    import csv
    from datetime import datetime, timezone

    contract_path = Path(contract_path)
    contract = load_yaml_contract(contract_path)
    inf = contract.get("inference") or {}
    salt = inf.get("split_salt")
    frac = inf.get("development_fraction")
    if not isinstance(salt, str) or not salt:
        raise ValueError("contract inference.split_salt must be a non-empty string")
    if frac is None:
        raise ValueError("contract inference.development_fraction is required")
    ids = sorted({str(x).strip() for x in patient_ids if str(x).strip()})
    if len(ids) < 2:
        raise ValueError("at least two eligible physical patients are required")
    assignments = deterministic_patient_split(ids, development_fraction=float(frac), salt=salt)

    patient_file = Path(patient_file); split_file = Path(split_file); gate_file = Path(gate_file)
    for p in (patient_file, split_file, gate_file):
        p.parent.mkdir(parents=True, exist_ok=True)
    patient_file.write_text("\n".join(ids) + "\n", encoding="utf-8")
    with split_file.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["patient_id", "split"])
        for pid in ids:
            w.writerow([pid, assignments[pid]])

    split_sha = sha256(split_file.read_bytes()).hexdigest()
    dependencies = []
    seen = set()
    for raw_path in dependency_paths:
        dep = Path(raw_path)
        key = str(dep.resolve())
        if key in seen:
            continue
        seen.add(key)
        digest_type, digest = dependency_sha256(dep)
        dependencies.append({
            "file": dep.name,
            "path_at_freeze": str(dep),
            "digest_type": digest_type,
            "sha256": digest,
        })
    gate = {
        "schema": "qualifyot-target-access-gate-v1",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "contract_file": contract_path.name,
        "contract_semantic_sha256": contract_sha256(contract_path),
        "patient_file": patient_file.name,
        "patient_set_sha256": patient_set_sha256(ids),
        "split_file": split_file.name,
        "split_file_sha256": split_sha,
        "dependencies": dependencies,
        "n_patients": len(ids),
        "n_development": sum(v == "development" for v in assignments.values()),
        "n_confirmation": sum(v == "confirmation" for v in assignments.values()),
        "target_access_allowed": True,
        "guardrail": "This gate certifies only pre-outcome contract/patient/split freezing. It does not certify absence of prior literature knowledge.",
    }
    gate_file.write_text(json.dumps(gate, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return gate


def verify_target_access_gate(
    *,
    contract_path: str | Path,
    patient_file: str | Path,
    split_file: str | Path,
    gate_file: str | Path,
    dependency_paths: Iterable[str | Path] | None = None,
) -> bool:
    """Fail closed if contract, patient set or deterministic split changed."""
    contract_path = Path(contract_path); patient_file = Path(patient_file); split_file = Path(split_file); gate_file = Path(gate_file)
    gate = json.loads(gate_file.read_text(encoding="utf-8"))
    if gate.get("schema") != "qualifyot-target-access-gate-v1" or not gate.get("target_access_allowed"):
        raise ValueError("invalid or disabled target-access gate")
    if gate.get("contract_semantic_sha256") != contract_sha256(contract_path):
        raise ValueError("contract changed after patient-set freeze")
    ids = [x.strip() for x in patient_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    if gate.get("patient_set_sha256") != patient_set_sha256(ids):
        raise ValueError("patient set changed after freeze")
    if gate.get("split_file_sha256") != sha256(split_file.read_bytes()).hexdigest():
        raise ValueError("split file changed after freeze")

    frozen_deps = gate.get("dependencies") or []
    if dependency_paths is None:
        dependency_paths = [d.get("path_at_freeze") for d in frozen_deps]
    provided = {Path(p).name: Path(p) for p in dependency_paths if p}
    frozen_names = {d.get("file") for d in frozen_deps}
    if set(provided) != frozen_names:
        raise ValueError("target-access dependency set differs from frozen gate")
    for dep in frozen_deps:
        path = provided[dep["file"]]
        dtype, digest = dependency_sha256(path)
        if dtype != dep.get("digest_type") or digest != dep.get("sha256"):
            raise ValueError(f"dependency changed after freeze: {dep['file']}")
    return True
