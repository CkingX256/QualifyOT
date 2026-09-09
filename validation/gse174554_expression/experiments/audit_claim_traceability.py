from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def display_number(value: object, digits: int = 4) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite manuscript number: {value}")
    return f"{number:.{digits}f}"


def build_claims() -> list[dict[str, str]]:
    phase05_path = ROOT / "logs" / "phase05_pseudobulk.json"
    phase06_path = ROOT / "logs" / "phase06_leakage_gate.json"
    phase07_path = ROOT / "logs" / "phase07_honest_confirmation.json"
    phase09_path = ROOT / "logs" / "phase09_robustness_delete_one.json"
    phase05 = json.loads(phase05_path.read_text(encoding="utf-8"))
    phase06 = json.loads(phase06_path.read_text(encoding="utf-8"))
    phase07 = json.loads(phase07_path.read_text(encoding="utf-8"))
    phase09 = json.loads(phase09_path.read_text(encoding="utf-8"))
    honest_path = ROOT / "results" / "honest_confirmation" / "honest_candidate_summary.csv"
    honest = read_csv(honest_path)
    expression_index = next(index for index, row in enumerate(honest) if row["candidate"] == "ExpressionElasticNet")
    expression = honest[expression_index]
    lopo_path = ROOT / "results" / "dependent_lopo" / "lopo_seed_summary.csv"
    lopo = read_csv(lopo_path)
    primary_lopo_index = next(
        index
        for index, row in enumerate(lopo)
        if row["candidate"] == "ExpressionElasticNet" and row["seed"] == "20260903"
    )
    primary_lopo = lopo[primary_lopo_index]
    winner = json.loads((ROOT / "results" / "dependent_lopo" / "accuracy_winner.json").read_text(encoding="utf-8"))

    claims: list[dict[str, str]] = []

    def add(claim_id: str, display: str, source: Path, locator: str, raw: object) -> None:
        claims.append(
            {
                "claim_id": claim_id,
                "display_value": display,
                "source_file": source.relative_to(ROOT).as_posix(),
                "source_locator": locator,
                "raw_value": str(raw),
            }
        )

    add("analysis_patients", str(phase05["patients"]), phase05_path, "patients", phase05["patients"])
    add("development_patients", str(phase05["development_patients"]), phase05_path, "development_patients", phase05["development_patients"])
    add("confirmation_patients", str(phase05["confirmation_patients"]), phase05_path, "confirmation_patients", phase05["confirmation_patients"])
    add("leakage_checks", str(phase06["checks"]), phase06_path, "checks", phase06["checks"])
    add("failed_leakage_checks", str(phase06["failed_checks"]), phase06_path, "failed_checks", phase06["failed_checks"])
    add("honest_qualified_holm", str(phase07["qualified_holm"]), phase07_path, "qualified_holm", phase07["qualified_holm"])
    for column in ("selected_lambda", "movement_estimate", "movement_lcb", "utility_estimate", "utility_lcb", "retention_estimate", "retention_lcb", "candidate_iut_p", "holm_adjusted_p", "candidate_mae", "reference_mae", "retained_mae"):
        add(
            f"expression_{column}",
            display_number(expression[column]),
            honest_path,
            f"row={expression_index + 2};candidate=ExpressionElasticNet;column={column}",
            expression[column],
        )
    for column in ("PDR", "PDR_lo", "PDR_hi", "PUC", "PUC_lo", "PUC_hi", "NPI", "NPI_lo", "NPI_hi", "candidate_mae", "reference_mae", "retained_mae"):
        add(
            f"lopo_expression_{column}",
            display_number(primary_lopo[column]),
            lopo_path,
            f"row={primary_lopo_index + 2};candidate=ExpressionElasticNet;seed=20260903;column={column}",
            primary_lopo[column],
        )
    add("lopo_accuracy_winner", str(winner["candidate"]), ROOT / "results" / "dependent_lopo" / "accuracy_winner.json", "candidate", winner["candidate"])
    add("delete_one_success", str(phase09["delete_one_success"]), phase09_path, "delete_one_success", phase09["delete_one_success"])
    add("delete_one_failed", str(phase09["delete_one_failed"]), phase09_path, "delete_one_failed", phase09["delete_one_failed"])
    add("determinism_pass", str(phase09["determinism_pass"]), phase09_path, "determinism_pass", phase09["determinism_pass"])
    add("determinism_fail", str(phase09["determinism_fail"]), phase09_path, "determinism_fail", phase09["determinism_fail"])
    return claims


def verify_claims(claims: list[dict[str, str]], manuscript_paths: list[Path]) -> tuple[bool, list[str]]:
    combined = "\n".join(path.read_text(encoding="utf-8", errors="strict") for path in manuscript_paths)
    missing = [row["claim_id"] for row in claims if f"CLAIM:{row['claim_id']}={row['display_value']}" not in combined]
    return not missing, missing


def write_claims(path: Path, claims: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["claim_id", "display_value", "source_file", "source_locator", "raw_value"])
        writer.writeheader()
        writer.writerows(claims)

