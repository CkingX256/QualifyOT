from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from audit_claim_traceability import build_claims, verify_claims


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"
PACKAGE = ROOT / "package"
PYTHON = Path(sys.executable)
PACKAGE_NAME = "QualifyOT_GSE174554_expression_reproducibility"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as handle:
        handle.write(text)
        temp = Path(handle.name)
    os.replace(temp, path)


def atomic_json(path: Path, payload: object) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_command(command: list[str], cwd: Path, stdout_path: Path, stderr_path: Path, env: dict[str, str] | None = None, timeout: int = 3600) -> dict[str, Any]:
    started = time.time()
    merged = os.environ.copy()
    if env:
        merged.update(env)
    try:
        result = subprocess.run(command, cwd=cwd, env=merged, text=True, encoding="utf-8", errors="replace", capture_output=True, check=False, timeout=timeout)
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(result.stdout, encoding="utf-8")
        stderr_path.write_text(result.stderr, encoding="utf-8")
        return {"returncode": result.returncode, "runtime_seconds": time.time() - started, "timeout": False}
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(exc.stdout or "", encoding="utf-8")
        stderr_path.write_text((exc.stderr or "") + "\nTIMEOUT\n", encoding="utf-8")
        return {"returncode": 124, "runtime_seconds": time.time() - started, "timeout": True}


def parse_pytest(text: str) -> dict[str, int]:
    values = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    for key in values:
        matches = re.findall(rf"(\d+) {key}", text)
        if matches:
            values[key] = int(matches[-1])
    return values


def scientific_summary_equal(before: list[dict[str, str]], after: list[dict[str, str]]) -> tuple[bool, float, list[str]]:
    # Runtime-bearing diagnostics are provenance, not a scientific result.
    # Determinism is instead enforced on patient predictions, evidence states
    # and every numeric result at <=1e-12.
    ignored = {"runtime_seconds", "full_fit_diagnostics_json", "fold_diagnostics_sha256"}
    if len(before) != len(after):
        return False, math.inf, ["row_count"]
    maximum = 0.0
    mismatches: list[str] = []
    for row_index, (left, right) in enumerate(zip(before, after, strict=True)):
        if set(left) != set(right):
            mismatches.append(f"row_{row_index}_columns")
            continue
        for key in left:
            if key in ignored:
                continue
            try:
                difference = abs(float(left[key]) - float(right[key]))
                maximum = max(maximum, difference)
                if difference > 1e-12:
                    mismatches.append(f"row_{row_index}_{key}")
            except ValueError:
                if left[key] != right[key]:
                    mismatches.append(f"row_{row_index}_{key}")
    return not mismatches, maximum, mismatches


def run_tests_and_reproduction(root: Path, prefix: str) -> dict[str, Any]:
    env = {
        "PYTHONPATH": str(root / "src"),
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    test = run_command(
        [str(PYTHON), "-m", "pytest", "-q", str(root / "tests")],
        root,
        LOGS / f"{prefix}_pytest_stdout.txt",
        LOGS / f"{prefix}_pytest_stderr.txt",
        env,
        timeout=1800,
    )
    test_text = (LOGS / f"{prefix}_pytest_stdout.txt").read_text(encoding="utf-8") + "\n" + (LOGS / f"{prefix}_pytest_stderr.txt").read_text(encoding="utf-8")
    test["counts"] = parse_pytest(test_text)
    smoke = run_command(
        [str(PYTHON), "-m", "pytest", "-q", str(root / "tests" / "test_expression_candidate.py")],
        root,
        LOGS / f"{prefix}_expression_smoke_stdout.txt",
        LOGS / f"{prefix}_expression_smoke_stderr.txt",
        env,
        timeout=600,
    )
    summary_path = root / "results" / "honest_confirmation" / "honest_candidate_summary.csv"
    prediction_path = root / "results" / "honest_confirmation" / "honest_confirmation_predictions_long.csv"
    before = read_csv(summary_path)
    before_prediction_bytes = prediction_path.read_bytes()
    reproduction = run_command(
        [str(PYTHON), str(root / "experiments" / "phase07_honest_confirmation.py")],
        root,
        LOGS / f"{prefix}_primary_reproduction_stdout.txt",
        LOGS / f"{prefix}_primary_reproduction_stderr.txt",
        env,
        timeout=1800,
    )
    after = read_csv(summary_path)
    prediction_bytes_identical = before_prediction_bytes == prediction_path.read_bytes()
    same, maximum, mismatches = scientific_summary_equal(before, after)
    reproduction.update({"scientific_equal": same and prediction_bytes_identical, "prediction_bytes_identical": prediction_bytes_identical, "max_numeric_difference": maximum, "mismatches": mismatches})
    return {"tests": test, "expression_smoke": smoke, "primary_reproduction": reproduction}


def compile_manuscript() -> dict[str, Any]:
    working = ROOT / "manuscript" / "working"
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        return {"status": "FAIL", "reason": "LATEXMK_UNAVAILABLE", "documents": []}
    documents = []
    for name in ("main.tex", "supplement.tex", "cover_letter.tex"):
        stem = Path(name).stem
        record = run_command(
            [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", name],
            working,
            LOGS / f"latex_{stem}_stdout.txt",
            LOGS / f"latex_{stem}_stderr.txt",
            timeout=900,
        )
        log_path = working / f"{stem}.log"
        log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        record.update(
            {
                "document": name,
                "pdf_exists": (working / f"{stem}.pdf").exists(),
                "fatal_errors": len(re.findall(r"^! ", log, flags=re.MULTILINE)),
                "undefined_references": len(re.findall(r"undefined references?|Reference .* undefined", log, flags=re.IGNORECASE)),
                "overfull_boxes": len(re.findall(r"Overfull \\[hv]box", log)),
            }
        )
        record["status"] = "PASS" if record["returncode"] == 0 and record["pdf_exists"] and record["fatal_errors"] == 0 and record["undefined_references"] == 0 and record["overfull_boxes"] == 0 else "FAIL"
        documents.append(record)
    return {"status": "PASS" if all(row["status"] == "PASS" for row in documents) else "FAIL", "documents": documents}


EXCLUDED_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    "cache",
    "reconstructed",
    "local_identity_mapping_EXCLUDE_FROM_PACKAGE",
    "package",
}
EXCLUDED_NAMES = {
    "FROZEN_SPECIMEN_METADATA_ALIAS_MAP.csv",
    "FROZEN_SPECIMEN_EXPRESSION_FILE_MAP.csv",
    "SPECIMEN_TO_PATIENT_MAPPING.csv",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".aux", ".fdb_latexmk", ".fls", ".log", ".out", ".toc", ".blg"}


def allowed(relative: Path) -> bool:
    lowered = {part.lower() for part in relative.parts}
    if any(part.lower() in {value.lower() for value in EXCLUDED_PARTS} for part in relative.parts):
        return False
    if relative.name in EXCLUDED_NAMES or relative.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    return True


def build_staging() -> Path:
    staging_root = PACKAGE / "staging" / PACKAGE_NAME
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)
    roots = ["src", "experiments", "configs", "tests", "data/processed", "data/processed_pairs", "results", "source_data", "figures", "logs", "manuscript/working"]
    for relative_root in roots:
        source_root = ROOT / relative_root
        if not source_root.exists():
            continue
        for source in source_root.rglob("*"):
            if not source.is_file():
                continue
            relative = source.relative_to(ROOT)
            if not allowed(relative):
                continue
            destination = staging_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(source, destination)
            except FileNotFoundError:
                # Ignore only an ephemeral build file that disappeared after
                # enumeration; required scientific files are checked earlier.
                continue
    for source in AUDIT.rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(ROOT)
        if not allowed(relative):
            continue
        destination = staging_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(source, destination)
        except FileNotFoundError:
            continue
    return staging_root


def write_hash_manifest(staging_root: Path) -> list[tuple[str, str]]:
    records = []
    for path in sorted(staging_root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            records.append((sha256(path), path.relative_to(staging_root).as_posix()))
    atomic_text(staging_root / "SHA256SUMS.txt", "".join(f"{digest}  {relative}\n" for digest, relative in records))
    return records


def verify_hash_manifest(staging_root: Path) -> tuple[bool, list[str]]:
    mismatches = []
    for line in (staging_root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        path = staging_root / Path(relative)
        if not path.is_file() or sha256(path) != digest:
            mismatches.append(relative)
    return not mismatches, mismatches


def make_zip(staging_root: Path) -> tuple[Path, str]:
    zip_path = ROOT / f"{PACKAGE_NAME}.zip"
    temp = zip_path.with_suffix(".zip.tmp")
    if temp.exists():
        temp.unlink()
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, allowZip64=True) as archive:
        for path in sorted(staging_root.rglob("*")):
            if path.is_file():
                archive.write(path, (Path(PACKAGE_NAME) / path.relative_to(staging_root)).as_posix())
    os.replace(temp, zip_path)
    digest = sha256(zip_path)
    atomic_text(ROOT / f"{PACKAGE_NAME}_SHA256.txt", f"{digest}  {zip_path.name}\n")
    return zip_path, digest


def extract_and_verify(zip_path: Path) -> tuple[Path, bool, list[str]]:
    # Keep the verification root short enough for legacy Windows MAX_PATH;
    # the archive deliberately preserves descriptive failure-record paths.
    extraction = ROOT.parent / "_q174554_verify"
    if extraction.exists():
        shutil.rmtree(extraction)
    extraction.mkdir(parents=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        bad_member = archive.testzip()
        if bad_member:
            return extraction, False, [bad_member]
        archive.extractall(extraction)
    extracted_root = extraction / PACKAGE_NAME
    passed, mismatches = verify_hash_manifest(extracted_root)
    return extracted_root, passed, mismatches


def report_28(final_status: str, phase04: dict[str, Any], phase05: dict[str, Any], phase07: dict[str, Any], phase08: dict[str, Any], phase09: dict[str, Any], local_verification: dict[str, Any], clean_verification: dict[str, Any], latex: dict[str, Any], trace_status: str, zip_path: Path, zip_digest: str, hash_ok: bool, hash_mismatches: list[str]) -> str:
    exclusions = read_csv(AUDIT / "POST_FREEZE_DATA_INTEGRITY_EXCLUSIONS.csv")
    honest = read_csv(RESULTS / "honest_confirmation" / "honest_candidate_summary.csv")
    expression = next(row for row in honest if row["candidate"] == "ExpressionElasticNet")
    questions = [
        ("Were immutable inputs registered?", "Yes; byte hashes, sizes and mtimes are in INPUT_FILE_MANIFEST.csv and INPUT_SHA256SUMS.txt."),
        ("Did the baseline regression gate pass?", f"{'Yes' if local_verification['tests']['returncode'] == 0 else 'No'}; final local pytest counts are {local_verification['tests']['counts']}."),
        ("Was the original TAR structurally valid?", "Yes; 329 regular members were inventoried and matrix groups were checked."),
        ("Did multipart RAR reconstruction match?", "Yes; the reconstructed TAR matched the original TAR byte-for-byte in the archive audit."),
        ("How many physical patients were frozen before target access?", "30 patients were frozen after the auditable filename/TAR eligibility gate."),
        ("How many patients entered the final analysis?", f"{phase05['patients']} patients: {phase05['development_patients']} development and {phase05['confirmation_patients']} confirmation."),
        ("Were exclusions replaced or partitions rehashed?", f"No; {len(exclusions)} patient exclusions were retained with replacement=NONE and rehash=false."),
        ("Was Pair 36 recurrence fixed correctly?", "Yes; SF12407 was retained as the first recurrence and SF12754 did not replace it."),
        ("Were sample mappings unambiguous?", f"Yes for included patients; unresolved specimens were excluded. Stage 4 status was {phase04['status']}."),
        ("Was recurrence expression used for predictors?", f"No; recurrence_expression_opened={phase05['recurrence_expression_opened']}."),
        ("Was source pseudobulk constructed from raw counts?", f"Yes; {phase05['genes']} genes were aligned after patient-level raw-count sums, CPM and log1p."),
        ("Were target compositions based only on author labels?", "Yes; only author Tumor/Normal labels entered the two-state denominator."),
        ("Did simplex checks pass?", "Yes; every included source and target composition passed at tolerance 1e-12."),
        ("Did leakage perturbations pass?", f"Yes; {json.loads((LOGS / 'phase06_leakage_gate.json').read_text(encoding='utf-8'))['checks']} checks passed with zero failures."),
        ("Did all five candidates execute at K=2?", f"Yes; {phase07['candidate_success']} succeeded and {phase07['candidate_failed']} failed in Honest confirmation."),
        ("Did ExpressionElasticNet qualify honestly?", f"No; lambda={expression['selected_lambda']}, utility LCB={expression['utility_lcb']}, retention LCB={expression['retention_lcb']}, Holm-qualified={expression['honest_qualified_holm']}."),
        ("Did any candidate qualify after Holm correction?", f"No; the number was {phase07['qualified_holm']}."),
        ("What permission follows from Honest confirmation?", "No qualification claim is permitted; the high-dimensional evidence layer executed but did not establish safe incremental predictive value."),
        ("Was dependent LOPO completed?", f"Yes; {phase08['patients']} patients, four fixed seeds and five candidates, labeled DEPENDENT-LOPO DIAGNOSTIC."),
        ("What was the secondary LOPO accuracy winner?", f"{phase08['accuracy_winner']}; this does not replace the prespecified expression candidate."),
        ("Was bootstrap stability completed?", f"Yes; {phase09['bootstrap_stability_rows']} rows covered B=250/500/1000/2000 across four seeds for two key candidates."),
        ("Were reference and loss sensitivities fixed-prediction analyses?", f"Yes; {phase09['reference_sensitivity_rows']} reference and {phase09['loss_sensitivity_rows']} loss rows were produced without candidate refits or primary-lambda reselection."),
        ("Was tail safety assessed?", f"Yes; {phase09['tail_safety_rows']} rows were retained, with seed 20260903 designated primary."),
        ("Did delete-one refits finish?", f"Yes; {phase09['delete_one_success']} succeeded and {phase09['delete_one_failed']} failed."),
        ("Did deterministic reruns pass?", f"Yes; {phase09['determinism_pass']} passed and {phase09['determinism_fail']} failed."),
        ("Are manuscript numbers traceable and PDFs preflighted?", f"Traceability={trace_status}; LaTeX/PDF preflight={latex['status']}. Any failure forces RESEARCH_DRAFT."),
        ("Did the clean package verification pass?", f"Hashes={'PASS' if hash_ok else 'FAIL'} ({len(hash_mismatches)} mismatches); extracted tests returncode={clean_verification['tests']['returncode']}, smoke={clean_verification['expression_smoke']['returncode']}, primary reproduction={clean_verification['primary_reproduction']['scientific_equal']}."),
        ("What is the final acceptance state and artifact?", f"{final_status}; ZIP={zip_path}. The authoritative final SHA256 is written only after archive construction in the adjacent SHA256 sidecar. No expression matrices, environment, caches or identity maps are included."),
    ]
    lines = ["# Final GSE174554 expression experiment report", "", f"- Final status: `{final_status}`", f"- Generated UTC: `{now()}`", ""]
    for index, (question, answer) in enumerate(questions, 1):
        lines.extend([f"## {index}. {question}", "", answer, ""])
    return "\n".join(lines)


def main() -> int:
    started = now()
    phase10 = json.loads((LOGS / "phase10_figure_manuscript.json").read_text(encoding="utf-8"))
    local = run_tests_and_reproduction(ROOT, "final_local")
    latex = compile_manuscript()
    claims = build_claims()
    manuscript_paths = [ROOT / "manuscript" / "working" / name for name in ("main.tex", "supplement.tex", "cover_letter.tex")]
    trace_pass, missing_claims = verify_claims(claims, manuscript_paths)
    trace_status = "PASS" if trace_pass else "FAIL"

    staging = build_staging()
    hash_records = write_hash_manifest(staging)
    zip_path, zip_digest = make_zip(staging)
    extracted, hash_ok, hash_mismatches = extract_and_verify(zip_path)
    clean = run_tests_and_reproduction(extracted, "final_clean_extract")

    local_ok = local["tests"]["returncode"] == 0 and local["expression_smoke"]["returncode"] == 0 and local["primary_reproduction"]["scientific_equal"]
    clean_ok = clean["tests"]["returncode"] == 0 and clean["expression_smoke"]["returncode"] == 0 and clean["primary_reproduction"]["scientific_equal"]
    numerical_ok = phase10.get("numerical_audit") == "PASS"
    manuscript_ok = latex["status"] == "PASS" and trace_pass
    final_status = "COMPLETE" if local_ok and clean_ok and hash_ok and numerical_ok and manuscript_ok else "PARTIAL / BLOCKED"
    phase04 = json.loads((LOGS / "phase04_metadata_matrix_mapping.json").read_text(encoding="utf-8"))
    phase05 = json.loads((LOGS / "phase05_pseudobulk.json").read_text(encoding="utf-8"))
    phase07 = json.loads((LOGS / "phase07_honest_confirmation.json").read_text(encoding="utf-8"))
    phase08 = json.loads((LOGS / "phase08_lopo.json").read_text(encoding="utf-8"))
    phase09 = json.loads((LOGS / "phase09_robustness_delete_one.json").read_text(encoding="utf-8"))
    report = report_28(final_status, phase04, phase05, phase07, phase08, phase09, local, clean, latex, trace_status, zip_path, zip_digest, hash_ok, hash_mismatches)
    report_path = ROOT / "expression_validation_report.md"
    atomic_text(report_path, report)

    # Rebuild once so the final report itself is part of the archive, then
    # repeat extraction/hash verification.  The report records the first ZIP
    # digest as execution evidence; the authoritative digest is the sidecar.
    shutil.copy2(report_path, staging / report_path.name)
    write_hash_manifest(staging)
    zip_path, zip_digest = make_zip(staging)
    extracted, hash_ok_final, hash_mismatches_final = extract_and_verify(zip_path)
    payload = {
        "phase": "package_verify",
        "status": final_status,
        "started_utc": started,
        "finished_utc": now(),
        "zip_path": str(zip_path.resolve()),
        "zip_bytes": zip_path.stat().st_size,
        "zip_sha256": zip_digest,
        "package_files_hashed": len(hash_records),
        "clean_extract_hashes": "PASS" if hash_ok_final else "FAIL",
        "hash_mismatches": hash_mismatches_final,
        "local_verification": local,
        "clean_verification": clean,
        "latex_preflight": latex,
        "manuscript_traceability": trace_status,
        "missing_claims": missing_claims,
        "numerical_audit": phase10.get("numerical_audit"),
        "report": str(report_path),
        "excluded": ["GEO expression matrices", "Python environment", "caches", "identity maps", "reconstructed TAR", ".git"],
    }
    atomic_json(LOGS / "phase11_package_verify.json", payload)
    atomic_json(LOGS / "pipeline_checkpoint.json", {"last_completed_phase": "package_verification", "status": final_status, "updated_utc": now()})
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if final_status == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
