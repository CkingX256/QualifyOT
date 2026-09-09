from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
AUDIT = ROOT / "audit"
LOGS = ROOT / "logs"
CODE_BASE = ROOT / "code_base"
MANUSCRIPT_BASE = ROOT / "manuscript" / "baseline"
PYTHON = PROJECT / "qualifyot_scientific_upgrade" / ".conda-env" / "python.exe"

INPUT_NAMES = (
    "GSE174554_RAW.tar",
    "GSE174554_RAW.part1.rar",
    "GSE174554_RAW.part2.rar",
    "GSE174554_RAW.part3.rar",
    "GSE174554_RAW.part4.rar",
    "GSE174554_Tumor_normal_metadata.txt.gz",
    "43018_2022_475_MOESM2_ESM.xlsx",
    "QualifyOT_code_baseline.zip",
    "QualifyOT_manuscript_baseline.zip",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", delete=False, dir=path.parent
    ) as handle:
        handle.write(text)
        tmp = Path(handle.name)
    os.replace(tmp, path)


def atomic_json(path: Path, payload: object) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(zip_path: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / ".extracted_from.json"
    zip_hash = sha256_file(zip_path)
    if marker.exists():
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if existing.get("sha256") == zip_hash:
            roots = [p for p in destination.iterdir() if p.is_dir()]
            return roots[0] if len(roots) == 1 else destination
        raise RuntimeError(f"Extraction destination already contains a different archive: {destination}")
    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"CRC failure in {zip_path.name}: {bad}")
        base = destination.resolve()
        for member in archive.infolist():
            candidate = (destination / member.filename).resolve()
            if base not in candidate.parents and candidate != base:
                raise RuntimeError(f"Unsafe ZIP member: {member.filename}")
        archive.extractall(destination)
        names = [m.filename for m in archive.infolist()]
    atomic_json(marker, {"archive": str(zip_path), "sha256": zip_hash, "entries": len(names), "extracted_utc": utc_now()})
    roots = [p for p in destination.iterdir() if p.is_dir()]
    return roots[0] if len(roots) == 1 else destination


def copy_working_baseline(extracted_root: Path) -> None:
    for name in ("src", "tests", "configs"):
        src = extracted_root / name
        dst = ROOT / name
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)
    for name in ("pyproject.toml", "pytest.ini", "requirements.txt"):
        src = extracted_root / name
        if src.exists():
            shutil.copy2(src, ROOT / name)


def run_baseline_tests(extracted_root: Path) -> dict[str, object]:
    if not PYTHON.exists():
        raise FileNotFoundError(f"Pinned project Python not found: {PYTHON}")
    env = os.environ.copy()
    env.update(
        {
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONPATH": str(extracted_root / "src"),
        }
    )
    command = [str(PYTHON), "-m", "pytest", "-q"]
    started = time.time()
    completed = subprocess.run(
        command,
        cwd=extracted_root,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.time() - started
    atomic_text(LOGS / "baseline_pytest_raw.txt", completed.stdout)
    return {
        "command": command,
        "cwd": str(extracted_root),
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "output_file": str(LOGS / "baseline_pytest_raw.txt"),
        "started_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(timespec="seconds"),
        "finished_utc": utc_now(),
    }


def main() -> int:
    started = utc_now()
    missing = [name for name in INPUT_NAMES if not (PROJECT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required input(s): {missing}")

    manifest: list[dict[str, object]] = []
    for name in INPUT_NAMES:
        path = PROJECT / name
        stat = path.stat()
        manifest.append(
            {
                "logical_role": (
                    "immutable_code_baseline" if name.startswith("QualifyOT_REPRODUCIBILITY")
                    else "immutable_manuscript_baseline" if name.startswith("QualifyOT_NatureMethods")
                    else "immutable_scientific_input"
                ),
                "absolute_path": str(path.resolve()),
                "filename": path.name,
                "size_bytes": stat.st_size,
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
                "sha256": sha256_file(path),
            }
        )

    csv_path = AUDIT / "INPUT_FILE_MANIFEST.csv"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=AUDIT) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
        tmp_csv = Path(handle.name)
    os.replace(tmp_csv, csv_path)
    atomic_text(AUDIT / "INPUT_SHA256SUMS.txt", "".join(f"{row['sha256']}  {row['filename']}\n" for row in manifest))

    code_zip = PROJECT / "QualifyOT_code_baseline.zip"
    manuscript_zip = PROJECT / "QualifyOT_manuscript_baseline.zip"
    extracted_code = safe_extract(code_zip, CODE_BASE)
    extracted_manuscript = safe_extract(manuscript_zip, MANUSCRIPT_BASE)
    copy_working_baseline(extracted_code)

    env_lines = [
        f"captured_utc={utc_now()}",
        f"os={platform.platform()}",
        f"machine={platform.machine()}",
        f"processor={platform.processor()}",
        f"python={PYTHON}",
    ]
    version = subprocess.run([str(PYTHON), "--version"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    env_lines.append(f"python_version={version.stdout.strip()}")
    packages = subprocess.run(
        [str(PYTHON), "-m", "pip", "freeze", "--all"],
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    env_lines.extend(["", "[pip-freeze]", packages.stdout.rstrip()])
    atomic_text(AUDIT / "ENVIRONMENT_REPORT.txt", "\n".join(env_lines) + "\n")

    test_record = run_baseline_tests(extracted_code)
    status = "SUCCESS" if test_record["returncode"] == 0 else "FAILED"
    phase = {
        "schema_version": 1,
        "experiment": "QualifyOT_GSE174554_expression-v1",
        "phase": "provenance_baseline",
        "status": status,
        "started_utc": started,
        "finished_utc": utc_now(),
        "target_metadata_opened": False,
        "input_manifest": str(csv_path),
        "code_baseline_root": str(extracted_code),
        "manuscript_baseline_root": str(extracted_manuscript),
        "baseline_test": test_record,
    }
    atomic_json(LOGS / "phase01_provenance_baseline.json", phase)
    atomic_json(LOGS / "pipeline_checkpoint.json", {"last_completed_phase": "provenance_baseline" if status == "SUCCESS" else None, "status": status, "updated_utc": utc_now()})
    report = [
        "# Stage 1: provenance and baseline gate",
        "",
        f"- Status: `{status}`",
        f"- Started (UTC): `{started}`",
        f"- Completed (UTC): `{phase['finished_utc']}`",
        f"- Immutable inputs registered: `{len(manifest)}`",
        f"- Code baseline: `{code_zip.name}`",
        f"- Manuscript baseline: `{manuscript_zip.name}`",
        f"- Baseline pytest return code: `{test_record['returncode']}`",
        f"- Target metadata opened: `false`",
        "",
        "The scientific target metadata was not read during this phase.",
    ]
    atomic_text(AUDIT / "STAGE01_PROVENANCE_BASELINE_REPORT.md", "\n".join(report) + "\n")
    print(json.dumps(phase, ensure_ascii=False, indent=2))
    return 0 if status == "SUCCESS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
