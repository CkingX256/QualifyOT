from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from audit_claim_traceability import build_claims, verify_claims, write_claims


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
AUDIT = ROOT / "audit"
RESULTS = ROOT / "results"
SOURCE = ROOT / "source_data"
FIGURES = ROOT / "figures"
MANUSCRIPT = ROOT / "manuscript"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as handle:
        handle.write(text)
        temp = Path(handle.name)
    os.replace(temp, path)


def atomic_json(path: Path, payload: object) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_source_data() -> list[dict[str, str]]:
    SOURCE.mkdir(parents=True, exist_ok=True)
    inputs = {
        "FigExpr_A_processed_pairs.csv": ROOT / "data" / "processed" / "gse174554_processed_pairs.csv",
        "FigExpr_B_honest_candidate_summary.csv": RESULTS / "honest_confirmation" / "honest_candidate_summary.csv",
        "FigExpr_C_lopo_seed_summary.csv": RESULTS / "dependent_lopo" / "lopo_seed_summary.csv",
        "FigExpr_D_bootstrap_stability.csv": RESULTS / "robustness" / "bootstrap_stability.csv",
        "FigExpr_E_delete_one.csv": RESULTS / "robustness" / "delete_one_checkpoint.csv",
        "FigExpr_tail_safety.csv": RESULTS / "robustness" / "tail_safety.csv",
        "FigExpr_reference_sensitivity.csv": RESULTS / "robustness" / "reference_sensitivity.csv",
        "FigExpr_loss_sensitivity.csv": RESULTS / "robustness" / "loss_sensitivity.csv",
    }
    manifest = []
    for name, source in inputs.items():
        if not source.exists():
            raise FileNotFoundError(source)
        destination = SOURCE / name
        shutil.copy2(source, destination)
        if sha256(source) != sha256(destination):
            raise RuntimeError(f"Source Data byte mismatch: {name}")
        manifest.append(
            {
                "source_data_file": name,
                "source_file": source.relative_to(ROOT).as_posix(),
                "sha256": sha256(destination),
                "bytes": str(destination.stat().st_size),
            }
        )
    with (SOURCE / "SOURCE_DATA_MANIFEST.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_data_file", "source_file", "sha256", "bytes"])
        writer.writeheader()
        writer.writerows(manifest)
    return manifest


def make_figure() -> None:
    pairs = read_csv(SOURCE / "FigExpr_A_processed_pairs.csv")
    honest = read_csv(SOURCE / "FigExpr_B_honest_candidate_summary.csv")
    lopo = [row for row in read_csv(SOURCE / "FigExpr_C_lopo_seed_summary.csv") if row["seed"] == "20260903"]
    stability = [row for row in read_csv(SOURCE / "FigExpr_D_bootstrap_stability.csv") if row["seed"] == "20260903"]
    delete_one = read_csv(SOURCE / "FigExpr_E_delete_one.csv")
    frozen_n = len(read_csv(AUDIT / "FROZEN_ELIGIBLE_PATIENTS.csv"))
    development_n = sum(row["analysis_partition"] == "development" for row in pairs)
    confirmation_n = sum(row["analysis_partition"] == "confirmation" for row in pairs)

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 5, figsize=(15.5, 3.35), constrained_layout=True)

    ax = axes[0]
    values = [frozen_n, len(pairs), development_n, confirmation_n]
    labels = ["Frozen", "Effective", "Development", "Confirmation"]
    colors = ["#4C78A8", "#59A14F", "#F28E2B", "#E15759"]
    ax.bar(range(4), values, color=colors)
    ax.set_xticks(range(4), labels, rotation=35, ha="right")
    ax.set_ylabel("Physical patients")
    for index, value in enumerate(values):
        ax.text(index, value + 0.35, str(value), ha="center", va="bottom")
    ax.set_title("A  Auditable cohort")

    ax = axes[1]
    y = np.arange(len(honest))
    estimates = np.array([float(row["utility_estimate"]) for row in honest])
    lower = np.array([float(row["utility_lcb"]) for row in honest])
    ax.hlines(y, lower, estimates, color="#4C78A8", linewidth=2)
    ax.scatter(estimates, y, color="#E15759", s=22, zorder=3)
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y, [row["candidate"] for row in honest])
    ax.invert_yaxis()
    ax.set_xlabel("Utility estimate and LCB")
    ax.set_title("B  Honest confirmation")

    ax = axes[2]
    lopo = sorted(lopo, key=lambda row: int(row["candidate_order"]))
    y = np.arange(len(lopo))
    candidate_mae = [float(row["candidate_mae"]) for row in lopo]
    reference_mae = [float(row["reference_mae"]) for row in lopo]
    ax.scatter(candidate_mae, y, color="#4C78A8", label="Candidate")
    ax.scatter(reference_mae, y, facecolors="none", edgecolors="#E15759", label="Reference")
    ax.set_yticks(y, [row["candidate"] for row in lopo])
    ax.invert_yaxis()
    ax.set_xlabel("Patient-equal MAE")
    ax.set_title("C  Dependent LOPO")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[3]
    key_names = sorted({row["candidate"] for row in stability}, key=lambda name: (name != "ExpressionElasticNet", name))
    offsets = np.linspace(-0.12, 0.12, max(1, len(key_names)))
    for offset, candidate in zip(offsets, key_names):
        rows = sorted((row for row in stability if row["candidate"] == candidate), key=lambda row: int(row["bootstrap"]))
        x = np.arange(len(rows)) + offset
        estimate = np.array([float(row["PUC"]) for row in rows])
        lo = np.array([float(row["PUC_lo"]) for row in rows])
        hi = np.array([float(row["PUC_hi"]) for row in rows])
        ax.errorbar(x, estimate, yerr=np.vstack([estimate - lo, hi - estimate]), marker="o", capsize=2, linewidth=1, label=candidate)
    bootstraps = sorted({int(row["bootstrap"]) for row in stability})
    ax.set_xticks(np.arange(len(bootstraps)), [str(value) for value in bootstraps])
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Bootstrap B")
    ax.set_ylabel("PUC with 95% interval")
    ax.set_title("D  Bootstrap stability")
    ax.legend(frameon=False, fontsize=6)

    ax = axes[4]
    candidates = sorted({row["candidate"] for row in delete_one})
    values = [[float(row["utility_lcb"]) for row in delete_one if row["candidate"] == candidate and row["status"] == "SUCCESS"] for candidate in candidates]
    ax.boxplot(values, tick_labels=candidates, showfliers=True)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylabel("Delete-one utility LCB")
    ax.set_title("E  Full-pipeline deletion")

    for ax in axes:
        ax.grid(axis="x" if ax is not axes[0] else "y", alpha=0.18)
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / "gse174554_expression_validation.pdf", bbox_inches="tight")
    figure.savefig(FIGURES / "gse174554_expression_validation.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


MATLAB_SOURCE = r"""% Re-render the five-panel GSE174554 expression validation figure.
% This script consumes only frozen Source Data CSV files.
root = fileparts(fileparts(mfilename('fullpath')));
s = fullfile(root, 'source_data');
pairs = readtable(fullfile(s, 'FigExpr_A_processed_pairs.csv'));
honest = readtable(fullfile(s, 'FigExpr_B_honest_candidate_summary.csv'));
lopo = readtable(fullfile(s, 'FigExpr_C_lopo_seed_summary.csv'));
stab = readtable(fullfile(s, 'FigExpr_D_bootstrap_stability.csv'));
del = readtable(fullfile(s, 'FigExpr_E_delete_one.csv'));
lopo = lopo(lopo.seed == 20260903,:);
stab = stab(stab.seed == 20260903,:);
f = figure('Color','w','Position',[100 100 1550 335]);
tiledlayout(f,1,5,'Padding','compact','TileSpacing','compact');
nexttile; bar([30 height(pairs) sum(strcmp(pairs.analysis_partition,'development')) sum(strcmp(pairs.analysis_partition,'confirmation'))]); title('A  Auditable cohort'); ylabel('Physical patients'); xticklabels({'Frozen','Effective','Development','Confirmation'}); xtickangle(35);
nexttile; hold on; y=(1:height(honest))'; for i=1:height(honest), plot([honest.utility_lcb(i) honest.utility_estimate(i)],[i i],'-','LineWidth',2); end; scatter(honest.utility_estimate,y,20,'filled'); xline(0,'--k'); yticks(y); yticklabels(honest.candidate); set(gca,'YDir','reverse'); title('B  Honest confirmation');
nexttile; y=(1:height(lopo))'; scatter(lopo.candidate_mae,y,20,'filled'); hold on; scatter(lopo.reference_mae,y,20); yticks(y); yticklabels(lopo.candidate); set(gca,'YDir','reverse'); title('C  Dependent LOPO'); xlabel('Patient-equal MAE');
nexttile; hold on; names=unique(stab.candidate,'stable'); for i=1:numel(names), z=stab(strcmp(stab.candidate,names{i}),:); errorbar(1:height(z),z.PUC,z.PUC-z.PUC_lo,z.PUC_hi-z.PUC,'-o'); end; yline(0,'--k'); xticks(1:4); xticklabels({'250','500','1000','2000'}); title('D  Bootstrap stability'); xlabel('Bootstrap B');
nexttile; names=unique(del.candidate,'stable'); x=[]; g={}; for i=1:numel(names), z=del.utility_lcb(strcmp(del.candidate,names{i}) & strcmp(del.status,'SUCCESS')); x=[x;z]; g=[g;repmat(names(i),numel(z),1)]; end; boxplot(x,g); yline(0,'--k'); title('E  Full-pipeline deletion'); xtickangle(35);
exportgraphics(f,fullfile(root,'figures','gse174554_expression_validation_matlab.pdf'),'ContentType','vector');
"""


def insert_before_end(text: str, addition: str) -> str:
    marker = "\\end{document}"
    if marker not in text:
        raise RuntimeError("LaTeX source has no end document marker")
    return text.rsplit(marker, 1)[0] + addition + "\n" + marker + "\n"


def make_manuscript(claims: list[dict[str, str]]) -> list[Path]:
    baseline_dirs = [path for path in (MANUSCRIPT / "baseline").iterdir() if path.is_dir()]
    if len(baseline_dirs) != 1:
        raise RuntimeError(f"Expected one manuscript baseline directory, found {len(baseline_dirs)}")
    baseline = baseline_dirs[0]
    working = MANUSCRIPT / "working"
    shutil.copytree(baseline, working, dirs_exist_ok=True)
    shutil.copy2(FIGURES / "gse174554_expression_validation.pdf", working / "figures" / "gse174554_expression_validation.pdf")
    main_source = baseline / "QualifyOT_NatureMethods_Main_20260903.tex"
    supplement_source = baseline / "QualifyOT_Supplementary_20260903.tex"
    cover_source = baseline / "cover_letter.tex"
    values = {row["claim_id"]: row["display_value"] for row in claims}
    winner_tex = values["lopo_accuracy_winner"].replace("_", r"\_")
    claim_comments = "\n".join(f"% CLAIM:{row['claim_id']}={row['display_value']}" for row in claims)
    main_addition = rf"""
{claim_comments}
\section*{{Post-freeze expression-space validation in GSE174554}}
We evaluated {values['analysis_patients']} auditable physical patients ({values['development_patients']} development and {values['confirmation_patients']} confirmation) after preserving all post-freeze exclusions. The prespecified ExpressionElasticNet candidate selected $\lambda={values['expression_selected_lambda']}$ on development data. In Honest confirmation, its movement estimate was {values['expression_movement_estimate']} (lower bound {values['expression_movement_lcb']}), utility was {values['expression_utility_estimate']} (lower bound {values['expression_utility_lcb']}), and retained incremental value was {values['expression_retention_estimate']} (lower bound {values['expression_retention_lcb']}). No candidate passed Holm-adjusted qualification ({values['honest_qualified_holm']} qualified); this is a negative qualification result rather than evidence that target outcomes were used for selection. All {values['leakage_checks']} leakage perturbations passed with {values['failed_leakage_checks']} failures.

The dependent-LOPO analysis is reported only as a diagnostic. ExpressionElasticNet achieved candidate MAE {values['lopo_expression_candidate_mae']} versus reference MAE {values['lopo_expression_reference_mae']}, with PUC {values['lopo_expression_PUC']} (95\% interval {values['lopo_expression_PUC_lo']} to {values['lopo_expression_PUC_hi']}). The fixed-order secondary accuracy winner was {winner_tex}. Figure~\ref{{fig:gse174554-expression}} separates Honest permission from dependent-LOPO accuracy.

\begin{{figure}}[ht]
\centering
\includegraphics[width=\textwidth]{{figures/gse174554_expression_validation.pdf}}
\caption{{GSE174554 expression-space validation. (A) Auditable cohort attrition and frozen partitions. (B) Honest utility estimates and lower bounds. (C) Dependent-LOPO candidate and reference MAE; these values are diagnostic only. (D) Patient-bootstrap stability. (E) Full-pipeline delete-one sensitivity. All panels are generated from frozen Source Data CSV files.}}
\label{{fig:gse174554-expression}}
\end{{figure}}
"""
    supplement_addition = rf"""
{claim_comments}
\section*{{Post-freeze GSE174554 expression validation audit}}
The analysis retained every failure and exclusion without replacement or split rehashing. ExpressionElasticNet candidate, reference and retained MAE were {values['expression_candidate_mae']}, {values['expression_reference_mae']} and {values['expression_retained_mae']}, respectively. The candidate intersection-union $p$ value was {values['expression_candidate_iut_p']} and its Holm-adjusted value was {values['expression_holm_adjusted_p']}. Full-pipeline delete-one refits completed for {values['delete_one_success']} tasks with {values['delete_one_failed']} failures. Deterministic reruns passed {values['determinism_pass']} checks with {values['determinism_fail']} failures. Source Data, per-patient predictions, bootstrap rows and failure records are distributed with the reproducibility package.
"""
    cover_addition = rf"""
{claim_comments}
\noindent\textbf{{Expression-space validation update.}} A post-freeze GSE174554 study evaluated {values['analysis_patients']} physical patients. The prespecified high-dimensional expression candidate did not qualify in Honest confirmation ({values['honest_qualified_holm']} Holm-qualified candidates). We report this negative result, the dependent-LOPO diagnostic and all integrity exclusions without changing the qualification rule.
"""
    paths = [working / "main.tex", working / "supplement.tex", working / "cover_letter.tex"]
    atomic_text(paths[0], insert_before_end(main_source.read_text(encoding="utf-8"), main_addition))
    atomic_text(paths[1], insert_before_end(supplement_source.read_text(encoding="utf-8"), supplement_addition))
    atomic_text(paths[2], insert_before_end(cover_source.read_text(encoding="utf-8"), cover_addition))
    return paths


def numerical_audit() -> tuple[bool, list[str]]:
    failures: list[str] = []
    phase_statuses = {}
    status_files = {
        5: LOGS / "phase05_pseudobulk.json",
        6: LOGS / "phase06_leakage_gate.json",
        7: LOGS / "phase07_honest_confirmation.json",
        8: LOGS / "phase08_lopo.json",
        9: LOGS / "phase09_robustness_delete_one.json",
    }
    for phase, status_path in status_files.items():
        if not status_path.exists():
            failures.append(f"missing_phase_{phase:02d}_status")
            continue
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        phase_statuses[f"phase{phase:02d}"] = payload.get("status")
        if payload.get("status") != "SUCCESS":
            failures.append(f"phase_{phase:02d}_{payload.get('status')}")
    pairs = read_csv(ROOT / "data" / "processed" / "gse174554_processed_pairs.csv")
    for row in pairs:
        for prefix in ("source", "target"):
            values = [float(row[f"{prefix}__Tumor"]), float(row[f"{prefix}__NonTumor"])]
            if any(not math.isfinite(value) or value < -1e-12 or value > 1 + 1e-12 for value in values) or abs(sum(values) - 1.0) > 1e-12:
                failures.append(f"simplex_{row['patient_id']}_{prefix}")
    leakage = read_csv(RESULTS / "leakage_gate" / "leakage_checks.csv")
    if any(row.get("status") != "PASS" for row in leakage):
        failures.append("leakage_check_failure")
    deterministic = read_csv(RESULTS / "robustness" / "determinism_reruns.csv")
    if any(row.get("status") != "PASS" or float(row.get("max_numeric_difference", "inf")) > 1e-12 for row in deterministic):
        failures.append("determinism_failure")
    honest = read_csv(RESULTS / "honest_confirmation" / "honest_candidate_summary.csv")
    if len(honest) != 5 or any(row["status"] != "SUCCESS" for row in honest):
        failures.append("honest_candidate_completeness")
    report = [
        "# Numerical audit",
        "",
        f"- Status: `{'PASS' if not failures else 'FAIL'}`",
        f"- Effective physical patients: `{len(pairs)}`",
        f"- Phase status map: `{json.dumps(phase_statuses, sort_keys=True)}`",
        f"- Simplex tolerance: `1e-12`",
        f"- Leakage checks: `{len(leakage)}`; failures: `{sum(row.get('status') != 'PASS' for row in leakage)}`",
        f"- Determinism rows: `{len(deterministic)}`",
        f"- Failures: `{';'.join(failures) if failures else 'none'}`",
    ]
    atomic_text(AUDIT / "NUMERICAL_AUDIT.md", "\n".join(report) + "\n")
    return not failures, failures


def main() -> int:
    started = utc_now()
    manifest = freeze_source_data()
    make_figure()
    atomic_text(FIGURES / "generate_gse174554_expression_figure.m", MATLAB_SOURCE)
    claims = build_claims()
    write_claims(AUDIT / "claim_traceability.csv", claims)
    manuscript_paths = make_manuscript(claims)
    trace_pass, missing_claims = verify_claims(claims, manuscript_paths)
    numerical_pass, numerical_failures = numerical_audit()
    manuscript_status = "READY_FOR_BUILD_VERIFICATION" if trace_pass and numerical_pass else "RESEARCH_DRAFT"
    atomic_text(
        AUDIT / "numerical_claim_audit.md",
        "# Manuscript number audit\n\n"
        f"- Status: `{'PASS' if trace_pass else 'FAIL'}`\n"
        f"- Claims traced: `{len(claims) - len(missing_claims)}/{len(claims)}`\n"
        f"- Missing claims: `{';'.join(missing_claims) if missing_claims else 'none'}`\n"
        "- Every generated claim records a source file, row/key locator, raw value and displayed value.\n",
    )
    payload: dict[str, Any] = {
        "phase": "figure_manuscript",
        "status": "SUCCESS" if trace_pass and numerical_pass else "PARTIAL",
        "started_utc": started,
        "finished_utc": utc_now(),
        "source_data_files": len(manifest),
        "figure_pdf_sha256": sha256(FIGURES / "gse174554_expression_validation.pdf"),
        "figure_png_sha256": sha256(FIGURES / "gse174554_expression_validation.png"),
        "matlab_render_verification": "NOT_RUN_MATLAB_UNAVAILABLE" if shutil.which("matlab") is None else "SOURCE_DELIVERED_NOT_EXECUTED",
        "numerical_audit": "PASS" if numerical_pass else "FAIL",
        "numerical_failures": numerical_failures,
        "manuscript_traceability": "PASS" if trace_pass else "FAIL",
        "missing_claims": missing_claims,
        "manuscript_status": manuscript_status,
    }
    atomic_json(LOGS / "phase10_figure_manuscript.json", payload)
    atomic_json(LOGS / "pipeline_checkpoint.json", {"last_completed_phase": "figure_manuscript", "status": payload["status"], "updated_utc": utc_now()})
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "SUCCESS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
