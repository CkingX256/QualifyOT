from __future__ import annotations

"""Hierarchical pseudoreplication stress test for many-cells/few-patients designs.

The simulation is exact for a Gaussian random-intercept data-generating process
without materializing individual cells.  For each physical patient i and cell j,

    Y_ij = theta + U_i + eps_ij,
    U_i ~ N(0, sigma_patient^2),
    eps_ij ~ N(0, sigma_cell^2),

with patients independent and cells conditionally independent within patient.

We compare a deliberately invalid naive cell-level one-sample t test (which
pretends all cells are independent) with a valid patient-level t test applied to
one mean per physical patient.  The sufficient-statistic construction is exact:
per-patient cell means are Gaussian and within-patient sums of squares are
independent scaled chi-square variables.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t


def _simulate_hierarchical_summary(
    *,
    rng: np.random.Generator,
    replicates: int,
    n_patients: int,
    cells_per_patient: int,
    theta: float,
    sigma_patient: float,
    sigma_cell: float,
) -> dict[str, np.ndarray]:
    if replicates < 1 or n_patients < 2 or cells_per_patient < 2:
        raise ValueError("replicates>=1, n_patients>=2 and cells_per_patient>=2 required")
    u = rng.normal(0.0, sigma_patient, size=(replicates, n_patients))
    eps_bar = rng.normal(
        0.0,
        sigma_cell / np.sqrt(float(cells_per_patient)),
        size=(replicates, n_patients),
    )
    patient_means = theta + u + eps_bar

    # Exact within-patient residual SS for normal errors, independent of eps_bar.
    within_ss = (sigma_cell**2) * rng.chisquare(
        df=cells_per_patient - 1,
        size=(replicates, n_patients),
    )

    grand_mean = patient_means.mean(axis=1)
    patient_sd = patient_means.std(axis=1, ddof=1)
    patient_se = patient_sd / np.sqrt(float(n_patients))
    patient_t = np.divide(
        grand_mean,
        patient_se,
        out=np.zeros_like(grand_mean),
        where=patient_se > 0,
    )

    total_n = n_patients * cells_per_patient
    between_ss = cells_per_patient * np.sum(
        (patient_means - grand_mean[:, None]) ** 2,
        axis=1,
    )
    total_ss = within_ss.sum(axis=1) + between_ss
    cell_sd = np.sqrt(total_ss / float(total_n - 1))
    naive_se = cell_sd / np.sqrt(float(total_n))
    naive_t = np.divide(
        grand_mean,
        naive_se,
        out=np.zeros_like(grand_mean),
        where=naive_se > 0,
    )

    return {
        "grand_mean": grand_mean,
        "patient_sd": patient_sd,
        "patient_t": patient_t,
        "naive_t": naive_t,
        "patient_se": patient_se,
        "naive_se": naive_se,
    }


def _null_cell_count_experiment(
    *,
    rng: np.random.Generator,
    replicates: int,
    n_patients: int,
    cell_counts: list[int],
    alpha: float,
    sigma_patient: float,
    sigma_cell: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    one_sided_patient_crit = student_t.ppf(1.0 - alpha, df=n_patients - 1)
    two_sided_patient_crit = student_t.ppf(1.0 - alpha / 2.0, df=n_patients - 1)
    patient_ci_crit = two_sided_patient_crit

    for m in cell_counts:
        sim = _simulate_hierarchical_summary(
            rng=rng,
            replicates=replicates,
            n_patients=n_patients,
            cells_per_patient=m,
            theta=0.0,
            sigma_patient=sigma_patient,
            sigma_cell=sigma_cell,
        )
        total_n = n_patients * m
        one_sided_naive_crit = student_t.ppf(1.0 - alpha, df=total_n - 1)
        two_sided_naive_crit = student_t.ppf(1.0 - alpha / 2.0, df=total_n - 1)
        naive_ci_crit = two_sided_naive_crit

        rows.append(
            {
                "n_patients": n_patients,
                "cells_per_patient": m,
                "total_cells": total_n,
                "replicates": replicates,
                "alpha": alpha,
                "sigma_patient": sigma_patient,
                "sigma_cell": sigma_cell,
                "intraclass_correlation": sigma_patient**2
                / (sigma_patient**2 + sigma_cell**2),
                "naive_cell_type1_one_sided": float(np.mean(sim["naive_t"] > one_sided_naive_crit)),
                "patient_type1_one_sided": float(np.mean(sim["patient_t"] > one_sided_patient_crit)),
                "naive_cell_type1_two_sided": float(np.mean(np.abs(sim["naive_t"]) > two_sided_naive_crit)),
                "patient_type1_two_sided": float(np.mean(np.abs(sim["patient_t"]) > two_sided_patient_crit)),
                "mean_naive_95ci_halfwidth": float(np.mean(naive_ci_crit * sim["naive_se"])),
                "mean_patient_95ci_halfwidth": float(np.mean(patient_ci_crit * sim["patient_se"])),
                "median_naive_abs_t": float(np.median(np.abs(sim["naive_t"]))),
                "median_patient_abs_t": float(np.median(np.abs(sim["patient_t"]))),
            }
        )
    return pd.DataFrame(rows)


def _patient_count_power_experiment(
    *,
    rng: np.random.Generator,
    replicates: int,
    patient_counts: list[int],
    cells_per_patient: int,
    theta: float,
    alpha: float,
    sigma_patient: float,
    sigma_cell: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    for n in patient_counts:
        sim = _simulate_hierarchical_summary(
            rng=rng,
            replicates=replicates,
            n_patients=n,
            cells_per_patient=cells_per_patient,
            theta=theta,
            sigma_patient=sigma_patient,
            sigma_cell=sigma_cell,
        )
        one_sided_crit = student_t.ppf(1.0 - alpha, df=n - 1)
        ci_crit = student_t.ppf(1.0 - alpha / 2.0, df=n - 1)
        rows.append(
            {
                "n_patients": n,
                "cells_per_patient": cells_per_patient,
                "replicates": replicates,
                "theta": theta,
                "alpha": alpha,
                "patient_one_sided_power": float(np.mean(sim["patient_t"] > one_sided_crit)),
                "mean_patient_95ci_halfwidth": float(np.mean(ci_crit * sim["patient_se"])),
                "mean_patient_estimate": float(np.mean(sim["grand_mean"])),
                "empirical_sd_patient_estimate": float(np.std(sim["grand_mean"], ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def _mechanical_duplication_experiment(
    *,
    rng: np.random.Generator,
    n_patients: int,
    cells_per_patient: int,
    duplicate_factors: list[int],
    theta: float,
    sigma_patient: float,
    sigma_cell: float,
) -> pd.DataFrame:
    # Materialize one modest data set so exact row replication is audited rather
    # than simulated by changing the data-generating process.
    u = rng.normal(0.0, sigma_patient, size=n_patients)
    eps = rng.normal(0.0, sigma_cell, size=(n_patients, cells_per_patient))
    y = theta + u[:, None] + eps
    patient_means = y.mean(axis=1)
    pmean = float(patient_means.mean())
    psd = float(patient_means.std(ddof=1))
    patient_t = pmean / (psd / np.sqrt(n_patients))

    rows: list[dict] = []
    base_naive_t = None
    for k in duplicate_factors:
        yk = np.repeat(y.reshape(-1), k)
        naive_mean = float(yk.mean())
        naive_sd = float(yk.std(ddof=1))
        naive_t = naive_mean / (naive_sd / np.sqrt(len(yk)))
        if base_naive_t is None:
            base_naive_t = naive_t
        rows.append(
            {
                "duplication_factor": k,
                "n_patients": n_patients,
                "original_cells_per_patient": cells_per_patient,
                "apparent_total_cell_rows": int(len(yk)),
                "patient_mean": pmean,
                "patient_t_stat": float(patient_t),
                "patient_t_shift_vs_k1": 0.0,
                "naive_cell_t_stat": float(naive_t),
                "naive_t_ratio_vs_k1": float(naive_t / base_naive_t) if base_naive_t else np.nan,
                "theoretical_sqrt_k": float(np.sqrt(k)),
            }
        )
    return pd.DataFrame(rows)


def _write_figures(null_df: pd.DataFrame, power_df: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        (out_dir / "PLOTTING_SKIPPED.txt").write_text(f"matplotlib unavailable: {exc}\n", encoding="utf-8")
        return

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(null_df["cells_per_patient"], null_df["naive_cell_type1_one_sided"], marker="o", label="Naive cell-level")
    ax.plot(null_df["cells_per_patient"], null_df["patient_type1_one_sided"], marker="o", label="Patient-level")
    ax.axhline(float(null_df["alpha"].iloc[0]), linestyle="--", linewidth=1, label="Nominal alpha")
    ax.set_xscale("log")
    ax.set_xlabel("Cells per physical patient")
    ax.set_ylabel("One-sided Type-I error")
    ax.set_title("More cells do not create independent patients")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "pseudoreplication_type1_error.png", dpi=220)
    fig.savefig(out_dir / "pseudoreplication_type1_error.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(power_df["n_patients"], power_df["patient_one_sided_power"], marker="o")
    ax.set_xlabel("Number of physical patients")
    ax.set_ylabel("Patient-level one-sided power")
    ax.set_ylim(0, 1.02)
    ax.set_title("Independent patients, not cell rows, drive inferential power")
    fig.tight_layout()
    fig.savefig(out_dir / "patient_count_power.png", dpi=220)
    fig.savefig(out_dir / "patient_count_power.pdf")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--replicates", type=int, default=50000)
    p.add_argument("--seed", type=int, default=20260902)
    p.add_argument("--n-patients", type=int, default=20)
    p.add_argument("--cell-counts", type=int, nargs="+", default=[10, 100, 1000, 10000, 100000])
    p.add_argument("--patient-counts", type=int, nargs="+", default=[10, 20, 40, 80])
    p.add_argument("--power-cells-per-patient", type=int, default=1000)
    p.add_argument("--power-theta", type=float, default=0.35)
    p.add_argument("--sigma-patient", type=float, default=1.0)
    p.add_argument("--sigma-cell", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--out", type=Path, default=Path("results/pseudoreplication"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    null_df = _null_cell_count_experiment(
        rng=rng,
        replicates=args.replicates,
        n_patients=args.n_patients,
        cell_counts=args.cell_counts,
        alpha=args.alpha,
        sigma_patient=args.sigma_patient,
        sigma_cell=args.sigma_cell,
    )
    power_df = _patient_count_power_experiment(
        rng=rng,
        replicates=args.replicates,
        patient_counts=args.patient_counts,
        cells_per_patient=args.power_cells_per_patient,
        theta=args.power_theta,
        alpha=args.alpha,
        sigma_patient=args.sigma_patient,
        sigma_cell=args.sigma_cell,
    )
    dup_df = _mechanical_duplication_experiment(
        rng=rng,
        n_patients=args.n_patients,
        cells_per_patient=100,
        duplicate_factors=[1, 10, 100, 1000],
        theta=0.25,
        sigma_patient=args.sigma_patient,
        sigma_cell=args.sigma_cell,
    )

    null_df.to_csv(args.out / "pseudoreplication_cell_count_null.csv", index=False)
    power_df.to_csv(args.out / "pseudoreplication_patient_count_power.csv", index=False)
    dup_df.to_csv(args.out / "pseudoreplication_mechanical_duplication.csv", index=False)

    summary = {
        "design": "Gaussian random-intercept exact sufficient-statistic simulation",
        "seed": args.seed,
        "replicates": args.replicates,
        "alpha": args.alpha,
        "n_patients_null": args.n_patients,
        "cell_counts": args.cell_counts,
        "sigma_patient": args.sigma_patient,
        "sigma_cell": args.sigma_cell,
        "power_theta": args.power_theta,
        "headline": {
            "naive_type1_at_min_cells": float(null_df.iloc[0]["naive_cell_type1_one_sided"]),
            "naive_type1_at_max_cells": float(null_df.iloc[-1]["naive_cell_type1_one_sided"]),
            "patient_type1_at_min_cells": float(null_df.iloc[0]["patient_type1_one_sided"]),
            "patient_type1_at_max_cells": float(null_df.iloc[-1]["patient_type1_one_sided"]),
            "patient_power_at_min_n": float(power_df.iloc[0]["patient_one_sided_power"]),
            "patient_power_at_max_n": float(power_df.iloc[-1]["patient_one_sided_power"]),
        },
        "interpretation_guardrail": (
            "This simulation demonstrates pseudoreplication under a specified hierarchical Gaussian DGP. "
            "It is not a claim that all cell-level analyses are invalid; cell-level models require methods that "
            "account for within-patient dependence when patient-level scientific claims are made."
        ),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    _write_figures(null_df, power_df, args.out)

    readme = f"""# Pseudoreplication stress test\n\nExecuted with seed `{args.seed}` and `{args.replicates:,}` Monte Carlo replicates per design cell.\n\nThe data-generating process is `Y_ij = theta + U_i + eps_ij`, with independent physical patients, `U_i ~ N(0,{args.sigma_patient**2:g})`, and within-patient cell noise `eps_ij ~ N(0,{args.sigma_cell**2:g})`. The simulation uses the exact Gaussian sufficient-statistic distributions for patient means and within-patient sums of squares, so very large cell counts are evaluated without materializing artificial cell matrices.\n\nPrimary null comparison: one-sided alpha={args.alpha:g}, `{args.n_patients}` physical patients, cells/patient = {args.cell_counts}. A naive cell-level t test is intentionally misspecified by treating all cells as independent; the patient-level test uses one value per physical patient.\n\nThe mechanical-duplication audit exactly repeats existing cell rows. The patient-level statistic is invariant by construction; the naive statistic is not.\n\nInterpretation: this is a controlled demonstration of the inferential distinction between representation-level information from many cells and independent patient-level evidence. It does **not** imply that every cell-level model is invalid; hierarchical/cluster-aware procedures can be valid for appropriately defined estimands.\n"""
    (args.out / "README.md").write_text(readme, encoding="utf-8")

    print(null_df.to_string(index=False))
    print("\nPatient-count power:\n", power_df.to_string(index=False))
    print("\nMechanical duplication:\n", dup_df.to_string(index=False))


if __name__ == "__main__":
    main()
