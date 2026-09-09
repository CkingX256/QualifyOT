from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT
FIG = ROOT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Experiment 1: development-only retention selection consistency under the same
# two-state simplex MAE used in the manuscript.  A target p in [0,1] represents
# the first simplex coordinate; the second is 1-p, so K=2 MAE equals |p_hat-p|.
# Reference and candidate errors are negatively correlated, so the population
# risk has an interior optimum; candidate alone is also better than reference.
# -----------------------------------------------------------------------------
SEED_POP = 20260909
SEED_RET = 20260910
rng = np.random.default_rng(SEED_POP)
M = 1_000_000
sd_ref, sd_cand, rho = 0.24, 0.16, -0.70
grid = np.round(np.linspace(0.0, 1.0, 11), 10)

target = rng.beta(2.0, 2.0, size=M)
z1 = rng.normal(size=M)
z2 = rng.normal(size=M)
ref = np.clip(target + sd_ref * z1, 0.0, 1.0)
cand = np.clip(target + sd_cand * (rho * z1 + math.sqrt(1-rho**2) * z2), 0.0, 1.0)
pop_risk = np.array([np.mean(np.abs((1-lam) * ref + lam * cand - target)) for lam in grid])
idx_star = int(np.argmin(pop_risk))
lambda_star = float(grid[idx_star])
min_risk = float(pop_risk[idx_star])
second_risk = float(np.partition(pop_risk, 1)[1])
gap = second_risk - min_risk

pd.DataFrame({"lambda": grid, "population_mae": pop_risk}).to_csv(OUT / "retention_population_risk.csv", index=False)

rng = np.random.default_rng(SEED_RET)
ret_rows = []
for n_dev in [10, 20, 40, 80, 160, 320]:
    reps = 10_000
    target = rng.beta(2.0, 2.0, size=(reps, n_dev))
    z1 = rng.normal(size=(reps, n_dev))
    z2 = rng.normal(size=(reps, n_dev))
    ref = np.clip(target + sd_ref * z1, 0.0, 1.0)
    cand = np.clip(target + sd_cand * (rho * z1 + math.sqrt(1-rho**2) * z2), 0.0, 1.0)
    empirical_risk = np.stack(
        [np.mean(np.abs((1-lam) * ref + lam * cand - target), axis=1) for lam in grid],
        axis=1,
    )
    chosen_idx = np.argmin(empirical_risk, axis=1)
    chosen = grid[chosen_idx]
    excess = pop_risk[chosen_idx] - min_risk
    ret_rows.append({
        "n_development": n_dev,
        "replicates": reps,
        "p_select_population_grid_optimum": float(np.mean(chosen_idx == idx_star)),
        "mean_selected_lambda": float(np.mean(chosen)),
        "p_select_nonzero_lambda": float(np.mean(chosen > 0)),
        "mean_population_excess_mae": float(np.mean(excess)),
        "q95_population_excess_mae": float(np.quantile(excess, 0.95)),
    })
ret_df = pd.DataFrame(ret_rows)
ret_df.to_csv(OUT / "retention_selection_consistency.csv", index=False)

fig, ax = plt.subplots(figsize=(5.1, 3.4))
ax.plot(ret_df["n_development"], ret_df["p_select_population_grid_optimum"], marker="o")
ax.set_xscale("log", base=2)
ax.set_ylim(0, 1.03)
ax.set_xlabel("Development patients")
ax.set_ylabel(r"$P(\hat\lambda=\lambda^*_{\rm grid})$")
ax.set_title("Independent development-validation selection concentrates")
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "supp_retention_selection_consistency.pdf", bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.1, 3.4))
ax.plot(ret_df["n_development"], ret_df["mean_population_excess_mae"], marker="o")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("Development patients")
ax.set_ylabel("Mean population excess MAE")
ax.set_title("Excess risk of selected retention weight")
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "supp_retention_excess_risk.pdf", bbox_inches="tight")
plt.close(fig)

# -----------------------------------------------------------------------------
# Experiment 2: validate the patient-number planning corollary for Honest
# Confirmation. Patient scores are bounded in an interval of width w and use
# the exact Hoeffding LCB in the manuscript. Three necessary axes are simulated.
# The Rademacher construction is intentionally high-variance within the bound.
# -----------------------------------------------------------------------------
SEED_POWER = 20260911
SEED_NULL = 20260912
alpha = 0.05
beta = 0.20
theta = 0.15
width = 0.40
n_bound = math.ceil(
    (width**2 / (2 * theta**2))
    * (math.sqrt(math.log(1 / alpha)) + math.sqrt(math.log(3 / beta))) ** 2
)

rng = np.random.default_rng(SEED_POWER)
power_rows = []
for n_conf in [10, 15, 20, 25, 30, 35, n_bound, 50, 60, 80, 100]:
    reps = 100_000
    eps = rng.choice([-width / 2, width / 2], size=(reps, n_conf, 3))
    scores = theta + eps
    means = scores.mean(axis=1)
    penalty = width * math.sqrt(math.log(1 / alpha) / (2 * n_conf))
    qualified = np.all(means - penalty > 0, axis=1)
    power_rows.append({
        "n_confirmation": n_conf,
        "replicates": reps,
        "effect_each_axis": theta,
        "interval_width": width,
        "empirical_qualification_power": float(np.mean(qualified)),
        "planned_n_bound": n_bound,
    })
power_df = pd.DataFrame(power_rows).drop_duplicates(subset=["n_confirmation"]).sort_values("n_confirmation")
power_df.to_csv(OUT / "honest_confirmation_sample_size_power.csv", index=False)

rng = np.random.default_rng(SEED_NULL)
null_rows = []
for n_conf in [10, 20, 30, n_bound, 60, 100]:
    reps = 200_000
    theta_vec = np.array([0.0, theta, theta])
    eps = rng.choice([-width / 2, width / 2], size=(reps, n_conf, 3))
    scores = theta_vec + eps
    means = scores.mean(axis=1)
    penalty = width * math.sqrt(math.log(1 / alpha) / (2 * n_conf))
    qualified = np.all(means - penalty > 0, axis=1)
    null_rows.append({
        "n_confirmation": n_conf,
        "replicates": reps,
        "null_axis_mean": 0.0,
        "other_axis_mean": theta,
        "interval_width": width,
        "false_qualification_rate": float(np.mean(qualified)),
    })
null_df = pd.DataFrame(null_rows)
null_df.to_csv(OUT / "honest_confirmation_sample_size_null.csv", index=False)

fig, ax = plt.subplots(figsize=(5.1, 3.4))
ax.plot(power_df["n_confirmation"], power_df["empirical_qualification_power"], marker="o", label="Empirical power")
ax.axhline(1-beta, linestyle="--", linewidth=1, label="Target 0.80")
ax.axvline(n_bound, linestyle=":", linewidth=1, label=f"Sufficient bound n={n_bound}")
ax.set_ylim(0, 1.03)
ax.set_xlabel("Independent confirmation patients")
ax.set_ylabel("Qualification probability")
ax.set_title("Honest-Confirmation planning bound")
ax.legend(frameon=False, fontsize=8)
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "supp_honest_sample_size_power.pdf", bbox_inches="tight")
plt.close(fig)

# -----------------------------------------------------------------------------
# Experiment 3: exact independent-contract shopping curve. This is analytic,
# with a small Monte Carlo check at K=27 to tie the formula to the manuscript's
# existing 0.7479 stress result without changing that frozen result.
# -----------------------------------------------------------------------------
contract_rows = []
for K in [1, 2, 5, 10, 20, 27, 50]:
    analytic = 1 - (1-alpha)**K
    contract_rows.append({"n_independent_contracts": K, "analytic_any_false_qualification": analytic})
contract_df = pd.DataFrame(contract_rows)
contract_df.to_csv(OUT / "contract_shopping_analytic_curve.csv", index=False)

rng = np.random.default_rng(20260913)
reps = 1_000_000
K = 27
mc_any = np.any(rng.random((reps, K)) < alpha, axis=1)
mc_rate = float(np.mean(mc_any))
analytic_27 = float(1-(1-alpha)**K)

# -----------------------------------------------------------------------------
# Audit summary
# -----------------------------------------------------------------------------
summary = "# Theory-validation experiment report\n\n"
summary += f"Retention experiment: two-state simplex MAE, grid {{0,0.1,...,1}}, population Monte Carlo n={M:,}. The population grid optimum was lambda*={lambda_star:.1f}; reference MAE={pop_risk[0]:.6f}, candidate MAE={pop_risk[-1]:.6f}, retained optimum MAE={min_risk:.6f}, nearest-grid risk gap={gap:.6f}. Selection used 10,000 independent development datasets per sample size.\n\n"
summary += "Retention selection results:\n\n" + ret_df.to_markdown(index=False) + "\n\n"
summary += f"Honest-confirmation planning experiment: alpha={alpha}, target overall power >= {1-beta:.2f}, three necessary axes each with mean theta={theta} and score width w={width}. The sufficient Hoeffding/union-bound formula gives n >= {n_bound}. At n={n_bound}, empirical qualification power was {float(power_df.loc[power_df.n_confirmation==n_bound,'empirical_qualification_power'].iloc[0]):.5f}. Under a least-favourable one-boundary-axis null, the empirical false-Qualification rate at n={n_bound} was {float(null_df.loc[null_df.n_confirmation==n_bound,'false_qualification_rate'].iloc[0]):.5f}.\n\n"
summary += "Power results:\n\n" + power_df.to_markdown(index=False) + "\n\n"
summary += "Boundary-null results:\n\n" + null_df.to_markdown(index=False) + "\n\n"
summary += f"Independent contract-shopping identity check: for K=27 and alpha=0.05, analytic 1-(1-alpha)^K={analytic_27:.6f}; Monte Carlo with {reps:,} replicates gave {mc_rate:.6f}. The previously frozen manuscript simulation reported 0.7479; this new check is not used to replace the frozen value.\n"
(OUT / "THEORY_VALIDATION_REPORT.md").write_text(summary, encoding="utf-8")

print(summary)
