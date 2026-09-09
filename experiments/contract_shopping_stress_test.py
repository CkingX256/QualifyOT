from __future__ import annotations

"""Controlled researcher-degrees-of-freedom stress test.

This experiment isolates one statistical consequence of post-outcome analysis-
contract shopping.  Under the least-favourable IUT null geometry, one necessary
axis is exactly on its null boundary while the other necessary axes are assumed
well supported.  Therefore a false Qualification occurs exactly when the
binding one-sided component test rejects.

We simulate K alternative contracts (e.g. reference x loss x ontology choices)
whose binding-axis z statistics are equicorrelated.  A frozen contract uses only
the first test.  A post-hoc researcher reports a Qualification if *any* contract
rejects at alpha.  As a calibration control, the family-aware rule requires
min(p) <= alpha/K, which is the first Holm step under the complete contract-family
null.

The experiment is not a claim that the real QualifyOT contracts are independent,
or that every alternative contract is scientifically exchangeable.  It is a
transparent stress test showing why changing a frozen estimand after seeing its
result creates an additional multiplicity problem even when each individual IUT
is valid at level alpha.
"""

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from scipy.stats import norm

SEED = 20260902
ALPHA = 0.05
REPLICATES = 200_000
KS = (1, 3, 9, 27)
RHOS = (0.0, 0.5, 0.9)
CHUNK = 20_000


def simulate(k: int, rho: float, *, rng: np.random.Generator) -> dict:
    zcrit = float(norm.ppf(1.0 - ALPHA))
    n_done = 0
    frozen_rej = 0
    any_uncorrected = 0
    family_aware = 0
    threshold_family = float(norm.ppf(1.0 - ALPHA / k))

    while n_done < REPLICATES:
        n = min(CHUNK, REPLICATES - n_done)
        common = rng.normal(size=(n, 1))
        eps = rng.normal(size=(n, k))
        z = math.sqrt(rho) * common + math.sqrt(1.0 - rho) * eps
        frozen_rej += int(np.count_nonzero(z[:, 0] > zcrit))
        max_z = z.max(axis=1)
        any_uncorrected += int(np.count_nonzero(max_z > zcrit))
        family_aware += int(np.count_nonzero(max_z > threshold_family))
        n_done += n

    independent_exact = 1.0 - (1.0 - ALPHA) ** k if rho == 0.0 else np.nan
    return {
        "contracts_K": k,
        "equicorrelation_rho": rho,
        "replicates": REPLICATES,
        "alpha": ALPHA,
        "frozen_contract_false_qualification": frozen_rej / REPLICATES,
        "posthoc_any_contract_false_qualification": any_uncorrected / REPLICATES,
        "family_aware_holm_first_step_false_qualification": family_aware / REPLICATES,
        "independent_exact_any_false_qualification": independent_exact,
        "inflation_vs_frozen": (any_uncorrected / REPLICATES) / (frozen_rej / REPLICATES),
    }


def main() -> None:
    outdir = Path("results/contract_shopping")
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows = [simulate(k, rho, rng=rng) for rho in RHOS for k in KS]
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "contract_shopping_null.csv", index=False)

    max_row = df.loc[df["posthoc_any_contract_false_qualification"].idxmax()].to_dict()
    summary = {
        "design": "least-favourable IUT boundary-null contract-shopping stress test",
        "seed": SEED,
        "replicates_per_cell": REPLICATES,
        "alpha": ALPHA,
        "contract_counts": list(KS),
        "equicorrelations": list(RHOS),
        "interpretation": (
            "A frozen contract retains the nominal one-sided level. Post-outcome selection over "
            "multiple individually valid contracts inflates the probability of at least one false "
            "Qualification. Treating all alternatives as a pre-specified family and applying the "
            "first Holm/Bonferroni threshold controls this complete-null stress test, but changes of "
            "scientific estimand still require transparent relocking rather than retrospective relabeling."
        ),
        "maximum_observed_posthoc_false_qualification": max_row,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (outdir / "README.md").write_text(
        "# Analysis-contract shopping stress test\n\n"
        f"Seed `{SEED}`; `{REPLICATES:,}` Monte Carlo replicates per design cell; one-sided alpha `{ALPHA}`.\n\n"
        "The binding IUT component is on its null boundary and the other necessary components are "
        "assumed supported. K alternative contracts have equicorrelated standard-normal binding-axis "
        "statistics. `frozen_contract_false_qualification` uses one pre-specified contract; "
        "`posthoc_any_contract_false_qualification` reports success if any unadjusted contract passes; "
        "`family_aware_holm_first_step_false_qualification` uses alpha/K, the first Holm threshold under "
        "the complete family null. This is a controlled statistical stress test, not an empirical estimate "
        "of investigator behaviour in the real cohorts.\n",
        encoding="utf-8",
    )
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
