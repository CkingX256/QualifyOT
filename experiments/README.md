# Experiments

These scripts reproduce focused method-validation analyses. They are separate from the reusable library in `src/qualifyot/`.

- `synthetic_overcomplete_refinement.py` — finite-library recovery under known-true, null and adverse data-generating regimes.
- `synthetic_subgroup_refinement.py` — mechanism-mixture stress test and patient-level discordance analysis.
- `synthetic_tailaware_refinement.py` — exploratory tail-aware graph-selection stress test.
- `run_edge_enpi_map.py` — patient-level edge contribution analysis with simultaneous max-t intervals.
- `run_all_real.py` — compact real-cohort rerun for the fixed data-driven candidate.
- `audit_data.py` — simplex and patient-count checks for processed pair tables.

Run scripts from the repository root after installing the package in editable mode.

## Principled decision and safe-redesign validation

`validate_principled_decision.py` exercises the new Movement–Utility–Deployment Retention IUT decision, graph confidence sets, Safe Graph Averaging, tail-safety profile and predictive-assurance planner.

Quick smoke validation:

```bash
PYTHONPATH=src:experiments python experiments/validate_principled_decision.py --suite quick
```

Individual suites can be run with `--suite iut`, `graph-set`, `safe-average`, `tail`, `real-core` or `real-safe`. The real-safe suite is deliberately separated because fully nested training-only graph averaging is computationally heavier.

The executed results used for the current validation are archived in the output directory configured by the script.

## Robust-extension validation

`validate_robust_extensions.py` executes subgroup-tail-safe graph averaging, continuous edge shrinkage, and Bayesian-bootstrap graph-library profiling.

`benchmark_statistical_decisions.py` provides a transparent method benchmark using the public `qualifyot.benchmarks` API. `benchmark_statistical_decisions_fast.py` is a vectorized high-replicate calibration driver used for the executed 1,000-replicate-per-cell summary in `results/robust_extensions/statistical_benchmark_highrep_summary.csv`.

The vectorized driver implements the same estimands as the public benchmark functions but is used only to make high-replicate synthetic calibration tractable. Package-level unit tests separately verify the public API implementations.

## Transformer candidate diagnostics

`validate_fused_methodology.py` validates the single recommended `GraphResidualTransformerCandidate` against its nested GraphFlow baseline. Earlier exploratory Transformer variants are not included in the submission package.
