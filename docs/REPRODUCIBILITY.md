# Reproducibility

The repository separates three tasks:

1. **Qualification**: frozen patient-level evidence evaluation.
2. **Refinement**: training-only graph selection that never uses the outer held-out target for graph choice.
3. **Diagnostics**: exploratory patient- and edge-level summaries that do not overwrite the locked evidence state.

Run the automated test suite before reproducing analyses:

```bash
pytest -q
```

The compact result tables in `results/` record the executed structural-recovery, mechanism-mixture, edge-ENPI, real-cohort refinement and clinical-eligibility analyses used to audit the method's operating boundaries.

The scripts in `experiments/` are research scripts, whereas `src/qualifyot/` is the reusable software library.
