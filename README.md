# QualifyOT

QualifyOT is a patient-level evidence-control framework for longitudinal single-cell prediction. It evaluates whether a frozen candidate predictor has enough independent-patient evidence to retain non-zero predictive influence under a prespecified analysis contract.

The primary decision has three necessary components:

1. **Movement** — the candidate changes prediction beyond a prespecified scientifically relevant margin.
2. **Utility** — the candidate lowers patient-equal risk relative to the frozen reference.
3. **Deployment Retention** — the deployment weight selected from development data remains beneficial when evaluated on confirmation patients.

Qualification is an intersection-union decision: all three claims must be supported. Candidate families are handled by candidate-level IUT p-values followed by Holm correction. Robustness, tail-risk and sensitivity analyses are reported separately and do not replace the primary decision.

## Statistical unit

The physical patient is the inferential unit. Cells or rows within a patient can improve measurement precision, but they do not create additional independent biological replicates. Patient normalization is therefore propagated through fitting and inferential summaries.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
# activate the environment, then
python -m pip install --upgrade pip
pip install -e ".[test]"
```

For the optional Transformer candidate:

```bash
pip install -e ".[test,transformer]"
```

Run the core regression suite with:

```bash
pytest -q
```

The core suite passes **128 tests**. The standalone GSE174554 expression-validation module has its own full regression suite:

```bash
cd validation/gse174554_expression
PYTHONPATH=src pytest -q
```

That module passes **136 tests**, including its extension-specific leakage, split and traceability checks.

## Minimal API example

```python
import pandas as pd
from qualifyot import PipelineConfig, QualifyOTPipeline

pairs = pd.read_csv("data/processed_pairs/GSE123813_pairs.csv")

pipeline = QualifyOTPipeline(
    states=["Memory", "Effector", "Exhausted", "Regulatory", "Other_T"],
    edges=[
        ("Memory", "Effector"),
        ("Effector", "Exhausted"),
        ("Memory", "Exhausted"),
    ],
    candidate="graphflow",
    config=PipelineConfig(
        bootstrap=5000,
        seed=20260831,
        reference_rule="nested",
    ),
)

result = pipeline.run(pairs)
print(result["core_evidence_state"])
print(result["core_axes"])
```

The inferential Movement contrast is

\[
G_M=A-\delta_M B,
\]

while the ratio \(M=A/B\) remains a descriptive effect-size summary when \(B>0\).

## Honest Confirmation

The inference layer distinguishes two finite-sample targets:

- **Cohort-conditional Honest Confirmation** conditions on the realized target-blind confirmation inputs and uses prediction-localized score ranges.
- **Population Honest Confirmation** uses covariate-independent global support and targets the unconditional patient-population mean.

These are different estimands and should be declared before confirmation outcomes are interpreted. See `docs/INFERENCE_THEORY.md`.

## GSE174554 source-expression validation

The high-dimensional expression validation is in `validation/gse174554_expression/`. It contains the frozen patient split, processed primary-source expression used for candidate construction, extension code, leakage and determinism audits, machine-readable results and figure source data. Recurrence expression is not used to construct the predictor.

The extension package is also installed from `src/qualifyot_gse174554/`, and its regression tests are included in the main test suite.

## Robustness analyses

The `validation/` directory contains the numerical analyses used to examine the operating boundaries of the evidence decision:

- `statistical_inference/` — cohort-conditional versus population Honest inference, Movement-margin sensitivity and Utility-only comparison;
- `theory/` — Retention-selection, patient-number and contract-shopping checks;
- `foundational_inputs/` — sensitivity to the scalarization coefficient and ontology coarsening;
- `split_and_nonlinear/` — repeated frozen-split and nonlinear source-expression capacity stress tests;
- `gse174554_expression/` — target-sealed source-expression validation and leakage audits.

Recorded random seeds and frozen contract salts are intentionally retained because they are part of the reproducibility record.

## Repository structure

- `src/qualifyot/` — evidence engine and candidate interfaces.
- `src/qualifyot_gse174554/` — GSE174554 expression-validation extension.
- `tests/` — integrated regression, leakage, determinism, contract and invariance tests.
- `experiments/` — simulation and real-data analysis scripts.
- `validation/` — statistical, theoretical and foundational-input robustness analyses.
- `configs/` — analysis configurations used by the executable studies.
- `data/` — redistributable processed or compact source-derived inputs.
- `provenance/` — data provenance, numerical checks and claim-to-output traceability.
- `docs/` — statistical definitions, input format and reproducibility notes.

## Data

The analyzed datasets are public. Accession identifiers and public-source information are summarized in `provenance/data_provenance.md`. Large third-party source datasets are not redistributed when unnecessary or restricted.

## Interpretation boundary

QualifyOT is predictive and contract-relative. A Qualified result does not establish causal lineage, biological mechanism, clinical benefit, individual-patient safety or universal transportability. Claims remain relative to the declared patient population, prediction task, eligibility rules, preprocessing, candidate family, reference, ontology, loss, Movement margin, deployment-weight set, split rule and inference rule.

## Citation

Citation metadata are provided in `CITATION.cff`. The manuscript citation can be added after publication.

## Repository

https://github.com/CkingX256/QualifyOT

## License

No software license is assigned in this repository. Reuse terms should be added only after agreement by all authors.
