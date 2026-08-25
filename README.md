# QualifyOT

**QualifyOT** is a patient-level framework for qualifying structured biological priors in longitudinal compositional prediction. It separates candidate deviation, direct predictive utility, conservative retention, net benefit, perturbation stability and transportability. A training-only refinement layer can compare a finite library of graph structures without using the outer held-out target for graph selection.

## What is included

- `src/qualifyot/` — reusable Python package and command-line interface.
- `tests/` — automated regression and leakage checks.
- `examples/` — tutorial and external-prediction adapter example.
- `data/processed_pairs/` — compact patient-level composition tables.
- `data/source_small/` — selected public processed metadata used by adapters.
- `experiments/` — reproducible research scripts for structural recovery, subgroup stress tests and edge-level analysis.
- `results/` — compact executed result summaries.
- `docs/` — input format, public-data scope and reproducibility notes.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[test]"
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[test]"
```

Verify the installation:

```bash
pytest -q
```

## Five-minute diagnostic

Validate a standardized pair table:

```bash
qualifyot-inspect validate-csv data/processed_pairs/GSE123813_pairs.csv \
  --states Memory,Effector,Exhausted,Regulatory,Other_T
```

Run a rapid exploratory GraphFlow screen:

```bash
qualifyot-inspect quicktest data/processed_pairs/GSE123813_pairs.csv \
  --states Memory,Effector,Exhausted,Regulatory,Other_T \
  --graph "Memory->Effector,Effector->Exhausted,Memory->Exhausted" \
  --bootstrap 300 \
  --out examples/gse123813_quicktest.html
```

The HTML report contains a provisional evidence state, limiting gates and patient-level retained-NPI contributions. `quicktest` is an exploratory convenience command; it does not replace the locked analysis or complete-pipeline perturbation assessment.

## Saved-metric classification

```bash
qualifyot-inspect classify examples/gse123813_metrics.json
qualifyot-inspect report examples/gse123813_metrics.json \
  --out-prefix examples/gse123813_report
```

## Python API

```python
import pandas as pd
from qualifyot.candidate_api import GraphFlowCandidate
from qualifyot.generic_engine import run_generic_lopo

states = ["Memory", "Effector", "Exhausted", "Regulatory", "Other_T"]
edges = [
    ("Memory", "Effector"),
    ("Effector", "Exhausted"),
    ("Memory", "Exhausted"),
]

pairs = pd.read_csv("data/processed_pairs/GSE123813_pairs.csv")
candidate = GraphFlowCandidate(
    states=states,
    edges=edges,
    alpha=1.0,
    patient_balanced=True,
)

result = run_generic_lopo(
    pairs,
    candidate,
    bootstrap=300,
    patient_balanced=True,
    states=states,
)

print(result["qualified"], result["gates"])
```

## Interpretation boundary

QualifyOT is an evidence framework, not a causal-lineage validator or clinical certification system. A favorable predictive contrast does not establish a biological mechanism, patient-level safety or prognostic validity. Training-only refinement creates a new candidate; it does not retroactively change the frozen evidence state of the original candidate.

## Citation

Please cite the accompanying QualifyOT manuscript. Update the repository URL in `CITATION.cff` after creating the public GitHub repository.

## License

No software license is assigned in this archive. Add a license only after all authors agree on the intended reuse terms.
