> In the current analysis, the third evidence axis is development-selected **Deployment Retention** $N_D$. The two Honest Confirmation estimands are defined in `INFERENCE_THEORY.md`.

# Transformer integration: one recommended candidate

QualifyOT now has one recommended Transformer architecture:

`GraphResidualTransformerCandidate`

It is a nonlinear replacement for the **edge-flow law**, not a replacement for QualifyOT.

```text
source composition + elapsed time
      -> GraphFlow ridge anchor
      -> tiny Transformer residual
      -> flows only on frozen directed edges
      -> simplex-feasible composition prediction
      -> ordinary patient-level Movement / Utility / Deployment Retention IUT
```

The design has four safeguards: GraphFlow is the exact zero-residual special case; neural corrections are bounded by source tail-state mass; candidate prediction never reads target columns; and the outer held-out outcome is used only by the QualifyOT evidence engine.

Recommended usage:

```python
from qualifyot import QualifyOTPipeline, PipelineConfig

pipe = QualifyOTPipeline(
    states=["Memory", "Effector", "Exhausted", "Regulatory", "Other_T"],
    edges=[
        ("Memory", "Effector"),
        ("Effector", "Exhausted"),
        ("Memory", "Exhausted"),
    ],
    candidate="graph-transformer",
    config=PipelineConfig(bootstrap=5000, seed=20260831),
)
result = pipe.run(pairs)
```

CLI:

```bash
qualifyot-inspect candidate-run data/processed_pairs/GSE123813_pairs.csv \
  --states Memory,Effector,Exhausted,Regulatory,Other_T \
  --candidate graph-transformer \
  --graph "Memory->Effector,Effector->Exhausted,Memory->Exhausted"
```

The former `ResidualStateTransformerCandidate`, `GraphAwareTransformerCandidate` and `TransformerGraphFlowCandidate` remain importable only for backward reproduction of the previous development release. They are not separate recommended methodologies.

See `FUSED_METHODOLOGY.md` for the mathematical definition and `results/fused_methodology/README_EXECUTED_RESULTS.md` for executed validation.
