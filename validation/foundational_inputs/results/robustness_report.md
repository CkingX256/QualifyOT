# Foundational-input robustness audit — GraphFlow eta and state ontology

This analysis is orthogonal to the locked primary contracts. It stress-tests two foundational inputs to the structured-candidate interface without allowing either sensitivity analysis to relabel a primary result.

## 1. GraphFlow scalarization coefficient eta

The residual-flow label program is

`min ||r||_1 + eta * 1^T f  subject to  B f + r = Delta p, f >= 0`,

with frozen `eta = 1e-6`.

Across 51 observed longitudinal pairs from GSE123813 (9), GSE236581 (34) and GSE120575 (8), flow labels were recomputed at

`{1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2}`.

At every tested coefficient in `{1e-6, 1e-5, 1e-4, 1e-3, 1e-2}`, the residual-flow label matched the frozen-eta label to the reported numerical precision: maximum absolute difference 0 and 0/255 pair-by-coefficient comparisons changed by more than `1e-8`.

At `1e-8` and `1e-7`, each of the six dataset-by-coefficient stress cells changed at least one flow label. Maximum deviations were 0.321502 for GSE123813, 0.364930 for GSE236581 and 0.013183 for GSE120575. These values are interpreted as an implementation-level linear-program degeneracy/tie-breaking stress regime, not as alternative performance tunings.

The complete patient-level LOPO GraphFlow evidence pipeline was rerun at `eta = 1e-6, 1e-4, 1e-2` with the same physical-patient folds, graph, patient weighting, nested reference and evidence settings. The qualitative state was invariant across these tested plateau values:

- GSE123813: Promising / Promising / Promising;
- GSE236581: Adverse / Adverse / Adverse;
- GSE120575: Equivocal / Equivocal / Equivocal.

This establishes local implementation robustness on the tested coefficient grid. It does not establish invariance to arbitrary eta, graph topology, solver tolerances or alternative LP formulations.

## 2. State ontology

Ontology changes the estimand itself, so it was audited separately from eta. Frozen GSE272993 nine-state source, target, reference, candidate and retained prediction arrays were re-expressed under five outcome-blind, simplex-preserving coarsenings declared for this sensitivity analysis, plus the original nine-state representation.

This is deliberately a **fixed-prediction re-expression audit**: it isolates how state resolution changes the patient-level evidence calculation while holding the already-fitted prediction arrays fixed. It is not a claim that retraining the candidate under each ontology would return identical predictions.

In the primary 1,000-bootstrap audit:

- Movement LCB remained positive in 6/6 mappings;
- Utility LCB was positive in 4/6 and non-positive in 2/6;
- development-selected Deployment Retention was exactly zero in 6/6;
- no ontology Qualified.

A second audit used four independent resampling seeds with 2,000 patient-level bootstrap replicates per seed. Across the resulting 24 ontology-by-seed evaluations:

- 0/24 Qualified;
- Deployment Retention remained zero throughout;
- the six-state adjacent ontology sat on the Utility boundary and alternated between Promising and Equivocal;
- the other five mappings retained their qualitative state across all four seeds.

The robust conclusion is therefore intentionally narrower than “ontology invariance”: none of the declared state resolutions created evidence for non-zero deployment influence, but Utility magnitude and near-boundary state assignment remain ontology-dependent. That dependence is expected because ontology is part of the frozen estimand rather than a cosmetic label dictionary.

## 3. Claim boundary

These experiments establish local robustness over a declared eta grid and six explicit outcome-blind ontology resolutions. They do not establish invariance to arbitrary state definitions, graph topologies, biological annotations, retrained ontology-specific models, clinical endpoints or solver formulations.
