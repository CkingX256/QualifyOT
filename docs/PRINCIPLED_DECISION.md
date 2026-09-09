# Principled qualification and safe redesign

QualifyOT separates a **core inferential decision** from diagnostics that describe stability, robustness and safety.

## Core Movement–Utility–Deployment-Retention decision

The primary decision is an intersection–union rule with three necessary axes:

1. **Movement** — the lower confidence bound for prediction movement exceeds a pre-specified minimally relevant margin.
2. **Utility** — the lower confidence bound for candidate-minus-reference predictive utility is positive.
3. **Deployment Retention** — the lower confidence bound for net benefit of the development-selected deployment weight is positive on confirmation.

A candidate is `Qualified` only when all three axes pass. The positive-weight outer-fold fraction is not a core gate; it is reported as **retention stability**. This avoids turning one-patient changes in small cohorts into an arbitrary percentage threshold.

The previous six-gate rule is retained in the package as a legacy reproduction path. It is not deleted or rewritten after seeing new results.

### Why the three necessary tests are not Bonferroni-split

Let the global alternative require all three statements to hold. The global null is therefore the union of the component nulls. If every component test is valid at level alpha, the probability that all components pass under the global null is bounded by alpha. This is the standard intersection–union argument. It does **not** remove the need to control multiplicity *inside* a component procedure, such as searching many mixing weights or resolving many graph edges. QualifyOT therefore retains simultaneous max-t control for those families.

## Evidence-deficit profile

A non-qualified result is accompanied by distances to the three decision boundaries:

- movement deficit;
- utility deficit;
- retention deficit.

The largest positive deficit is reported as the limiting evidence axis. This turns `Promising` or `Equivocal` into an actionable statement about what evidence is missing rather than a terminal label.

## Partial graph identification

`qualifyot.partial_identification.graph_confidence_set` constructs simultaneous pairwise risk intervals over a pre-specified graph library. A graph is excluded only when another graph has significantly lower patient-level risk under the simultaneous family. The remaining set is a **risk confidence set**, not a forced single winner.

If edge metadata are supplied, the package reports:

- `core_edges`: present in every graph in the confidence set;
- `optional_edges`: present in some but not all supported graphs;
- `excluded_edges`: absent from every supported graph.

This is the preferred interpretation when patient evidence cannot identify a unique graph.

## Safe Graph Averaging

`qualifyot.safe_averaging.safe_graph_average` selects a convex mixture over a pre-specified prediction library. The frozen graph is always present as a fallback. Safety is checked against that frozen graph with a max-t upper confidence bound calibrated over the complete mixture grid, so the grid search itself does not silently become an unadjusted model-shopping step.

`qualifyot.graph_design.SafeGraphAveragingCandidate` embeds the procedure in the candidate API. Graph weights are selected from patient-split predictions **inside `fit()` only**. Outer held-out targets are never used to choose graph weights. The resulting ensemble is a new candidate and must still be evaluated by the outer evidence engine.

## Tail-safety profile

Average benefit does not imply patient-subgroup safety. `qualifyot.tail_safety.tail_safety_profile` therefore reports an orthogonal safety profile based on patient-level harm relative to the reference, including:

- mean harm;
- harmed-patient fraction;
- an upper-tail quantile;
- upper-tail CVaR and its patient-bootstrap interval.

Tail safety is a diagnostic axis, not a mandatory qualification gate in the current implementation.

## Predictive assurance

`qualifyot.replication_planning.bayesian_bootstrap_assurance` integrates uncertainty in the observed patient-effect distribution with a Bayesian bootstrap before simulating future patient evidence. This is more cautious than treating the empirical effect distribution as known. It remains an assumption-dependent design aid and is not a universal prospective guarantee.

## Scope

These procedures qualify predictive evidence. They do not by themselves establish lineage causality, clinical efficacy, prognostic validity or individual-patient safety.
