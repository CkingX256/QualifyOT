> In the current analysis, the third evidence axis is development-selected **Deployment Retention** $N_D$. The two Honest Confirmation estimands are defined in `INFERENCE_THEORY.md`.

# QualifyOT robust statistical extensions

## Scope

This upgrade keeps the **Movement–Utility–Deployment Retention intersection–union test (IUT)** as the primary qualification rule. It does **not** replace the core decision with a Bayes factor, a posterior edge probability, or a sample-size-dependent empirical threshold. The new modules are orthogonal evidence and design layers intended to address small-*n* calibration, benchmark transparency, structural partial identification, and tail/subgroup risk.

## Why the proposed full Bayesian replacement was not adopted as the primary rule

Several attractive ideas in the proposed redesign are statistically useful, but three proposed equivalences are not valid without additional assumptions.

1. **IUT conjunction is not an ordinary multiple-discovery family.** If all three necessary component tests are level-α, the global conjunction has Type-I error at most α under the union null. Bonferroni or closed testing across Movement, Utility and Deployment Retention is therefore not automatically required. Multiplicity **inside** searched weight/edge/graph families remains a different problem and still receives simultaneous control.
2. **PSIS-LOO, Bayes factors and posterior probabilities of ΔELPD are different objects.** They cannot be combined into a single evidence scale without a fully specified probabilistic model and prior-sensitivity analysis. QualifyOT already produces actual held-out-patient predictions, so a patient-equal proper log score can be computed directly without using PSIS merely to approximate a LOO score.
3. **A larger Bayesian edge-inclusion score does not resolve a data-resolution limit by itself.** In small cohorts, apparent edge resolution may be prior-driven. The package therefore reports Bayesian-bootstrap *predictive library* probabilities only as an orthogonal profile and keeps simultaneous graph confidence sets as the structural-resolution result.

## New modules

### `bayesian_evidence.py`

Implements Rubin's Bayesian bootstrap on one value per physical patient. For patient contributions \(x_i\), draw

\[
w\sim\operatorname{Dirichlet}(1,\ldots,1),\qquad \theta_w=\sum_i w_i x_i.
\]

The output includes \(P(\theta_w>\delta\mid x)\) and a credible interval. This is a non-parametric posterior over the empirical patient distribution, **not** a Bayes factor and **not** a theorem for uncertainty caused by refitting the complete prediction pipeline.

The graph-library function reports the probability that each pre-specified graph has the smallest patient-weighted predictive risk under Bayesian-bootstrap draws. Edge inclusion is induced from these predictive model-selection probabilities. These are not causal edge posteriors.

### `benchmarks.py`

Provides transparent head-to-head patient-level comparators:

- one-sided paired *t* test;
- paired sign-flip randomization test (with its sign-exchangeability assumption made explicit);
- percentile bootstrap lower bound;
- bootstrap-*t* lower bound;
- Bayesian-bootstrap probability of positive mean effect.

These are comparators and sensitivity profiles, not hidden qualification gates.

### `proper_scoring.py`

Adds a patient-equal normalized multinomial log-score sensitivity. The primary locked metric remains patient-equal MAE for backward compatibility. Because the candidate and reference predictions are already truly held out at the physical-patient level, direct log-score evaluation is preferred over introducing PSIS solely to approximate LOO prediction.

### `robust_weight_selection.py`

Adds an optional **tail-safe retained-influence selector**. Let \(D_i(\lambda)\) be patient-level harm relative to the reference. A positive weight must satisfy both

\[
\operatorname{UCB}\{E[D(\lambda)]\}\le \delta_{\rm mean}
\]

and an empirical simultaneous CVaR constraint

\[
\operatorname{UCB}\{\operatorname{CVaR}_{\tau}(D(\lambda))\}\le \delta_{\rm tail}.
\]

The zero-weight reference is an identity fallback. The CVaR bootstrap is an empirical robustness procedure, not an exact finite-sample subgroup-safety theorem.

### `robust_averaging.py`

Extends Safe Graph Averaging with the same two-level mean/tail non-harm screen relative to the frozen graph. A redesigned graph mixture remains a new candidate and must be re-locked and re-qualified.

### `edge_shrinkage.py`

Adds **Continuous Edge Shrinkage**, a design-mode alternative to winner-takes-all graph selection. It learns attenuation coefficients \(a_e\in[0,1]\) on a pre-specified directed supergraph from training-patient OOF flow predictions and a pre-specified regularization path. It is intentionally *not* called graphical LASSO, because graphical LASSO refers to sparse precision-matrix estimation.

### `calibration.py`

Provides empirical null critical values for calibration diagnostics. These are not used to silently replace the scientific Movement margin \(\delta_M\). A sample-size-dependent Movement threshold would change the scientific meaning of the estimand with *n* and is therefore kept outside the primary rule.

## Important negative result from executed subgroup stress test

The mean-risk selector retained high candidate weight under 10–30% reverse-mechanism minorities while almost all reverse-mechanism patients were harmed. The tail-safe selector avoided this harm by falling back to zero candidate weight, but at the cost of discarding benefit for the majority. This is an intended and important trade-off: **tail safety is not free power**.

## Structural interpretation

Partial graph identification and Bayesian-bootstrap graph profiles can coexist. A graph can be retained in the simultaneous confidence set while having low probability of being the draw-wise best graph. Conversely, a graph can most often be best under Bayesian-bootstrap weights while the confidence set remains large. This is evidence of limited structural resolution, not a contradiction.
