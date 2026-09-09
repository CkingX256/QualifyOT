# Statistical positioning

For the inferential definitions used in the manuscript, including the two Honest Confirmation scopes and development-selected Deployment Retention, see **`INFERENCE_THEORY.md`**. Historical numerical results remain unchanged.

# Statistical positioning for Nature Methods review

## Primary inferential statement

QualifyOT asks whether a pre-specified candidate deserves predictive influence under a frozen patient-level estimand. The primary decision is

\[
H_1=H_{1,M}\cap H_{1,U}\cap H_{1,R},\qquad
H_0=H_{0,M}\cup H_{0,U}\cup H_{0,R}.
\]

Movement, Utility and Deployment Retention are necessary claims. If each component test is valid at level α under its own null, then under any global-null parameter point at least one component null is true and

\[
P(\text{all pass})\le P(\text{that true-null component passes})\le \alpha.
\]

This is distinct from simultaneous edge, graph, or weight-grid search, where family-wise control remains required.

## What is not claimed

- no exact finite-sample theorem for dependent LOPO bootstrap components at *n*=9;
- no causal interpretation of graph confidence sets or predictive edge profiles;
- no universal subgroup-safety guarantee from average NPI;
- no universal superiority of QualifyOT over one-standard-error CV or simple paired tests;
- no claim that Bayesian-bootstrap probabilities are Bayes factors.

## Why a fully hierarchical Bayesian model is optional rather than primary

A hierarchical multinomial/logistic-normal model would be scientifically legitimate only after fixing: the likelihood, random-effects structure, prior scales, graph-flow parameterization, prior sensitivity, and how its target estimand maps to the existing MAE-based locked evidence contract. Introducing such a model as a replacement would change both the candidate and the estimand. The current upgrade therefore adds Bayesian-bootstrap and proper-scoring sensitivities without retroactively redefining the locked primary analysis.
