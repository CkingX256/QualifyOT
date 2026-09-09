> In the current analysis, the third evidence axis is development-selected **Deployment Retention** $N_D$. The two Honest Confirmation estimands are defined in `INFERENCE_THEORY.md`.

# QualifyOT fused methodology

## One method, four layers

QualifyOT is intentionally organized as one evidence workflow rather than a collection of competing neural and statistical branches:

```text
Frozen scientific specification
(states + directed graph + reference rule + loss + Movement margin)
        |
        v
Structured candidate
GraphFlow anchor + bounded Transformer residual on the same graph
        |
        v
Patient-level qualification
Movement AND Utility AND Retention (intersection-union test)
        |
        +--> orthogonal profiles: stability / estimand robustness / tail safety / transportability
        |
        v
If structural evidence is unresolved
partial graph identification -> safe graph averaging -> new candidate -> re-lock -> re-qualify
```

The Transformer is therefore **not a fourth evidence gate**, not a graph learner, and not a replacement for the patient-level decision rule.

## 1. Structured candidate: GraphFlow nested inside a tiny Transformer

Let the frozen graph have incidence matrix \(B\in\mathbb R^{K\times E}\).  The ordinary GraphFlow anchor produces non-negative edge-flow predictions

\[
f_0(x)\ge 0,
\]

from source composition and elapsed time.  For patient/pair predictor input \(x\), a one-layer state-token Transformer produces one scalar residual score \(r_e(x)\) for each **existing frozen edge** \(e=(a\to b)\).  The proposed edge flow is

\[
\tilde f_e(x)
=
\Big[f_{0,e}(x)
+\rho\,p^{\mathrm{src}}_a\tanh r_e(x)\Big]_+,
\qquad 0\le \rho\le 1.
\]

The correction is therefore bounded by a fixed fraction \(\rho\) of source mass at the edge tail.  No new edge can be invented by the Transformer.

A deterministic feasibility scale \(s(x)\in[0,1]\) is then chosen as the largest common scale for which

\[
p^{\mathrm{src}} + B\{s(x)\tilde f(x)\}\ge0.
\]

The candidate prediction is

\[
\hat p^{\mathrm{GRT}}(x)
=
p^{\mathrm{src}}+B\{s(x)\tilde f(x)\}.
\]

Because every incidence column sums to zero,

\[
\mathbf 1^\top B=0,
\]

so total mass is preserved.  Feasibility scaling guarantees non-negativity; hence the prediction remains on the simplex.

### Exact nesting property

When \(\rho=0\), or when the zero-initialized residual head is identically zero,

\[
\hat p^{\mathrm{GRT}}=\hat p^{\mathrm{GraphFlow}}.
\]

Thus GraphFlow is a literal special case of the neural candidate rather than a separate methodological branch.

### Why the Transformer is trained end-to-end on composition

The previous experimental implementation trained a neural head against one residual-flow decomposition of the target.  In an over-complete graph that decomposition need not be unique.  The fused implementation removes this pseudo-label layer: the Transformer is trained directly against the observed target composition, with GraphFlow anchoring and patient-balanced weights.

The objective is a patient-balanced smooth composition loss plus an anchor penalty on departure from GraphFlow.  The outer held-out patient is never used for neural optimization.

## 2. Qualification remains Movement–Utility–Deployment Retention

For the frozen candidate:

\[
M=\text{candidate movement},\qquad
U=R_{\rm ref}-R_{\rm cand},\qquad
N=R_{\rm ref}-R_{\rm retained}.
\]

The primary state is Qualified only when

\[
\operatorname{LCB}(M)>\delta_M,
\quad
\operatorname{LCB}(U)>0,
\quad
\operatorname{LCB}(N)>0.
\]

These are three necessary claims, so their conjunction is an intersection-union test.  The Transformer does not change this logic and does not receive a privileged pass.

Retention stability, perturbation stability, reference/ontology robustness, transportability and tail safety remain orthogonal evidence profiles.

## 3. Structural uncertainty is not forced into a neural posterior

If a graph is not uniquely supported, QualifyOT does not use attention weights as edge probabilities.  A pre-specified finite graph library is compared at the physical-patient level with simultaneous uncertainty, yielding a predictive graph confidence set.  Edges can then be classified as core, optional or excluded relative to that library.

If redesign is scientifically justified, Safe Graph Averaging forms a training-only convex prediction over the pre-specified library while retaining the frozen graph as fallback.  The redesigned predictor is a **new candidate** and must be re-locked and re-qualified.

## 4. Small-n defaults

The recommended Transformer is deliberately tiny:

- one encoder layer;
- `d_model=8`;
- two attention heads;
- feed-forward width 16;
- zero dropout in deterministic reference runs;
- `residual_fraction=0.20`;
- patient-balanced training;
- fixed deterministic seed;
- GraphFlow anchor;
- no outcome-adaptive architecture search.

The residual fraction was checked only in synthetic development data (`0, 0.10, 0.20, 0.35`).  The default 0.20 is an interior conservative value, not a threshold selected from a real cohort.

## 5. Interpretation boundary

A GraphResidualTransformer result is predictive evidence under a frozen graph.  It is not evidence that Transformer attention recovered a causal lineage graph.  A Qualified state remains conditional on the reference, ontology, loss, patient population and frozen candidate specification.
