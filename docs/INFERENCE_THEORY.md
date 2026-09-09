# QualifyOT inference foundation

## Status and scope

This document defines the inferential interpretation used by the manuscript. Earlier outputs may use the shorter label **Retention** or NPI; the manuscript uses **Deployment Retention** to make explicit that the weight is selected from development data before confirmation.

The physical patient is the inferential unit. The complete estimand-defining contract is

\[
\gamma=(\mathcal P,\tau,\mathcal E,\Phi,\mathcal C,\mathcal H,r,\mathcal O,L,\delta_M,\Lambda,\mathcal S,\mathcal I),
\]

covering the patient population/set, prediction task or horizon, eligibility/integrity/missingness rules, preprocessing, candidate family and search protocol, reference, ontology, loss, Movement margin, deployment-weight set, split/seed rule and inference rule.

Let development data be \(D\) and let \(\widehat\lambda(D)\) be the deployment weight frozen before confirmation. For a fixed candidate,

\[
G_M=A-\delta_M B,\qquad
U=R_{\rm ref}-R_{\rm cand},\qquad
N_D=R_{\rm ref}-R_{\widehat\lambda(D)}.
\]

Qualification requires all three claims to be positive under the declared inference scope.

## 1. Why Deployment Retention is not oracle Retention

If an oracle could choose \(\lambda^*\in[0,1]\) after population risk were known, and \(\lambda=1\) exactly reproduces the candidate, then

\[
N^*=R(0)-R(\lambda^*)\ge R(0)-R(1)=U.
\]

Hence positive Utility already implies positive *oracle* Retention. The non-redundant QualifyOT claim is instead

\[
N_D=R(0)-R(\widehat\lambda(D)),
\]

the transfer performance of the policy selected before confirmation. Conditional on \(D\), \(\widehat\lambda(D)\) is fixed and therefore does not consume confirmation type-I error. Selection quality is a separate question: finite-grid concentration follows under independent development validation, while overlapping OOF development scores require an additional stability or weak-dependence argument.

## 2. Intersection–union logic and candidate families

For component alternatives \(H_{1M},H_{1U},H_{1N_D}\), the global alternative is their intersection and the global null their union. If each component test is level \(\alpha\) under its own null, rejecting only when all three reject has global size at most \(\alpha\), without an axis-independence assumption.

For candidate \(c\), valid component p-values give

\[
p_c=\max\{p_{cM},p_{cU},p_{cN_D}\}.
\]

This is a valid p-value for the candidate union null. Holm applied across the frozen candidate-level p-values controls family-wise error under arbitrary candidate dependence. For the same component p-values, every candidate accepted by a flat Bonferroni rule requiring all three components \(\le\alpha/(3C)\) is also reached and rejected by candidate-level IUT–Holm. This is a structural containment relative to that flat rule, not dominance over all multiple-testing methods.

## 3. Movement as a margin contrast

The descriptive ratio is

\[
M=A/B,
\]

where \(A\) is candidate movement relative to Persistence and \(B\) is observed source-to-target movement. If \(B>0\),

\[
M>\delta_M\iff G_M=A-\delta_M B>0.
\]

The contrast avoids direct inference on a near-zero denominator while preserving the scientific margin. The primary margin remains \(\delta_M=0.01\). A diagnostic analysis over \(\{0.005,0.01,0.02,0.05\}\) does not redefine that contract.

## 4. Honest Confirmation has two distinct estimand scopes

The previous localized proof was too easy to misread as an unconditional population statement because its interval endpoints may depend on a confirmation patient's target-blind input \(X_i\). The repaired API separates the two valid questions explicitly.

### 4.1 Cohort-conditional Honest Confirmation

Freeze all development-side choices using \(D\). For independent confirmation records \((X_i,Y_i)\), condition on \((D,X_{1:n})\). Prediction is target-blind and therefore fixed at this stage.

With normalized Hellinger distance,

\[
G_{Mi}\in[A_i-\delta_M,A_i].
\]

For patient-level MAE, reverse triangle inequality gives target-free, prediction-localized envelopes

\[
|U_i|\le C_{Ui}=L_i(\hat p_{\rm ref},\hat p_{\rm cand}),
\]

\[
|N_{Di}|\le C_{Ni}=L_i(\hat p_{\rm ref},\hat p_{\widehat\lambda(D)}).
\]

If independent \(S_i\in[a_i,b_i]\), heterogeneous Hoeffding gives the one-sided lower bound

\[
\bar S-\sqrt{\frac{\log(1/\alpha)\sum_i(b_i-a_i)^2}{2n^2}}.
\]

Because the endpoints above can depend on \(X_i\), this scope targets

\[
\frac1n\sum_{i=1}^n E[S_i\mid D,X_i],
\]

not automatically \(E[S]\) over a random future covariate distribution. The public functions are `honest_confirmation_iut_cohort_conditional` and the compatibility alias `honest_confirmation_iut_localized`.

### 4.2 Population Honest Confirmation

For iid confirmation patients from a declared target population, use deterministic support that does **not** depend on realized confirmation covariates. Normalized Hellinger gives

\[
G_{Mi}\in[-\delta_M,1].
\]

For \(K\)-state simplex MAE,

\[
0\le L_i\le 2/K,
\]

hence

\[
U_i,N_{Di}\in[-2/K,2/K].
\]

Ordinary Hoeffding with these global ranges targets the unconditional population means. The public function is `honest_confirmation_iut_population`.

`honest_confirm_predictions(..., return_both=True)` computes both pre-declared scopes from exactly the same frozen prediction arrays; the scopes must not be selected after inspecting which is favorable.

### 4.3 Target-sealed representation agnosticism

Conditional on the complete development procedure, representation learning, feature selection, architecture choice and hyperparameter selection can be arbitrarily complex without *adding confirmation type-I error*, provided confirmation outcomes remain sealed until scoring, physical patients satisfy the relevant independence assumption and the declared score-support conditions hold. This is an inference-validity statement, not a prediction-generalization or calibration guarantee.

## 5. Target-blind/per-patient integrity selection

If independent patient records are filtered by a pre-specified separable indicator \(I_i=h(Z_i)\), conditioning on the selection indicators preserves cross-patient independence. For iid records, retained observations follow \(P(\cdot\mid I=1)\). Thus post-freeze integrity/mappability failures can be retained without replacement or repartitioning if the rule was pre-specified and applied per patient, but the inferential population becomes the integrity-passing population. This does not justify transportability to excluded patients and does not excuse outcome-direction-based filtering.

## 6. Efficient LOPO is not Honest finite-sample inference

LOPO outer scores share highly overlapping training sets. Resampling the already-computed outer score vector does not restore independence. A sufficient first-order bridge is

\[
\max_i|\psi(X_i;\widehat{\mathcal A}_{-i})-\psi(X_i;\mathcal A_0)|=o_p(n^{-1/2}),
\]

for a deterministic population-limit procedure \(\mathcal A_0\), together with an ordinary CLT for \(\psi(X;\mathcal A_0)\). This is a sufficient stability condition, not an assertion that every nested selector or neural/transport procedure satisfies it. Efficient LOPO therefore remains a diagnostic/asymptotic mode unless explicit conditions are verified.

## 7. Relationship to learn-then-test and generic predictive risk control

Generic held-out risk-control and learn-then-test procedures already provide principled ways to certify scalar predictive risks and control multiplicity over algorithm families. QualifyOT does not claim those generic ingredients as new. Its claimed contribution is the domain-specific contract-indexed decision object for longitudinal single-cell prediction: physical-patient replication, a scientific Movement margin, candidate Utility, development-selected Deployment Retention, and candidate-family control under a frozen analysis contract. Generic risk-control tools can be used within or alongside this framework.

## 8. Current claim boundary

A Qualified result is predictive and contract-relative. It does not prove causal lineage, biological mechanism, clinical benefit, individual-patient safety or transportability outside the declared population. The GSE315928 analysis is post-specification, not untouched prospective confirmation. GSE174554 establishes target-sealed source-expression compatibility, not a matched state-of-the-art temporal-expression benchmark. The manuscript states these remaining empirical boundaries directly rather than claiming that they have been completed.
