# External confirmation protocol

This protocol governs future genuinely target-sealed confirmation. Existing GSE315928 remains explicitly **post-specification** and is not retroactively promoted to untouched confirmation.

## Before any confirmation target is accessed

Freeze and hash all estimand-defining elements:

1. physical patient population/set and patient IDs;
2. prediction transition/horizon;
3. eligibility, integrity, missingness and failure rules;
4. preprocessing/normalization/representation construction;
5. complete candidate family and method versions;
6. hyperparameter/search budgets and random seeds;
7. reference rule;
8. ontology/annotation adapter;
9. patient-equal loss;
10. Movement margin `delta_M`;
11. Deployment-Retention weight grid and selector;
12. split/seed rule;
13. inference scope(s), component tests and candidate-family multiplicity rule.

Where feasible, also freeze confirmation predictions before target unlock. Create SHA256 manifests and an external timestamped record before proceeding.

## Hard target-access gate

Confirmation outcomes/target compositions must not influence eligibility, representation, feature selection, graph construction, model fitting, hyperparameters, candidate ordering, reference selection, Deployment-Retention weight selection or inference-scope choice. If metadata needed for integrity checking contains target-related fields, the pre-specified per-patient rule and accessed fields must be logged; any resulting exclusion changes the target population to the integrity-passing population and must not trigger replacement or repartitioning.

## One-shot scoring

After the frozen manifest is verified unchanged, unlock the target once and score all pre-specified candidates. Report both scopes only if both were declared before unlock:

- **cohort-conditional**: localized prediction-dependent ranges; estimand is the average conditional effect over realized target-blind confirmation inputs;
- **population**: covariate-independent global support; estimand is the unconditional iid patient-population mean.

For each candidate compute Movement `G_M`, Utility `U`, development-selected Deployment Retention `N_D`, component one-sided p-values/LCBs, candidate IUT p-value `max(p_M,p_U,p_ND)` and Holm correction over the frozen family. A technical failure must follow a pre-declared failure policy; candidates may not be silently removed after outcomes are seen.

## Efficient LOPO

LOPO may be reported as a data-efficient secondary diagnostic. Because outer fits overlap, fixed-score bootstrap/t intervals are not relabeled as Honest finite-sample inference. Any asymptotic claim must state and justify the required stability conditions.

