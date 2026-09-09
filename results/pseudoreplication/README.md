# Pseudoreplication stress test

Executed with seed `20260902` and `50,000` Monte Carlo replicates per design cell.

The data-generating process is `Y_ij = theta + U_i + eps_ij`, with independent physical patients, `U_i ~ N(0,1)`, and within-patient cell noise `eps_ij ~ N(0,1)`. The simulation uses the exact Gaussian sufficient-statistic distributions for patient means and within-patient sums of squares, so very large cell counts are evaluated without materializing artificial cell matrices.

Primary null comparison: one-sided alpha=0.05, `20` physical patients, cells/patient = [10, 100, 1000, 10000, 100000]. A naive cell-level t test is intentionally misspecified by treating all cells as independent; the patient-level test uses one value per physical patient.

The mechanical-duplication audit exactly repeats existing cell rows. The patient-level statistic is invariant by construction; the naive statistic is not.

Interpretation: this is a controlled demonstration of the inferential distinction between representation-level information from many cells and independent patient-level evidence. It does **not** imply that every cell-level model is invalid; hierarchical/cluster-aware procedures can be valid for appropriately defined estimands.
