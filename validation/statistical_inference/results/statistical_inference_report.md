# Statistical inference audit

## Dual-scope Honest Confirmation
- Analyses audited: 9 (5 GSE174554 + 4 GSE315928).
- Cohort-conditional Qualified: 0/9.
- Population Qualified: 0/9.
- ExpressionElasticNet GSE174554: cc LCBs G_M=0.278542, U=-0.117003, N_D=-7.74046e-16; population LCBs G_M=-0.108481, U=-0.684921, N_D=-0.774046.

## Movement-margin sensitivity
- Positive Movement LCBs across delta_M in {0.005,0.01,0.02,0.05}: 32/32.
- Minimum diagnostic Movement LCB: 0.046668.
- This is a sensitivity audit and does not alter the primary pre-specified delta_M=0.01.

## Scalar Utility-only analogue versus QualifyOT
| scenario                      |   n |   replicates |   movement_mean |   utility_mean |   deployment_retention_mean |   utility_only_positive |   qualifyot_positive |   hoeffding_radius |
|:------------------------------|----:|-------------:|----------------:|---------------:|----------------------------:|------------------------:|---------------------:|-------------------:|
| all_three_supported           |  80 |        20000 |            0.12 |           0.12 |                        0.12 |                 0.9998  |              0.9994  |          0.0547333 |
| movement_boundary             |  80 |        20000 |            0    |           0.12 |                        0.12 |                 0.9998  |              0.01035 |          0.0547333 |
| utility_boundary              |  80 |        20000 |            0.12 |           0    |                        0.12 |                 0.01055 |              0.01055 |          0.0547333 |
| deployment_retention_boundary |  80 |        20000 |            0.12 |           0.12 |                        0    |                 0.9997  |              0.00865 |          0.0547333 |

This comparison is a decision-target stress test, not a reproduction or superiority claim against Learn-then-Test.
