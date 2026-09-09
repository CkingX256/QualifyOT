# Manuscript numerical consistency audit

Generated directly from machine-readable experiment outputs.

| Claim | Observed | Expected | Status |
|---|---:|---:|---|
| Naive type-I at 10 cells | 0.24648 | 0.24648 | PASS |
| Naive type-I at 1e5 cells | 0.49728 | 0.49728 | PASS |
| Patient type-I min rounded | 0.0477 | 0.0477 | PASS |
| Patient type-I max rounded | 0.0508 | 0.0508 | PASS |
| Duplication 1000 naive t ratio | 31.6307 | 31.6307 | PASS |
| Reference state changes | 5 | 5 | PASS |
| Reference can create Qualified | 3 | 3 | PASS |
| Loss state changes | 3 | 3 | PASS |
| GSE123813 delete-one n | 9 | 9 | PASS |
| GSE123813 Qualified | 5 | 5 | PASS |
| GSE123813 Promising | 2 | 2 | PASS |
| GSE123813 Equivocal | 2 | 2 | PASS |
| GSE236581 delete-one n | 21 | 21 | PASS |
| GSE236581 Adverse | 16 | 16 | PASS |
| GSE236581 Equivocal | 5 | 5 | PASS |
| AML delete-one n | 21 | 21 | PASS |
| AML Qualified | 11 | 11 | PASS |
| AML Promising | 3 | 3 | PASS |
| AML Equivocal | 7 | 7 | PASS |
| GSE272993 delete-one n | 35 | 35 | PASS |
| GSE272993 Promising | 33 | 33 | PASS |
| GSE272993 Equivocal | 2 | 2 | PASS |
| GSE272993 Retention zero 35/35 | 35 | 35 | PASS |
| Accuracy-winner cohorts | 9 | 9 | PASS |
| Accuracy winners core-qualified | 2 | 2 | PASS |
| Accuracy winners Holm-diagnostic qualified | 1 | 1 | PASS |
| GSE315928 DirectDelta risk | 0.1742070900100125 | 0.174207 | PASS |
| GSE315928 OT risk | 0.1743111105319003 | 0.174311 | PASS |
| GSE315928 reference risk | 0.2027475861257694 | 0.202748 | PASS |
| GSE315928 OT core seeds | 4 | 4 | PASS |
| GSE315928 Holm survivors all runs | 0 | 0 | PASS |
| GSE315928 OT p | 0.0227052232507631 | 0.022705 | PASS |
| GSE315928 honest n dev | 20 | 20 | PASS |
| GSE315928 honest n conf | 13 | 13 | PASS |
| GSE315928 all improve MAE | True | True | PASS |
| GSE315928 honest Qualified | 0 | 0 | PASS |
| GSE315928 family Qualified | 0 | 0 | PASS |
| GSE315928 all lambda zero | True | True | PASS |
| GSE123813 seed-stability 16 runs | 16 | 16 | PASS |
| GSE123813 qualified fraction | 1.0 | 1.0 | PASS |
| GSE272993 seed-stability 16 runs | 16 | 16 | PASS |

**Overall: PASS (41/41 checks).**
