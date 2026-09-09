# Data provenance

QualifyOT treats the physical patient as the inferential unit. The repository contains redistributable processed patient-level inputs and compact source-derived metadata used by the reported analyses. Large third-party source datasets are not redistributed; public accession identifiers and retrieval information are provided below.

## Processed longitudinal cohorts

The executable patient-pair tables are stored in `data/processed_pairs/`.

| Accession / cohort | File | Role |
|---|---|---|
| GSE123813 | `GSE123813_pairs.csv` | locked real-data analysis |
| GSE175522 | `GSE175522_D0_D7_Bcell_pairs.csv` | real-data analysis |
| GSE272993 | `GSE272993_9state_pairs.csv` | nine-state Retention-limited analysis |
| GSE235063 | `GSE235063_AML_primary_pairs.csv` | AML analysis |
| GSE120575 | `GSE120575_pairs.csv` | real-data stress analysis |
| GSE229772 | `GSE229772_HCC_primary_pairs.csv` | HCC analysis |
| GSE179994 | `GSE179994_pairs.csv` | real-data analysis |
| GSE236581 | `GSE236581_pairs.csv` | adverse/reference-relativity analysis |
| Sound Life | `SoundLife_D0_D7_Bcell_5state_pairs.csv` | larger-cohort analysis |
| GSE315928 | `GSE315928_B_F1_pairs.csv` and secondary transition tables | external post-specification validation |

Compact source metadata used to reconstruct several processed tables are retained in `data/source_small/` where redistribution is appropriate.

## GSE174554 expression validation

The high-dimensional source-expression validation is under `validation/gse174554_expression/`. The repository includes the processed primary-source expression matrix used for candidate construction, frozen patient partitions, target-access and leakage audits, patient-level result tables, source data and extension code. Recurrence expression was not used for predictor construction.

## Integrity principle

No unavailable cell-level expression object is reconstructed from composition proportions. Analyses that require third-party raw data should retrieve the corresponding public source rather than infer or synthesize missing measurements.
