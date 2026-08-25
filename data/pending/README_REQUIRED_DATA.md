# GSE197268 status — intentionally not fabricated

This package does not contain a cell-state-annotated GSE197268 composition table. The official GEO deposit contains 109 per-sample 10x expression matrices, while the authors' repository states that cell-level metadata and clustered AnnData objects were hosted on a Google Drive link. In the current audit the Drive folders were visible but the `cell_metadata` folder was empty, so a faithful author-annotation table could not be recovered locally.

Therefore GSE197268 is **not scored** in the supplied reference results. Do not substitute guessed labels or post-hoc clustering and call it confirmatory validation.

Verified public sample geometry in this package:
- Baseline -> D7 host PBMC: 20 paired patients
- Infusion -> D7-CART: 21 paired patients
- Infusion -> D7 all PBMC: 28 paired patients
- D7 -> D14 host PBMC: 4 paired patients

Recommended primary reconstruction target: Infusion -> D7-CART (21 paired patients), after recovering or independently freezing a cell-state annotation rule.

Official sources:
- GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE197268
- Authors' code: https://github.com/getzlab/Haradhvala_et_al_2022
- Author-linked Drive: https://drive.google.com/drive/folders/1vw7J8HqUX22ICZmJ0UjAYEBpVjRJ9U9-?usp=sharing

The expression matrices are large and are not redistributed in this compact package.
