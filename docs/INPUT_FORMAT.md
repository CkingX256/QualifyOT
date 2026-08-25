# Input format

QualifyOT operates on patient-level longitudinal composition pairs.

A pair table must contain:

- `patient_id`: physical patient identifier.
- `pair_id`: unique transition identifier; optional for `quicktest`, which will create one if absent.
- `source_time` and `target_time`: ordered visit labels or numeric times.
- `source__STATE` and `target__STATE` columns for every state.

Each source and target composition must be non-negative and sum to one.

Example for five states:

```text
patient_id,pair_id,source_time,target_time,source__Memory,...,target__Other_T
P01,P01:pre->post,pre,post,0.42,...,0.08
```

The biological state definition, graph, reference rule and primary data-quality filters should be fixed before held-out target scoring when the analysis is intended to be confirmatory.
