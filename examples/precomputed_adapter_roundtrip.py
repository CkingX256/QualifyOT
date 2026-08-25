from pathlib import Path
import json
import numpy as np
import pandas as pd
from qualifyot.candidate_api import PrecomputedPredictionCandidate

ROOT = Path(__file__).resolve().parents[1]
pairs = pd.read_csv(ROOT / "data/processed_pairs/GSE123813_pairs.csv")
states = ["Memory", "Effector", "Exhausted", "Regulatory", "Other_T"]
# Software-contract demonstration only: source compositions are materialized as
# externally supplied simplex predictions. This is not a CellRank/scVelo benchmark.
for state in states:
    pairs[f"external_demo__{state}"] = pairs[f"source__{state}"]
expected = pairs[[f"external_demo__{s}" for s in states]].to_numpy(float)
candidate = PrecomputedPredictionCandidate(states, prefix="external_demo", name="ExternalRoundTrip")
observed = candidate.fit(pairs).predict(pairs)
summary = {
    "rows": int(len(pairs)),
    "states": states,
    "max_abs_roundtrip_error": float(np.max(np.abs(observed - expected))),
    "simplex_max_error": float(np.max(np.abs(observed.sum(axis=1)-1.0))),
    "interpretation": "Adapter I/O contract verified; this is not an external-method performance benchmark."
}
out = ROOT / "examples/precomputed_adapter_roundtrip.json"
out.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
