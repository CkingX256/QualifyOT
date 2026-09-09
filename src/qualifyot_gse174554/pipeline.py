from __future__ import annotations

from pathlib import Path

import pandas as pd

from qualifyot.adaptive_candidates import PatientMultiScaleDeltaCandidate
from qualifyot.candidates import DirectDeltaRidgeCandidate, RobustBlendCandidate
from qualifyot.comparison_candidates import CompositionEntropicOTCandidate

from .candidate import ExpressionElasticNetCandidate
from .expression import SourceExpressionStore


STATES = ["Tumor", "NonTumor"]
CANDIDATE_ORDER = [
    "ExpressionElasticNet",
    "RobustBlend",
    "DirectDeltaRidge",
    "PatientMultiScaleDelta",
    "CompositionEntropicOT",
]


def load_processed_inputs(root: str | Path) -> tuple[pd.DataFrame, SourceExpressionStore]:
    root = Path(root)
    pairs = pd.read_csv(root / "data" / "processed" / "gse174554_processed_pairs.csv", keep_default_na=False)
    store = SourceExpressionStore.load(root / "data" / "processed" / "source_expression_log1p_cpm.npz")
    if set(pairs.patient_id.astype(str)) != set(store.patient_ids):
        raise ValueError("pair table and source expression store patient sets differ")
    return pairs, store


def candidate_family(store: SourceExpressionStore):
    return [
        ExpressionElasticNetCandidate(store, states=tuple(STATES)),
        RobustBlendCandidate(states=list(STATES), name="RobustBlend"),
        DirectDeltaRidgeCandidate(states=list(STATES), alpha=10.0, patient_balanced=True, name="DirectDeltaRidge"),
        PatientMultiScaleDeltaCandidate(states=list(STATES), name="PatientMultiScaleDelta"),
        CompositionEntropicOTCandidate(states=list(STATES), epsilon=0.25, patient_balanced=True, name="CompositionEntropicOT"),
    ]

