from __future__ import annotations

import numpy as np
import pandas as pd

from qualifyot_gse174554.candidate import ExpressionElasticNetCandidate, euclidean_simplex_projection
from qualifyot_gse174554.expression import SourceExpressionStore


def _fixture():
    patients = tuple(f"P{i}" for i in range(8))
    genes = tuple(f"G{i}" for i in range(20))
    values = np.arange(160, dtype=float).reshape(8, 20) / 100.0
    store = SourceExpressionStore(patients, genes, values)
    s = np.linspace(0.2, 0.8, 8)
    frame = pd.DataFrame(
        {
            "patient_id": patients,
            "pair_id": patients,
            "source_time": 0,
            "target_time": 1,
            "source__Tumor": s,
            "source__NonTumor": 1 - s,
            "target__Tumor": np.clip(s + 0.05, 0, 1),
            "target__NonTumor": 1 - np.clip(s + 0.05, 0, 1),
        }
    )
    return store, frame


def test_projection_is_simplex_and_euclidean_known_case():
    observed = euclidean_simplex_projection(np.array([[1.2, -0.2], [0.25, 0.75]]))
    assert np.allclose(observed, [[1.0, 0.0], [0.25, 0.75]])
    assert np.all(observed >= 0)
    assert np.allclose(observed.sum(1), 1)


def test_prediction_does_not_require_or_read_target_columns():
    store, frame = _fixture()
    model = ExpressionElasticNetCandidate(store, max_genes=10).fit(frame.iloc[:6])
    source_only = frame.iloc[6:].drop(columns=["target__Tumor", "target__NonTumor"])
    a = model.predict(source_only)
    mutated = frame.iloc[6:].copy()
    mutated[["target__Tumor", "target__NonTumor"]] = mutated[["target__NonTumor", "target__Tumor"]].to_numpy()
    b = model.predict(mutated)
    assert np.array_equal(a, b)


def test_fresh_shares_only_immutable_store():
    store, frame = _fixture()
    fitted = ExpressionElasticNetCandidate(store, max_genes=10).fit(frame.iloc[:6])
    fresh = fitted.fresh()
    assert fresh.source_store is fitted.source_store
    assert fresh._model is None and fresh._scaler is None and fresh._selected_indices is None
