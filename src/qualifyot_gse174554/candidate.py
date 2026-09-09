from __future__ import annotations

import hashlib
import json
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import psutil
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.preprocessing import StandardScaler

from qualifyot.candidate_api import CandidatePredictor

from .expression import SourceExpressionStore


def euclidean_simplex_projection(values: np.ndarray) -> np.ndarray:
    """Project each row onto the probability simplex in Euclidean distance."""

    x = np.asarray(values, dtype=np.float64)
    one_dimensional = x.ndim == 1
    if one_dimensional:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] == 0 or not np.isfinite(x).all():
        raise ValueError("simplex projection needs a finite non-empty matrix")
    out = np.empty_like(x)
    for i, row in enumerate(x):
        ordered = np.sort(row)[::-1]
        cssv = np.cumsum(ordered) - 1.0
        valid = ordered - cssv / np.arange(1, len(row) + 1) > 0
        rho = int(np.flatnonzero(valid)[-1])
        theta = cssv[rho] / float(rho + 1)
        out[i] = np.maximum(row - theta, 0.0)
        out[i] /= out[i].sum()
    return out[0] if one_dimensional else out


@dataclass
class ExpressionElasticNetCandidate(CandidatePredictor):
    source_store: SourceExpressionStore
    states: tuple[str, str] = ("Tumor", "NonTumor")
    alpha: float = 0.01
    l1_ratio: float = 0.5
    max_genes: int = 2000
    max_iter: int = 100000
    tol: float = 1e-7
    selection: str = "cyclic"
    name: str = "ExpressionElasticNet"
    _selected_indices: np.ndarray | None = None
    _scaler: StandardScaler | None = None
    _model: MultiTaskElasticNet | None = None
    _diagnostics: dict | None = None

    def fresh(self) -> "ExpressionElasticNetCandidate":
        # The immutable source store is intentionally shared; fitted feature,
        # scaler and model state are always newly allocated.
        return type(self)(
            source_store=self.source_store,
            states=tuple(self.states),
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            max_genes=self.max_genes,
            max_iter=self.max_iter,
            tol=self.tol,
            selection=self.selection,
            name=self.name,
        )

    def fit(self, pairs: pd.DataFrame) -> "ExpressionElasticNetCandidate":
        started = time.perf_counter()
        patient_ids = pairs.patient_id.astype(str).tolist()
        x = self.source_store.rows(patient_ids)
        source_columns = [f"source__{state}" for state in self.states]
        target_columns = [f"target__{state}" for state in self.states]
        missing = [column for column in source_columns + target_columns if column not in pairs.columns]
        if missing:
            raise KeyError(f"ExpressionElasticNet.fit missing columns: {missing}")
        source = pairs[source_columns].to_numpy(np.float64)
        target = pairs[target_columns].to_numpy(np.float64)
        variances = np.var(x, axis=0, ddof=0)
        order = sorted(range(x.shape[1]), key=lambda j: (-float(variances[j]), self.source_store.gene_ids[j], j))
        selected = np.asarray(order[: min(self.max_genes, x.shape[1])], dtype=np.int64)
        scaler = StandardScaler(with_mean=True, with_std=True)
        z = scaler.fit_transform(x[:, selected])
        model = MultiTaskElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_intercept=True,
            max_iter=self.max_iter,
            tol=self.tol,
            selection=self.selection,
            random_state=None,
        )
        caught: list[warnings.WarningMessage]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(z, target - source)
        convergence_messages = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
        selected_genes = [self.source_store.gene_ids[j] for j in selected]
        scaler_material = np.concatenate([scaler.mean_, scaler.scale_]).astype(np.float64).tobytes()
        self._selected_indices = selected
        self._scaler = scaler
        self._model = model
        self._diagnostics = {
            "source_store_content_sha256": self.source_store.content_sha256(),
            "training_patient_count": len(set(patient_ids)),
            "training_patient_hash": hashlib.sha256("\0".join(sorted(set(patient_ids))).encode()).hexdigest(),
            "selected_gene_count": len(selected_genes),
            "selected_gene_list_sha256": hashlib.sha256("\0".join(selected_genes).encode()).hexdigest(),
            "scaler_sha256": hashlib.sha256(scaler_material).hexdigest(),
            "iterations": np.asarray(model.n_iter_).astype(int).tolist(),
            "convergence_state": "CONVERGED" if not convergence_messages else "CONVERGENCE_WARNING",
            "convergence_messages": convergence_messages,
            "runtime_seconds": time.perf_counter() - started,
            "peak_rss_bytes": psutil.Process().memory_info().rss,
            "parameters": {
                "alpha": self.alpha,
                "l1_ratio": self.l1_ratio,
                "max_genes": self.max_genes,
                "max_iter": self.max_iter,
                "tol": self.tol,
                "selection": self.selection,
            },
        }
        return self

    def predict(self, pairs: pd.DataFrame) -> np.ndarray:
        if self._selected_indices is None or self._scaler is None or self._model is None:
            raise RuntimeError("ExpressionElasticNet candidate not fitted")
        source_columns = [f"source__{state}" for state in self.states]
        missing = [column for column in source_columns if column not in pairs.columns]
        if missing:
            raise KeyError(f"ExpressionElasticNet.predict missing source columns: {missing}")
        x = self.source_store.rows(pairs.patient_id.astype(str).tolist())
        z = self._scaler.transform(x[:, self._selected_indices])
        source = pairs[source_columns].to_numpy(np.float64)
        prediction = euclidean_simplex_projection(source + self._model.predict(z))
        return self.validate_predictions(prediction, len(pairs), atol=1e-12)

    def candidate_diagnostics(self) -> dict:
        if self._diagnostics is None:
            raise RuntimeError("ExpressionElasticNet candidate not fitted")
        return json.loads(json.dumps(self._diagnostics))

