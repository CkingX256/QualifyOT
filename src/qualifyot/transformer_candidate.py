from __future__ import annotations

"""Graph-constrained residual Transformer for QualifyOT.

This module deliberately exposes one primary Transformer methodology.

The Transformer does *not* replace QualifyOT's statistical evidence engine and
it does not create a second qualification rule.  It only replaces the linear
edge-flow law inside a frozen GraphFlow candidate:

    source composition + time
        -> GraphFlow ridge anchor
        -> small state-token Transformer residual
        -> non-negative flows on the *same frozen graph*
        -> simplex-feasible target composition
        -> ordinary QualifyOT Movement–Utility–Retention IUT.

The design is intentionally conservative for small independent-patient cohorts:

* the graph is fixed before held-out scoring;
* the ridge GraphFlow prediction is the exact zero-residual anchor;
* the neural residual is bounded by a fraction of source-state mass;
* training is patient-balanced and uses only training targets;
* prediction never reads target columns;
* final scientific influence is still determined by the outer QualifyOT
  evidence engine, not by neural-network training loss.

Legacy experimental Transformer classes from the previous development release
remain available in :mod:`qualifyot.legacy_transformer` for reproducibility,
but they are no longer the recommended API.
"""

from dataclasses import dataclass
import random
from typing import Iterable

import numpy as np
import pandas as pd

from .candidate_api import CandidatePredictor, GraphFlowCandidate
from .model import (
    feature_matrix,
    source_feature_matrix,
    incidence,
    patient_sample_weights,
    project_flow,
)

try:  # optional dependency: core QualifyOT remains torch-free
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError(
            "GraphResidualTransformerCandidate requires PyTorch. "
            "Install with `pip install .[transformer]` or install torch>=2.2."
        )


def _seed_everything(seed: int) -> None:
    """Deterministic CPU setup used by the tiny-network implementation."""
    _require_torch()
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:  # pragma: no cover - old torch compatibility
        torch.use_deterministic_algorithms(True)


def _validate_graph(states: list[str], edges: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    edges = [tuple(e) for e in edges]
    if not edges:
        raise ValueError("graph-transformer requires at least one pre-specified directed edge")
    known = set(states)
    if len(set(edges)) != len(edges):
        raise ValueError("duplicate directed edges are not permitted")
    for a, b in edges:
        if a not in known or b not in known:
            raise KeyError(f"edge {(a, b)} uses a state outside the configured state space")
        if a == b:
            raise ValueError("self-loops are not permitted")
    return edges


def _clr(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Stable centered log-ratio features; used only as predictor-side inputs."""
    x = np.asarray(x, float)
    z = np.log(np.clip(x, eps, None))
    return z - z.mean(axis=1, keepdims=True)


if nn is not None:
    class _GraphResidualTransformer(nn.Module):
        """State encoder plus a shared edge-residual decoder.

        There is intentionally no separate graph-attention mode.  Global state
        attention models cross-state context; structural restriction occurs at
        the decoder, which can emit flow only on the frozen directed edges.
        This keeps the statistical meaning of the graph transparent.
        """

        def __init__(
            self,
            *,
            n_states: int,
            d_model: int,
            nhead: int,
            num_layers: int,
            dim_feedforward: int,
            dropout: float,
        ):
            super().__init__()
            # token = source abundance, CLR(source), standardized elapsed time
            self.token_proj = nn.Linear(3, d_model)
            self.state_embedding = nn.Embedding(n_states, d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=num_layers, enable_nested_tensor=False
            )
            self.norm = nn.LayerNorm(d_model)
            # Shared edge law: tail context + head context + anchored edge flow.
            self.edge_head = nn.Sequential(
                nn.Linear(2 * d_model + 1, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )
            # Exact GraphFlow anchor at initialization.
            nn.init.zeros_(self.edge_head[-1].weight)
            nn.init.zeros_(self.edge_head[-1].bias)

        def forward(
            self,
            token_x: "torch.Tensor",
            source: "torch.Tensor",
            base_flow: "torch.Tensor",
            tails: "torch.Tensor",
            heads: "torch.Tensor",
            residual_fraction: float,
            incidence_t: "torch.Tensor",
            eps: float = 1e-7,
        ):
            b, k, _ = token_x.shape
            ids = torch.arange(k, device=token_x.device).unsqueeze(0).expand(b, -1)
            z = self.norm(self.encoder(self.token_proj(token_x) + self.state_embedding(ids)))

            edge_x = torch.cat(
                [z[:, tails, :], z[:, heads, :], torch.log1p(base_flow).unsqueeze(-1)],
                dim=-1,
            )
            raw = self.edge_head(edge_x).squeeze(-1)

            # Scale-free bounded correction: an edge can change by at most a
            # fixed fraction of the mass currently available at its tail state.
            tail_mass = source[:, tails]
            correction = float(residual_fraction) * tail_mass * torch.tanh(raw)
            flow = torch.relu(base_flow + correction)

            # Differentiable global feasibility scaling.  Incidence columns sum
            # to zero, so scaling preserves total mass.  It only activates when
            # proposed outgoing flow would make a state negative.
            delta = flow @ incidence_t
            negative = delta < 0
            inf = torch.full_like(delta, float("inf"))
            ratio = torch.where(
                negative,
                torch.clamp(source - eps, min=0.0) / torch.clamp(-delta, min=eps),
                inf,
            )
            scale = torch.minimum(
                torch.ones(b, device=source.device, dtype=source.dtype),
                ratio.min(dim=1).values,
            )
            scale = torch.clamp(scale, min=0.0, max=1.0)
            effective_flow = flow * scale.unsqueeze(1)
            pred = source + effective_flow @ incidence_t
            pred = torch.clamp(pred, min=0.0)
            pred = pred / torch.clamp(pred.sum(dim=1, keepdim=True), min=eps)
            return pred, effective_flow, correction, scale


@dataclass
class GraphResidualTransformerCandidate(CandidatePredictor):
    """Unified nonlinear structured candidate for QualifyOT.

    The frozen directed graph defines *where* mass may move.  GraphFlow supplies
    a low-variance linear edge-flow anchor.  A tiny Transformer learns only a
    bounded nonlinear residual in the edge-flow law.  The model is trained
    end-to-end against target compositions rather than against an arbitrary
    residual-flow decomposition, which removes a previous pseudo-label layer.

    This is the recommended Transformer API.  It remains only a candidate
    generator: the outer QualifyOT IUT independently decides whether its
    movement, direct utility and retained influence are supported.
    """

    states: tuple | list
    edges: tuple | list | None = None
    alpha: float = 1.0
    d_model: int = 8
    nhead: int = 2
    num_layers: int = 1
    dim_feedforward: int = 16
    dropout: float = 0.0
    residual_fraction: float = 0.20
    epochs: int = 20
    learning_rate: float = 2e-3
    weight_decay: float = 1e-3
    anchor_strength: float = 0.15
    grad_clip: float = 1.0
    patient_balanced: bool = True
    seed: int = 20260831
    name: str = "GraphResidualTransformer"

    _base: object = None
    _model: object = None
    _B: np.ndarray | None = None
    _tails: np.ndarray | None = None
    _heads: np.ndarray | None = None
    _dt_median: float | None = None
    _dt_scale: float | None = None
    _diagnostics: dict | None = None

    def __post_init__(self):
        self.states = list(self.states)
        if len(self.states) < 2:
            raise ValueError("graph-transformer requires at least two states")
        self.edges = _validate_graph(self.states, self.edges)
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self.d_model <= 0 or self.nhead <= 0 or self.d_model % self.nhead != 0:
            raise ValueError("d_model must be positive and divisible by nhead")
        if self.num_layers < 1 or self.dim_feedforward < 1:
            raise ValueError("num_layers and dim_feedforward must be positive")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0,1)")
        if not (0.0 <= self.residual_fraction <= 1.0):
            raise ValueError("residual_fraction must be in [0,1]")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.anchor_strength < 0:
            raise ValueError("optimizer and anchor hyperparameters must be non-negative")

    def fresh(self):
        return type(self)(
            states=list(self.states),
            edges=list(self.edges),
            alpha=self.alpha,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            residual_fraction=self.residual_fraction,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            anchor_strength=self.anchor_strength,
            grad_clip=self.grad_clip,
            patient_balanced=self.patient_balanced,
            seed=self.seed,
            name=self.name,
        )

    def _tokens(self, pairs: pd.DataFrame, *, fit: bool):
        X, S = source_feature_matrix(pairs, states=self.states)
        dt = np.asarray(X[:, -1], float)
        if fit:
            med = float(np.nanmedian(dt)) if len(dt) else 0.0
            sd = float(np.nanstd(dt)) if len(dt) else 1.0
            self._dt_median = med if np.isfinite(med) else 0.0
            self._dt_scale = sd if np.isfinite(sd) and sd > 1e-8 else 1.0
        if self._dt_median is None or self._dt_scale is None:
            raise RuntimeError("candidate not fitted")
        zdt = ((dt - self._dt_median) / self._dt_scale)[:, None]
        zdt = np.repeat(zdt, len(self.states), axis=1)
        tok = np.stack([S, _clr(S), zdt], axis=2).astype(np.float32)
        return tok, X, S

    def _project_anchor_flows(self, X: np.ndarray, S: np.ndarray) -> np.ndarray:
        raw = np.asarray(self._base._model.predict_flows(X), float)
        projected = []
        for s, f in zip(S, raw):
            fp, _ = project_flow(s, f, self._B)
            projected.append(fp)
        return np.asarray(projected, float)

    def _torch_decode(self, token_x, source, base_flow):
        idx = {s: i for i, s in enumerate(self.states)}
        tails = torch.tensor([idx[a] for a, _ in self.edges], dtype=torch.long)
        heads = torch.tensor([idx[b] for _, b in self.edges], dtype=torch.long)
        Bt = torch.tensor(self._B.T, dtype=torch.float32)
        return self._model(
            torch.tensor(token_x, dtype=torch.float32),
            torch.tensor(source, dtype=torch.float32),
            torch.tensor(base_flow, dtype=torch.float32),
            tails,
            heads,
            self.residual_fraction,
            Bt,
        )

    def fit(self, pairs: pd.DataFrame):
        _require_torch()
        if "patient_id" not in pairs.columns:
            raise KeyError("patient_id is required for patient-balanced fitting")
        _, S, T = feature_matrix(pairs, states=self.states)
        patients = pairs.patient_id.astype(str).to_numpy()
        n_pat = int(np.unique(patients).size)
        if n_pat < 2:
            raise ValueError("at least two independent training patients are required")

        _seed_everything(self.seed)
        self._B = incidence(self.states, self.edges)
        idx = {s: i for i, s in enumerate(self.states)}
        self._tails = np.asarray([idx[a] for a, _ in self.edges], int)
        self._heads = np.asarray([idx[b] for _, b in self.edges], int)

        # Low-variance structured anchor.  All target use here is inside fit().
        self._base = GraphFlowCandidate(
            alpha=self.alpha,
            patient_balanced=self.patient_balanced,
            states=self.states,
            edges=self.edges,
            name="GraphFlowAnchor",
        ).fit(pairs)

        token_x, X, S2 = self._tokens(pairs, fit=True)
        base_flow = self._project_anchor_flows(X, S2)
        base_pred = np.vstack([s + self._B @ f for s, f in zip(S2, base_flow)])
        base_pred = np.maximum(base_pred, 0.0)
        base_pred /= base_pred.sum(axis=1, keepdims=True)

        model = _GraphResidualTransformer(
            n_states=len(self.states),
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )
        self._model = model
        opt = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        tx = torch.tensor(token_x, dtype=torch.float32)
        ts = torch.tensor(S2, dtype=torch.float32)
        tb = torch.tensor(base_flow, dtype=torch.float32)
        ty = torch.tensor(np.asarray(T, np.float32), dtype=torch.float32)
        tails = torch.tensor(self._tails, dtype=torch.long)
        heads = torch.tensor(self._heads, dtype=torch.long)
        Bt = torch.tensor(self._B.T, dtype=torch.float32)
        if self.patient_balanced:
            sw = patient_sample_weights(patients)
        else:
            sw = np.ones(len(pairs), float)
        tw = torch.tensor(sw / np.mean(sw), dtype=torch.float32)

        losses: list[float] = []
        model.train()
        for _ in range(int(self.epochs)):
            opt.zero_grad(set_to_none=True)
            pred, eff_flow, correction, _ = model(
                tx, ts, tb, tails, heads, self.residual_fraction, Bt
            )
            # Align training with the patient-level MAE estimand while using a
            # smooth local loss.  Anchor penalty prevents unnecessary neural
            # departure from GraphFlow in tiny cohorts.
            data_loss = torch.nn.functional.smooth_l1_loss(
                pred, ty, reduction="none", beta=0.02
            ).mean(dim=1)
            capacity = torch.clamp(ts[:, tails], min=1e-4)
            anchor = torch.square((eff_flow - tb) / capacity).mean(dim=1)
            per = data_loss + self.anchor_strength * anchor
            loss = (tw * per).sum() / tw.sum()
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite graph-transformer training loss")
            loss.backward()
            if self.grad_clip is not None and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(self.grad_clip))
            opt.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            pred, eff_flow, correction, scale = model(
                tx, ts, tb, tails, heads, self.residual_fraction, Bt
            )
        pred_np = pred.cpu().numpy().astype(float)
        pred_np = np.maximum(pred_np, 0.0)
        pred_np = pred_np / pred_np.sum(axis=1, keepdims=True)
        flow_np = eff_flow.cpu().numpy().astype(float)
        corr_np = correction.cpu().numpy().astype(float)
        scale_np = scale.cpu().numpy().astype(float)
        self.validate_predictions(pred_np, len(pairs))

        w = sw / sw.sum()
        base_mae = float(np.sum(w * np.abs(T - base_pred).mean(axis=1)))
        fused_mae = float(np.sum(w * np.abs(T - pred_np).mean(axis=1)))
        cap = np.clip(S2[:, self._tails], 1e-8, None)
        relative_corr = np.abs(flow_np - base_flow) / cap
        self._diagnostics = {
            "method": "graph-constrained residual Transformer",
            "training_pairs": int(len(pairs)),
            "training_patients": n_pat,
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
            "epochs": int(self.epochs),
            "initial_objective": float(losses[0]),
            "final_objective": float(losses[-1]),
            "graphflow_train_mae": base_mae,
            "fused_train_mae": fused_mae,
            "mean_abs_flow_correction": float(np.mean(np.abs(flow_np - base_flow))),
            "mean_relative_tail_mass_correction": float(np.mean(relative_corr)),
            "max_relative_tail_mass_correction": float(np.max(relative_corr)),
            "feasibility_scaling_fraction": float(np.mean(scale_np < 1.0 - 1e-7)),
            "mean_feasibility_scale": float(np.mean(scale_np)),
            "residual_fraction": float(self.residual_fraction),
            "anchor_strength": float(self.anchor_strength),
            "seed": int(self.seed),
            "patient_balanced": bool(self.patient_balanced),
            # Compatibility keys from the prior experimental implementation.
            "base_flow_mae": base_mae,
            "transformer_flow_mae": fused_mae,
        }
        return self

    def _predict_all(self, pairs: pd.DataFrame):
        _require_torch()
        if self._model is None or self._base is None or self._B is None:
            raise RuntimeError("candidate not fitted")
        token_x, X, S = self._tokens(pairs, fit=False)
        base_flow = self._project_anchor_flows(X, S)
        self._model.eval()
        with torch.no_grad():
            pred, eff_flow, correction, scale = self._torch_decode(token_x, S, base_flow)
        pred_np = pred.cpu().numpy().astype(float)
        pred_np = np.maximum(pred_np, 0.0)
        pred_np = pred_np / pred_np.sum(axis=1, keepdims=True)
        return (
            pred_np,
            eff_flow.cpu().numpy().astype(float),
            correction.cpu().numpy().astype(float),
            scale.cpu().numpy().astype(float),
        )

    def predict_flows(self, pairs: pd.DataFrame) -> np.ndarray:
        """Return effective feasible flows; source + B@flow equals prediction."""
        _, flow, _, _ = self._predict_all(pairs)
        return flow

    def predict(self, pairs: pd.DataFrame) -> np.ndarray:
        pred, _, _, _ = self._predict_all(pairs)
        return self.validate_predictions(pred, len(pairs))

    def candidate_diagnostics(self) -> dict:
        if self._diagnostics is None:
            raise RuntimeError("candidate not fitted")
        return dict(self._diagnostics)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------
# The previous development release exposed three experimental Transformer
# classes.  They remain importable for exact reproduction, but new analyses
# should use GraphResidualTransformerCandidate.  Keeping them outside the main
# implementation prevents the primary methodology from fragmenting into three
# competing neural variants.
from .legacy_transformer import (  # noqa: E402  (intentional late optional import)
    ResidualStateTransformerCandidate,
    GraphAwareTransformerCandidate,
)


class TransformerGraphFlowCandidate(GraphResidualTransformerCandidate):
    """Compatibility name for the former TransformerGraphFlow candidate.

    New code should instantiate :class:`GraphResidualTransformerCandidate`.
    The old ``flow_correction_scale`` keyword is accepted and conservatively
    mapped to a dimensionless tail-mass residual fraction.
    """

    def __init__(self, *args, flow_correction_scale=None, attention_mode=None, **kwargs):
        if flow_correction_scale is not None and "residual_fraction" not in kwargs:
            # Old values were absolute flow corrections (typically <=0.08).
            # A conservative dimensionless mapping preserves bounded behavior.
            kwargs["residual_fraction"] = min(1.0, max(0.0, 2.5 * float(flow_correction_scale)))
        # attention_mode is intentionally ignored: graph restriction now lives
        # at the edge decoder, eliminating redundant graph-mask semantics.
        kwargs.setdefault("name", "TransformerGraphFlow")
        super().__init__(*args, **kwargs)
