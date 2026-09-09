from __future__ import annotations

"""Small-data transformer candidates for QualifyOT.

The transformer is intentionally *not* part of the qualification rule.  It is
only a candidate generator that must pass the same frozen patient-level
Movement–Utility–Retention evidence engine as every other predictor.

Design principles
-----------------
1. Training-only fitting; prediction never reads ``target__<state>`` columns.
2. A low-variance RobustBlend prediction is the anchor.  The transformer learns
   a bounded log-ratio residual rather than an unconstrained target vector.
3. Each cell-state is represented as one token.  Self-attention can therefore
   learn cross-state interactions while the output remains on the simplex.
4. Optional graph-aware attention masks can restrict message passing to a
   pre-specified graph.  This is a predictive inductive bias, not a causal claim.
5. PyTorch is optional.  Importing QualifyOT does not require torch; fitting this
   candidate raises a clear ImportError when the optional dependency is absent.
"""

from dataclasses import dataclass
from typing import Iterable
import math
import random
import numpy as np
import pandas as pd

from .candidate_api import CandidatePredictor
from .candidates import RobustBlendCandidate
from .model import feature_matrix, source_feature_matrix, patient_sample_weights

try:  # optional dependency
    import torch
    from torch import nn
except Exception:  # pragma: no cover - tested through dependency guard
    torch = None
    nn = None


def _require_torch():
    if torch is None or nn is None:
        raise ImportError(
            "ResidualStateTransformerCandidate requires PyTorch. "
            "Install QualifyOT with `pip install .[transformer]` or install torch>=2.2."
        )


def _seed_everything(seed: int) -> None:
    _require_torch()
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    # CPU operations used below are deterministic.  warn_only avoids hard
    # failures on future backends while preserving reproducible seeds.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:  # older torch
        torch.use_deterministic_algorithms(True)


def _graph_mask(states: list[str], edges: Iterable[tuple[str, str]] | None, mode: str):
    """Return an additive Transformer attention mask.

    PyTorch masks are indexed [query_state, key_state].  Under ``directed``
    mode, destination state ``b`` may attend to its pre-specified predecessor
    ``a`` for edge a->b, plus every state attends to itself.  ``undirected``
    permits both directions.  ``full`` returns None.
    """
    _require_torch()
    mode = str(mode).lower()
    if mode in {"full", "none"}:
        return None
    if mode not in {"directed", "undirected"}:
        raise ValueError("attention_mode must be 'full', 'directed', or 'undirected'")
    if edges is None:
        raise ValueError(f"attention_mode={mode!r} requires a pre-specified edge list")
    idx = {s: i for i, s in enumerate(states)}
    allow = np.eye(len(states), dtype=bool)
    for a, b in edges:
        if a not in idx or b not in idx:
            raise KeyError(f"edge {(a,b)} uses a state outside the configured state space")
        if a == b:
            raise ValueError("self-loops are not permitted in graph-aware attention")
        allow[idx[b], idx[a]] = True
        if mode == "undirected":
            allow[idx[a], idx[b]] = True
    mask = np.where(allow, 0.0, -np.inf).astype(np.float32)
    return torch.tensor(mask, dtype=torch.float32)


if nn is not None:
    class _ResidualStateTransformer(nn.Module):
        def __init__(self, n_states: int, token_features: int, d_model: int, nhead: int,
                     num_layers: int, dim_feedforward: int, dropout: float,
                     residual_scale: float):
            super().__init__()
            self.n_states = int(n_states)
            self.residual_scale = float(residual_scale)
            self.token_proj = nn.Linear(token_features, d_model)
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
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d_model)
            self.out = nn.Linear(d_model, 1)
            # Exact baseline anchor at initialization: before optimization the
            # network correction is identically zero.
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

        def forward(self, token_x, base_prob, attention_mask=None):
            b, k, _ = token_x.shape
            ids = torch.arange(k, device=token_x.device).unsqueeze(0).expand(b, -1)
            z = self.token_proj(token_x) + self.state_embedding(ids)
            z = self.encoder(z, mask=attention_mask)
            raw = self.out(self.norm(z)).squeeze(-1)
            # Remove softmax translation non-identifiability and bound the
            # correction so the tiny network cannot make arbitrarily large
            # log-ratio moves on a tiny cohort.
            raw = raw - raw.mean(dim=1, keepdim=True)
            corr = self.residual_scale * torch.tanh(raw)
            logits = torch.log(torch.clamp(base_prob, min=1e-8)) + corr
            return torch.softmax(logits, dim=1), corr


@dataclass
class ResidualStateTransformerCandidate(CandidatePredictor):
    """Tiny residual transformer anchored to :class:`RobustBlendCandidate`.

    The model treats states as tokens and predicts a bounded correction to the
    log-composition produced by RobustBlend.  It is deliberately small because
    QualifyOT targets cohorts with few independent patients.  All hyperparameters
    are fixed before held-out scoring; the evidence layer, not training loss,
    decides whether the candidate earns predictive influence.
    """

    states: tuple | list
    d_model: int = 16
    nhead: int = 2
    num_layers: int = 1
    dim_feedforward: int = 32
    dropout: float = 0.0
    residual_scale: float = 0.35
    epochs: int = 80
    learning_rate: float = 2e-3
    weight_decay: float = 1e-3
    anchor_strength: float = 0.25
    mae_strength: float = 0.25
    grad_clip: float = 1.0
    patient_balanced: bool = True
    seed: int = 20260830
    attention_mode: str = "full"
    edges: tuple | list | None = None
    base_alpha: float = 10.0
    base_blend: float = 0.5
    name: str = "ResidualStateTransformer"

    _model: object = None
    _base: object = None
    _dt_median: float | None = None
    _dt_scale: float | None = None
    _attention_mask: object = None
    _diagnostics: dict | None = None

    def __post_init__(self):
        self.states = list(self.states)
        if len(self.states) < 2:
            raise ValueError("transformer candidate requires at least two states")
        if self.d_model <= 0 or self.nhead <= 0 or self.d_model % self.nhead != 0:
            raise ValueError("d_model must be positive and divisible by nhead")
        if self.num_layers < 1 or self.dim_feedforward < 1:
            raise ValueError("num_layers and dim_feedforward must be positive")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0,1)")
        if self.residual_scale < 0:
            raise ValueError("residual_scale must be non-negative")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer hyperparameters")
        if self.anchor_strength < 0 or self.mae_strength < 0:
            raise ValueError("loss strengths must be non-negative")

    def fresh(self):
        return type(self)(
            states=list(self.states), d_model=self.d_model, nhead=self.nhead,
            num_layers=self.num_layers, dim_feedforward=self.dim_feedforward,
            dropout=self.dropout, residual_scale=self.residual_scale,
            epochs=self.epochs, learning_rate=self.learning_rate,
            weight_decay=self.weight_decay, anchor_strength=self.anchor_strength,
            mae_strength=self.mae_strength, grad_clip=self.grad_clip,
            patient_balanced=self.patient_balanced, seed=self.seed,
            attention_mode=self.attention_mode,
            edges=None if self.edges is None else [tuple(e) for e in self.edges],
            base_alpha=self.base_alpha, base_blend=self.base_blend, name=self.name,
        )

    def _token_features(self, pairs: pd.DataFrame, base_pred: np.ndarray, fit: bool):
        X, S = source_feature_matrix(pairs, states=self.states)
        dt = np.asarray(X[:, -1], float)
        if fit:
            med = float(np.nanmedian(dt)) if len(dt) else 0.0
            sd = float(np.nanstd(dt)) if len(dt) else 1.0
            self._dt_median = med if np.isfinite(med) else 0.0
            self._dt_scale = sd if np.isfinite(sd) and sd > 1e-8 else 1.0
        if self._dt_median is None or self._dt_scale is None:
            raise RuntimeError("transformer candidate not fitted")
        zdt = ((dt - self._dt_median) / self._dt_scale)[:, None]
        zdt = np.repeat(zdt, len(self.states), axis=1)
        # Each state token sees local source abundance, anchored prediction,
        # anchored delta and a globally shared elapsed-time feature.
        tok = np.stack([S, base_pred, base_pred - S, zdt], axis=2)
        return tok.astype(np.float32), S

    def fit(self, pairs: pd.DataFrame):
        _require_torch()
        if "patient_id" not in pairs.columns:
            raise KeyError("patient_id is required for patient-balanced transformer fitting")
        _, _, target = feature_matrix(pairs, states=self.states)
        if len(pairs) < 2:
            raise ValueError("at least two training rows are required")
        n_pat = int(pairs.patient_id.astype(str).nunique())
        if n_pat < 2:
            raise ValueError("at least two independent training patients are required")

        _seed_everything(self.seed)
        self._base = RobustBlendCandidate(
            states=list(self.states), alpha=self.base_alpha, blend=self.base_blend,
            name="TransformerAnchor_RobustBlend",
        ).fit(pairs)
        base = self._base.predict(pairs)
        token_x, _ = self._token_features(pairs, base, fit=True)
        self._attention_mask = _graph_mask(self.states, self.edges, self.attention_mode)

        model = _ResidualStateTransformer(
            n_states=len(self.states), token_features=token_x.shape[2],
            d_model=self.d_model, nhead=self.nhead, num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward, dropout=self.dropout,
            residual_scale=self.residual_scale,
        )
        opt = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        tx = torch.tensor(token_x, dtype=torch.float32)
        tb = torch.tensor(base, dtype=torch.float32)
        ty = torch.tensor(np.asarray(target, np.float32), dtype=torch.float32)
        if self.patient_balanced:
            sw = patient_sample_weights(pairs.patient_id.astype(str).to_numpy())
        else:
            sw = np.ones(len(pairs), dtype=float)
        tw = torch.tensor(sw / np.mean(sw), dtype=torch.float32)

        losses = []
        model.train()
        for _ in range(int(self.epochs)):
            opt.zero_grad(set_to_none=True)
            pred, corr = model(tx, tb, self._attention_mask)
            log_score = -(ty * torch.log(torch.clamp(pred, min=1e-8))).sum(dim=1)
            mae = torch.abs(ty - pred).mean(dim=1)
            # Symmetric local anchor: penalize unnecessary departure from the
            # low-variance baseline, especially important in small-n cohorts.
            anchor = torch.square(pred - tb).mean(dim=1)
            per = log_score + self.mae_strength * mae + self.anchor_strength * anchor
            loss = (tw * per).sum() / tw.sum()
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite transformer training loss")
            loss.backward()
            if self.grad_clip is not None and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(self.grad_clip))
            opt.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            pred, corr = model(tx, tb, self._attention_mask)
        pred_np = pred.cpu().numpy().astype(float)
        pred_np = np.maximum(pred_np, 0.0); pred_np = pred_np / pred_np.sum(axis=1, keepdims=True)
        corr_np = corr.cpu().numpy().astype(float)
        self.validate_predictions(pred_np, len(pairs))
        w = sw / sw.sum()
        baseline_mae = float(np.sum(w * np.abs(target - base).mean(axis=1)))
        transformer_mae = float(np.sum(w * np.abs(target - pred_np).mean(axis=1)))
        params = int(sum(p.numel() for p in model.parameters()))
        self._diagnostics = {
            "training_pairs": int(len(pairs)),
            "training_patients": n_pat,
            "parameter_count": params,
            "epochs": int(self.epochs),
            "final_objective": float(losses[-1]),
            "initial_objective": float(losses[0]),
            "baseline_train_mae": baseline_mae,
            "transformer_train_mae": transformer_mae,
            "mean_abs_logratio_correction": float(np.mean(np.abs(corr_np))),
            "max_abs_logratio_correction": float(np.max(np.abs(corr_np))),
            "attention_mode": self.attention_mode,
            "residual_scale": float(self.residual_scale),
            "seed": int(self.seed),
            "patient_balanced": bool(self.patient_balanced),
        }
        self._model = model
        return self

    def predict(self, pairs: pd.DataFrame):
        _require_torch()
        if self._model is None or self._base is None:
            raise RuntimeError("candidate not fitted")
        base = self._base.predict(pairs)
        token_x, _ = self._token_features(pairs, base, fit=False)
        self._model.eval()
        with torch.no_grad():
            pred, _ = self._model(
                torch.tensor(token_x, dtype=torch.float32),
                torch.tensor(base, dtype=torch.float32),
                self._attention_mask,
            )
        out = pred.cpu().numpy().astype(float)
        out = np.maximum(out, 0.0); out = out / out.sum(axis=1, keepdims=True)
        return self.validate_predictions(out, len(pairs))

    def candidate_diagnostics(self) -> dict:
        if self._diagnostics is None:
            raise RuntimeError("candidate not fitted")
        return dict(self._diagnostics)


@dataclass
class GraphAwareTransformerCandidate(ResidualStateTransformerCandidate):
    """Residual state transformer with a pre-specified graph attention mask."""

    attention_mode: str = "directed"
    name: str = "GraphAwareTransformer"

if nn is not None:
    class _GraphFlowResidualTransformer(nn.Module):
        """State encoder plus edge decoder for bounded corrections to ridge flow."""
        def __init__(self, n_states: int, n_edges: int, d_model: int, nhead: int,
                     num_layers: int, dim_feedforward: int, dropout: float,
                     flow_correction_scale: float):
            super().__init__()
            self.flow_correction_scale=float(flow_correction_scale)
            self.token_proj=nn.Linear(2,d_model)  # source abundance + standardized dt
            self.state_embedding=nn.Embedding(n_states,d_model)
            layer=nn.TransformerEncoderLayer(
                d_model=d_model,nhead=nhead,dim_feedforward=dim_feedforward,
                dropout=dropout,activation='gelu',batch_first=True,norm_first=True,
            )
            self.encoder=nn.TransformerEncoder(layer,num_layers=num_layers,enable_nested_tensor=False)
            self.edge_embedding=nn.Embedding(n_edges,d_model)
            self.edge_mlp=nn.Sequential(
                nn.Linear(3*d_model,d_model),nn.GELU(),nn.Linear(d_model,1)
            )
            nn.init.zeros_(self.edge_mlp[-1].weight); nn.init.zeros_(self.edge_mlp[-1].bias)

        def forward(self, token_x, base_flow, tails, heads, attention_mask=None):
            b,k,_=token_x.shape
            ids=torch.arange(k,device=token_x.device).unsqueeze(0).expand(b,-1)
            z=self.token_proj(token_x)+self.state_embedding(ids)
            z=self.encoder(z,mask=attention_mask)
            eids=torch.arange(len(tails),device=token_x.device).unsqueeze(0).expand(b,-1)
            et=self.edge_embedding(eids)
            zr=torch.cat([z[:,tails,:],z[:,heads,:],et],dim=-1)
            raw=self.edge_mlp(zr).squeeze(-1)
            corr=self.flow_correction_scale*torch.tanh(raw)
            flow=torch.relu(base_flow+corr)
            return flow,corr


@dataclass
class TransformerGraphFlowCandidate(CandidatePredictor):
    """Nonlinear graph-constrained candidate with a tiny Transformer flow head.

    The original GraphFlow ridge model remains the anchor.  A state-token
    Transformer learns only a bounded correction to its non-negative edge-flow
    predictions.  Final target compositions are still produced through the
    frozen incidence matrix and the same simplex-feasible flow projection.

    This class therefore changes the *predictive head* while preserving the
    pre-specified graph semantics.  It must be qualified by the same outer
    patient-level evidence engine; a better training fit is not itself evidence.
    """
    states: tuple | list
    edges: tuple | list | None = None
    alpha: float = 1.0
    d_model: int = 16
    nhead: int = 2
    num_layers: int = 1
    dim_feedforward: int = 32
    dropout: float = 0.0
    flow_correction_scale: float = 0.08
    epochs: int = 80
    learning_rate: float = 2e-3
    weight_decay: float = 1e-3
    anchor_strength: float = 0.20
    grad_clip: float = 1.0
    patient_balanced: bool = True
    seed: int = 20260830
    attention_mode: str = 'directed'
    name: str = 'TransformerGraphFlow'

    _base: object = None
    _model: object = None
    _dt_median: float | None = None
    _dt_scale: float | None = None
    _attention_mask: object = None
    _B: np.ndarray | None = None
    _diagnostics: dict | None = None

    def __post_init__(self):
        self.states=list(self.states); self.edges=[] if self.edges is None else [tuple(e) for e in self.edges]
        if not self.edges: raise ValueError('TransformerGraphFlow requires at least one pre-specified edge')
        if self.d_model<=0 or self.nhead<=0 or self.d_model%self.nhead!=0:
            raise ValueError('d_model must be positive and divisible by nhead')
        if self.flow_correction_scale<0: raise ValueError('flow_correction_scale must be non-negative')
        if self.epochs<1: raise ValueError('epochs must be positive')

    def fresh(self):
        return type(self)(states=list(self.states),edges=list(self.edges),alpha=self.alpha,
            d_model=self.d_model,nhead=self.nhead,num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,dropout=self.dropout,
            flow_correction_scale=self.flow_correction_scale,epochs=self.epochs,
            learning_rate=self.learning_rate,weight_decay=self.weight_decay,
            anchor_strength=self.anchor_strength,grad_clip=self.grad_clip,
            patient_balanced=self.patient_balanced,seed=self.seed,
            attention_mode=self.attention_mode,name=self.name)

    def _tokens(self,pairs,fit=False):
        X,S=source_feature_matrix(pairs,states=self.states); dt=np.asarray(X[:,-1],float)
        if fit:
            med=float(np.nanmedian(dt)) if len(dt) else 0.0; sd=float(np.nanstd(dt)) if len(dt) else 1.0
            self._dt_median=med if np.isfinite(med) else 0.0
            self._dt_scale=sd if np.isfinite(sd) and sd>1e-8 else 1.0
        if self._dt_median is None or self._dt_scale is None: raise RuntimeError('candidate not fitted')
        zdt=((dt-self._dt_median)/self._dt_scale)[:,None]
        tok=np.stack([S,np.repeat(zdt,len(self.states),axis=1)],axis=2).astype(np.float32)
        return tok,X,S

    def fit(self,pairs: pd.DataFrame):
        _require_torch()
        from .candidate_api import GraphFlowCandidate
        from .model import incidence, residual_flow
        if 'patient_id' not in pairs.columns: raise KeyError('patient_id is required')
        X,S,T=feature_matrix(pairs,states=self.states)
        n_pat=int(pairs.patient_id.astype(str).nunique())
        if n_pat<2: raise ValueError('at least two independent training patients are required')
        _seed_everything(self.seed)
        self._B=incidence(self.states,self.edges)
        self._base=GraphFlowCandidate(alpha=self.alpha,patient_balanced=self.patient_balanced,
                                      states=self.states,edges=self.edges,name='TransformerGraphFlowAnchor').fit(pairs)
        base_flow=np.asarray(self._base._model.predict_flows(X),float)
        flow_labels=np.vstack([residual_flow(s,t,Bmat=self._B)[0] for s,t in zip(S,T)])
        token_x,_,_=self._tokens(pairs,fit=True)
        self._attention_mask=_graph_mask(self.states,self.edges,self.attention_mode)
        idx={s:i for i,s in enumerate(self.states)}
        tails=torch.tensor([idx[a] for a,b in self.edges],dtype=torch.long)
        heads=torch.tensor([idx[b] for a,b in self.edges],dtype=torch.long)
        model=_GraphFlowResidualTransformer(len(self.states),len(self.edges),self.d_model,self.nhead,
                                             self.num_layers,self.dim_feedforward,self.dropout,
                                             self.flow_correction_scale)
        opt=torch.optim.AdamW(model.parameters(),lr=self.learning_rate,weight_decay=self.weight_decay)
        tx=torch.tensor(token_x,dtype=torch.float32); tb=torch.tensor(base_flow,dtype=torch.float32)
        ty=torch.tensor(flow_labels,dtype=torch.float32)
        sw=patient_sample_weights(pairs.patient_id.astype(str).to_numpy()) if self.patient_balanced else np.ones(len(pairs))
        tw=torch.tensor(sw/np.mean(sw),dtype=torch.float32)
        losses=[]; model.train()
        for _ in range(int(self.epochs)):
            opt.zero_grad(set_to_none=True)
            pred,corr=model(tx,tb,tails,heads,self._attention_mask)
            fitloss=torch.nn.functional.smooth_l1_loss(pred,ty,reduction='none').mean(dim=1)
            anchor=torch.square(pred-tb).mean(dim=1)
            per=fitloss+self.anchor_strength*anchor
            loss=(tw*per).sum()/tw.sum()
            if not torch.isfinite(loss): raise RuntimeError('non-finite TransformerGraphFlow training loss')
            loss.backward()
            if self.grad_clip is not None and self.grad_clip>0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),float(self.grad_clip))
            opt.step(); losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad(): pred,corr=model(tx,tb,tails,heads,self._attention_mask)
        pf=pred.cpu().numpy().astype(float); cf=corr.cpu().numpy().astype(float)
        self._model=model
        self._diagnostics={
            'training_pairs':int(len(pairs)),'training_patients':n_pat,
            'parameter_count':int(sum(p.numel() for p in model.parameters())),
            'epochs':int(self.epochs),'initial_objective':losses[0],'final_objective':losses[-1],
            'base_flow_mae':float(np.abs(base_flow-flow_labels).mean()),
            'transformer_flow_mae':float(np.abs(pf-flow_labels).mean()),
            'mean_abs_flow_correction':float(np.abs(cf).mean()),
            'max_abs_flow_correction':float(np.abs(cf).max()),
            'attention_mode':self.attention_mode,'flow_correction_scale':float(self.flow_correction_scale),
            'seed':int(self.seed),'patient_balanced':bool(self.patient_balanced),
        }
        return self

    def predict_flows(self,pairs: pd.DataFrame) -> np.ndarray:
        _require_torch()
        if self._model is None or self._base is None: raise RuntimeError('candidate not fitted')
        token_x,X,_=self._tokens(pairs,fit=False)
        base_flow=np.asarray(self._base._model.predict_flows(X),float)
        idx={s:i for i,s in enumerate(self.states)}
        tails=torch.tensor([idx[a] for a,b in self.edges],dtype=torch.long)
        heads=torch.tensor([idx[b] for a,b in self.edges],dtype=torch.long)
        self._model.eval()
        with torch.no_grad():
            pred,_=self._model(torch.tensor(token_x,dtype=torch.float32),torch.tensor(base_flow,dtype=torch.float32),
                               tails,heads,self._attention_mask)
        return pred.cpu().numpy().astype(float)

    def predict(self,pairs: pd.DataFrame):
        from .model import project_flow
        _,S=source_feature_matrix(pairs,states=self.states)
        F=self.predict_flows(pairs)
        out=np.vstack([project_flow(s,f,self._B)[1] for s,f in zip(S,F)])
        return self.validate_predictions(out,len(pairs))

    def candidate_diagnostics(self) -> dict:
        if self._diagnostics is None: raise RuntimeError('candidate not fitted')
        return dict(self._diagnostics)
