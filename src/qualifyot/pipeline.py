from __future__ import annotations

"""Single-entry QualifyOT pipeline.

The public method is intentionally small:

1. freeze an ontology, directed graph and candidate generator;
2. fit/predict only inside patient-level training folds;
3. evaluate Movement, Utility and Retention with the existing IUT;
4. report orthogonal robustness/safety profiles separately.

This wrapper does not create new statistics.  It only gives users one coherent
entry point while retaining the lower-level APIs for reproducibility.
"""

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from .candidate_api import CandidatePredictor, GraphFlowCandidate
from .decision import IUTConfig
from .generic_engine import run_generic_lopo


@dataclass(frozen=True)
class PipelineConfig:
    bootstrap: int = 5000
    seed: int = 20260831
    reference_rule: Literal["nested", "Persistence", "CohortMean", "MeanDelta"] = "nested"
    patient_balanced: bool = True
    orthogonal_profiles: bool = True
    profile_draws: int = 10000
    movement_core_mode: Literal["legacy_ratio", "margin_contrast"] = "margin_contrast"
    movement_contrast_method: Literal["t", "percentile", "bca"] = "t"
    iut: IUTConfig = field(default_factory=IUTConfig)

    def __post_init__(self):
        if self.bootstrap < 1:
            raise ValueError("bootstrap must be positive")
        if self.profile_draws < 1:
            raise ValueError("profile_draws must be positive")
        if self.movement_core_mode not in {"legacy_ratio", "margin_contrast"}:
            raise ValueError("movement_core_mode must be legacy_ratio or margin_contrast")
        if self.movement_contrast_method not in {"t", "percentile", "bca"}:
            raise ValueError("movement_contrast_method must be t, percentile or bca")


class QualifyOTPipeline:
    """One coherent API for a frozen graph candidate and patient-level evidence.

    Parameters
    ----------
    states, edges
        Frozen ontology and, for graph-based candidates, the directed predictive graph.
        ``edges`` may be omitted for candidate-agnostic predictors such as
        ``"patient-multiscale"`` or for a custom non-graph CandidatePredictor.
    candidate
        ``"patient-multiscale"`` (patient-unit coherent generic candidate),
        ``"graphflow"`` (canonical low-capacity structured candidate),
        ``"graph-transformer"`` (pre-specified nonlinear structured extension), or a custom
        :class:`CandidatePredictor` instance.

    Notes
    -----
    The default core Movement inference is the margin-contrast margin contrast
    ``G_M = A - delta_M B``. ``legacy_ratio`` remains available only for
    historical locked-result reproduction.

    The Transformer is never part of the qualification rule.  Both GraphFlow
    and GraphResidualTransformer enter the exact same outer evidence engine.
    """

    def __init__(
        self,
        states,
        edges=None,
        *,
        candidate: str | CandidatePredictor = "graphflow",
        config: PipelineConfig | None = None,
        transformer_kwargs: dict | None = None,
        graphflow_alpha: float = 1.0,
    ):
        self.states = list(states)
        self.edges = [] if edges is None else [tuple(e) for e in edges]
        if not self.states:
            raise ValueError("states cannot be empty")
        self.config = config or PipelineConfig()
        self.transformer_kwargs = dict(transformer_kwargs or {})
        self.graphflow_alpha = float(graphflow_alpha)
        self._candidate_spec = candidate

    def candidate_factory(self) -> CandidatePredictor:
        if isinstance(self._candidate_spec, CandidatePredictor):
            return self._candidate_spec.fresh()
        method = str(self._candidate_spec).strip().lower()
        if method in {"patient-multiscale", "patientscale", "psd"}:
            from .adaptive_candidates import PatientMultiScaleDeltaCandidate
            return PatientMultiScaleDeltaCandidate(states=self.states)
        if method in {"robustblend", "robust-blend"}:
            from .candidates import RobustBlendCandidate
            return RobustBlendCandidate(states=self.states)
        if method == "graphflow":
            if not self.edges:
                raise ValueError("a frozen directed graph is required for graphflow")
            return GraphFlowCandidate(
                states=self.states,
                edges=self.edges,
                alpha=self.graphflow_alpha,
                patient_balanced=self.config.patient_balanced,
                name="GraphFlow",
            )
        if method in {"graph-transformer", "transformer", "fused"}:
            if not self.edges:
                raise ValueError("a frozen directed graph is required for graph-transformer")
            from .transformer_candidate import GraphResidualTransformerCandidate
            kw = dict(self.transformer_kwargs)
            kw.setdefault("alpha", self.graphflow_alpha)
            kw.setdefault("patient_balanced", self.config.patient_balanced)
            kw.setdefault("seed", self.config.seed)
            return GraphResidualTransformerCandidate(
                states=self.states,
                edges=self.edges,
                **kw,
            )
        raise ValueError(
            "candidate must be 'patient-multiscale', 'robustblend', 'graph-transformer', "
            "'graphflow', or CandidatePredictor"
        )

    def run(self, pairs: pd.DataFrame, *, return_predictions: bool = False) -> dict:
        if "patient_id" not in pairs.columns:
            raise KeyError("patient_id is required")
        cand = self.candidate_factory()
        cfg = self.config
        out = run_generic_lopo(
            pairs,
            cand,
            bootstrap=cfg.bootstrap,
            seed=cfg.seed,
            patient_balanced=cfg.patient_balanced,
            states=self.states,
            reference_rule=cfg.reference_rule,
            iut_cfg=cfg.iut,
            orthogonal_profiles=cfg.orthogonal_profiles,
            profile_draws=cfg.profile_draws,
            return_predictions=return_predictions,
            movement_contrast_method=cfg.movement_contrast_method,
            use_movement_contrast_for_core=(cfg.movement_core_mode == "margin_contrast"),
        )
        out["methodology"] = {
            "candidate_layer": cand.name,
            "frozen_graph": [list(e) for e in self.edges] if self.edges else None,
            "qualification": "Movement-Utility-Retention intersection-union test",
            "patient_is_inferential_unit": True,
            "heldout_target_used_for_tuning": False,
            "orthogonal_profiles_are_core_gates": False,
            "movement_core_mode": cfg.movement_core_mode,
            "movement_contrast_method": cfg.movement_contrast_method,
        }
        return out
