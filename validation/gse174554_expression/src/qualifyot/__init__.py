"""QualifyOT: patient-level evidence control for structured predictors."""

from .decision import IUTConfig, CoreEvidenceState, classify_iut, classify_iut_contrast
from .bayesian_evidence import (
    BayesianBootstrapMean,
    BayesianGraphProfile,
    bayesian_bootstrap_mean,
    bayesian_bootstrap_graph_profile,
)
from .partial_identification import GraphConfidenceSet, graph_confidence_set
from .safe_averaging import SafeAverageResult, HonestSafeAverageResult, safe_graph_average, honest_safe_graph_average
from .robust_averaging import RobustAverageResult, robust_graph_average
from .pipeline import PipelineConfig, QualifyOTPipeline
from .transformer_candidate import GraphResidualTransformerCandidate
from .comparison_candidates import CompositionEntropicOTCandidate
from .inference import (
    FrozenAnalysisContract,
    MovementMarginEvidence,
    HonestConfirmationResult,
    movement_margin_evidence,
    honest_confirmation_iut,
    honest_confirmation_iut_cohort_conditional,
    honest_confirmation_iut_localized,
    honest_confirmation_iut_population,
    honest_confirmation_pvalues_population,
    honest_confirm_predictions,
    nearest_grid_risk_regret_bound,
)

__all__ = [
    "IUTConfig", "CoreEvidenceState", "classify_iut", "classify_iut_contrast",
    "BayesianBootstrapMean", "BayesianGraphProfile",
    "bayesian_bootstrap_mean", "bayesian_bootstrap_graph_profile",
    "GraphConfidenceSet", "graph_confidence_set",
    "SafeAverageResult", "HonestSafeAverageResult", "safe_graph_average", "honest_safe_graph_average",
    "RobustAverageResult", "robust_graph_average",
    "PipelineConfig", "QualifyOTPipeline",
    "GraphResidualTransformerCandidate", "CompositionEntropicOTCandidate",
    "FrozenAnalysisContract", "MovementMarginEvidence", "HonestConfirmationResult",
    "movement_margin_evidence", "honest_confirmation_iut",
    "honest_confirmation_iut_cohort_conditional", "honest_confirmation_iut_localized",
    "honest_confirmation_iut_population", "honest_confirmation_pvalues_population",
    "honest_confirmation_pvalues_localized", "heterogeneous_hoeffding_pvalue",
    "honest_confirm_predictions", "nearest_grid_risk_regret_bound",
    "contract_sha256", "dependency_sha256", "write_contract_digest", "verify_contract_digest",
    "patient_set_sha256", "deterministic_patient_split",
    "freeze_patient_set_and_split", "verify_target_access_gate",
]

from .external_contract import (
    contract_sha256, dependency_sha256, write_contract_digest, verify_contract_digest,
    patient_set_sha256, deterministic_patient_split,
)

from .inference import honest_confirmation_pvalues_localized, heterogeneous_hoeffding_pvalue
from .external_contract import freeze_patient_set_and_split, verify_target_access_gate
