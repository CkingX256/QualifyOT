from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping
import math


@dataclass(frozen=True)
class IUTConfig:
    """Configuration for the principled three-axis qualification rule.

    Qualification is the intersection of three necessary claims: Movement,
    Utility and Retention.  The movement margin is a pre-specified minimally
    relevant PDR-scale margin; it must not be tuned from the held-out outcomes.
    """

    movement_margin: float = 0.01
    utility_margin: float = 0.0
    retention_margin: float = 0.0
    component_alpha: float = 0.05

    def __post_init__(self):
        vals = (self.movement_margin, self.utility_margin, self.retention_margin, self.component_alpha)
        if not all(math.isfinite(float(x)) for x in vals):
            raise ValueError("IUT configuration values must be finite")
        if self.movement_margin < 0:
            raise ValueError("movement_margin must be non-negative")
        if not (0 < self.component_alpha < 1):
            raise ValueError("component_alpha must lie in (0, 1)")

    @classmethod
    def from_mapping(cls, mapping):
        return cls(
            movement_margin=float(mapping.get("movement_margin", 0.01)),
            utility_margin=float(mapping.get("utility_margin", 0.0)),
            retention_margin=float(mapping.get("retention_margin", 0.0)),
            component_alpha=float(mapping.get("component_alpha", 0.05)),
        )


@dataclass(frozen=True)
class EvidenceDeficit:
    movement: float
    utility: float
    retention: float

    @property
    def limiting_axis(self) -> str | None:
        vals = {"Movement": self.movement, "Utility": self.utility, "Retention": self.retention}
        positive = {k: v for k, v in vals.items() if v > 0}
        return None if not positive else max(positive, key=positive.get)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["limiting_axis"] = self.limiting_axis
        return d


@dataclass(frozen=True)
class CoreEvidenceState:
    label: str
    qualified: bool
    axes: Mapping[str, bool]
    deficits: EvidenceDeficit
    retention_stability: float | None = None
    component_alpha: float = 0.05

    @property
    def limiting_axes(self) -> tuple[str, ...]:
        return tuple(k for k, v in self.axes.items() if not v)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "qualified": self.qualified,
            "axes": dict(self.axes),
            "limiting_axes": list(self.limiting_axes),
            "deficits": self.deficits.to_dict(),
            "retention_stability": self.retention_stability,
            "component_alpha": self.component_alpha,
        }


def classify_iut(
    *,
    pdr_lcb: float,
    puc_lcb: float,
    puc_ucb: float,
    npi_lcb: float,
    positive_weight_fraction: float | None = None,
    cfg: IUTConfig = IUTConfig(),
) -> CoreEvidenceState:
    """Classify evidence using the Movement–Utility–Retention IUT.

    ``puc_ucb`` is used only to distinguish supported direct harm (Adverse)
    from unresolved Utility.  It is not an additional qualification gate.
    Positive-weight frequency is a retention-stability descriptor only.
    """
    vals = (pdr_lcb, puc_lcb, puc_ucb, npi_lcb)
    if not all(math.isfinite(float(x)) for x in vals):
        raise ValueError("classification bounds must be finite")
    if positive_weight_fraction is not None and not (0 <= float(positive_weight_fraction) <= 1):
        raise ValueError("positive_weight_fraction must lie in [0, 1]")

    movement = bool(float(pdr_lcb) > cfg.movement_margin)
    utility = bool(float(puc_lcb) > cfg.utility_margin)
    retention = bool(float(npi_lcb) > cfg.retention_margin)
    axes = {"Movement": movement, "Utility": utility, "Retention": retention}

    qualified = bool(movement and utility and retention)
    if qualified:
        label = "Qualified"
    elif not movement:
        label = "Negligible"
    elif float(puc_ucb) < cfg.utility_margin:
        label = "Adverse"
    elif utility and not retention:
        label = "Promising"
    else:
        label = "Equivocal"

    deficits = EvidenceDeficit(
        movement=max(0.0, cfg.movement_margin - float(pdr_lcb)),
        utility=max(0.0, cfg.utility_margin - float(puc_lcb)),
        retention=max(0.0, cfg.retention_margin - float(npi_lcb)),
    )
    return CoreEvidenceState(
        label=label,
        qualified=qualified,
        axes=axes,
        deficits=deficits,
        retention_stability=None if positive_weight_fraction is None else float(positive_weight_fraction),
        component_alpha=float(cfg.component_alpha),
    )


def global_iut_type1_bound(component_levels) -> float:
    """Return the Type-I bound for a conjunction of necessary component tests.

    For H0 = union_j H0j and H1 = intersection_j H1j, passing the global
    decision requires rejecting every component null.  At any parameter point
    in H0, at least one H0j is true, hence P(all pass) <= P(test j passes) <=
    alpha_j.  The valid uniform bound is therefore max_j alpha_j.  This says
    nothing about multiplicity inside a searched family (weights, edges, etc.).
    """
    levels = [float(x) for x in component_levels]
    if not levels:
        raise ValueError("component_levels must be non-empty")
    if any((not math.isfinite(x) or x < 0 or x > 1) for x in levels):
        raise ValueError("component test levels must lie in [0, 1]")
    return max(levels)


def retained_risk_improvement(npi: float, reference_risk: float, retained_risk: float, *, atol: float = 1e-12) -> bool:
    """Audit the defining identity NPI = R_ref - R_retained.

    When the identity holds, NPI > 0 is exactly equivalent to retained risk
    being below reference risk; the latter must not be counted as a second
    independent inferential gate in the principled rule.
    """
    vals = (float(npi), float(reference_risk), float(retained_risk))
    if not all(math.isfinite(x) for x in vals):
        raise ValueError("NPI and risks must be finite")
    if not math.isclose(vals[0], vals[1] - vals[2], rel_tol=1e-9, abs_tol=atol):
        raise ValueError("NPI is inconsistent with reference_risk - retained_risk")
    return bool(vals[0] > 0)


def classify_iut_contrast(
    *,
    movement_contrast_lcb: float,
    utility_lcb: float,
    utility_ucb: float,
    retention_lcb: float,
    positive_weight_fraction: float | None = None,
    cfg: IUTConfig = IUTConfig(),
) -> CoreEvidenceState:
    """Classify using the Movement *margin contrast* as the inferential axis.

    The scientific Movement claim M > delta_M is algebraically equivalent to
    E[A - delta_M B] > 0 whenever E[B] > 0.  The contrast therefore avoids
    ratio instability while preserving the same scientific threshold.  The
    ratio M remains a descriptive effect size and denominator diagnostics are
    reported separately.
    """
    vals = (movement_contrast_lcb, utility_lcb, utility_ucb, retention_lcb)
    if not all(math.isfinite(float(x)) for x in vals):
        raise ValueError("classification bounds must be finite")
    if positive_weight_fraction is not None and not (0 <= float(positive_weight_fraction) <= 1):
        raise ValueError("positive_weight_fraction must lie in [0, 1]")
    movement = bool(float(movement_contrast_lcb) > 0.0)
    utility = bool(float(utility_lcb) > cfg.utility_margin)
    retention = bool(float(retention_lcb) > cfg.retention_margin)
    axes = {"Movement": movement, "Utility": utility, "Retention": retention}
    qualified = bool(movement and utility and retention)
    if qualified:
        label = "Qualified"
    elif not movement:
        label = "Negligible"
    elif float(utility_ucb) < cfg.utility_margin:
        label = "Adverse"
    elif utility and not retention:
        label = "Promising"
    else:
        label = "Equivocal"
    deficits = EvidenceDeficit(
        movement=max(0.0, -float(movement_contrast_lcb)),
        utility=max(0.0, cfg.utility_margin - float(utility_lcb)),
        retention=max(0.0, cfg.retention_margin - float(retention_lcb)),
    )
    return CoreEvidenceState(
        label=label,
        qualified=qualified,
        axes=axes,
        deficits=deficits,
        retention_stability=None if positive_weight_fraction is None else float(positive_weight_fraction),
        component_alpha=float(cfg.component_alpha),
    )
