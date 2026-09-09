import numpy as np

from qualifyot.decision import IUTConfig, classify_iut, global_iut_type1_bound


def test_iut_qualification_uses_three_inferential_axes_only():
    s = classify_iut(
        pdr_lcb=.20, puc_lcb=.01, puc_ucb=.03, npi_lcb=.004,
        positive_weight_fraction=.10,
    )
    assert s.qualified
    assert s.label == "Qualified"
    # Fold frequency is a stability descriptor, not a qualification gate.
    assert s.retention_stability == .10
    assert s.limiting_axes == ()


def test_iut_deficit_localizes_retention_boundary():
    s = classify_iut(
        pdr_lcb=.20, puc_lcb=.01, puc_ucb=.03, npi_lcb=-.002,
        positive_weight_fraction=.8,
    )
    assert not s.qualified and s.label == "Promising"
    assert s.deficits.retention == .002
    assert s.deficits.limiting_axis == "Retention"


def test_iut_adverse_uses_utility_upper_bound_without_new_gate():
    s = classify_iut(pdr_lcb=.2, puc_lcb=-.03, puc_ucb=-.01, npi_lcb=-.01)
    assert s.label == "Adverse"
    assert not s.axes["Utility"]


def test_iut_global_type1_bound_is_max_component_level():
    assert np.isclose(global_iut_type1_bound([.05, .05, .05]), .05)
    assert np.isclose(global_iut_type1_bound([.01, .05, .025]), .05)
