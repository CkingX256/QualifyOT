import numpy as np

from qualifyot.safe_averaging import safe_graph_average, simplex_grid


def _simple_case():
    n = 40
    target = np.tile([.7, .3], (n, 1))
    frozen = np.tile([.55, .45], (n, 1))
    good = np.tile([.68, .32], (n, 1))
    bad = np.tile([.35, .65], (n, 1))
    pred = np.stack([frozen, good, bad], axis=1)
    pats = np.array([f"p{i}" for i in range(n)])
    return target, pred, pats


def test_simplex_grid_closes_exactly():
    g = simplex_grid(3, .25)
    assert np.allclose(g.sum(1), 1)
    assert np.all(g >= 0)


def test_safe_graph_average_selects_supported_improvement():
    y, pred, pats = _simple_case()
    out = safe_graph_average(y, pred, pats, frozen_index=0, step=.5, bootstrap=500, seed=11, one_se=False)
    assert out.weights[1] > 0
    assert out.mean_risk < out.frozen_risk
    assert out.safety_ucb <= 1e-12


def test_safe_graph_average_falls_back_when_alternatives_are_worse():
    y, pred, pats = _simple_case()
    pred[:, 1, :] = [.40, .60]
    out = safe_graph_average(y, pred, pats, frozen_index=0, step=.5, bootstrap=400, seed=12, one_se=False)
    assert np.isclose(out.weights[0], 1.0)
