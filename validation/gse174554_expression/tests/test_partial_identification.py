import numpy as np

from qualifyot.partial_identification import graph_confidence_set


def test_graph_confidence_set_excludes_clearly_worse_model():
    rng = np.random.default_rng(3)
    n = 80
    base = rng.normal(.10, .015, size=n)
    losses = np.column_stack([
        base,
        base + rng.normal(0, .004, size=n),
        base + .06 + rng.normal(0, .004, size=n),
    ])
    pats = np.array([f"p{i}" for i in range(n)])
    edges = {
        "g0": [("A", "B"), ("B", "C")],
        "g1": [("A", "B"), ("A", "C")],
        "g2": [("C", "A")],
    }
    out = graph_confidence_set(losses, pats, ["g0", "g1", "g2"], graph_edges=edges, bootstrap=800, seed=7)
    assert "g2" in out.excluded
    assert len(out.members) >= 1
    assert ("C", "A") in out.excluded_edges


def test_graph_confidence_set_keeps_indistinguishable_models():
    rng = np.random.default_rng(8)
    n = 60
    base = rng.normal(.1, .02, size=n)
    losses = np.column_stack([base, base + rng.normal(0, .001, n)])
    pats = np.array([f"p{i}" for i in range(n)])
    out = graph_confidence_set(losses, pats, ["a", "b"], bootstrap=600, seed=9)
    assert set(out.members) == {"a", "b"}
