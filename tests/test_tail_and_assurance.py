import numpy as np

from qualifyot.tail_safety import tail_safety_profile
from qualifyot.replication_planning import bayesian_bootstrap_assurance, assurance_target


def test_tail_safety_detects_minority_harm_despite_average_gain():
    n = 20
    y = np.tile([.5, .5], (n, 1))
    ref = np.tile([.6, .4], (n, 1))
    ret = np.tile([.54, .46], (n, 1))
    # Four patients are substantially worse than the reference.
    ret[-4:, :] = [.75, .25]
    pats = np.array([f"p{i}" for i in range(n)])
    out = tail_safety_profile(y, ref, ret, pats, tail_fraction=.2, bootstrap=800, seed=5)
    assert out.mean_harm < 0
    assert out.cvar_harm > 0
    assert not out.tail_nonharm_supported
    assert np.isclose(out.harmed_fraction, .2)


def test_bayesian_bootstrap_assurance_increases_with_sample_size():
    x = np.array([.006, .012, .015, .018, .010, .014, .009, .016, .011, .013])
    plan = bayesian_bootstrap_assurance(x, [10, 30, 80], posterior_draws=80, future_draws=120, seed=9)
    assert plan.assurance_mean.iloc[-1] >= plan.assurance_mean.iloc[0]
    t = assurance_target(plan, target=.5)
    assert t in {10, 30, 80, None}
