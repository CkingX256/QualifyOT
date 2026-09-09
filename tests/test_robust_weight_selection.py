import numpy as np
from qualifyot.robust_weight_selection import tail_safe_msw
from qualifyot.weight_selection import one_se_msw


def test_tail_safe_weight_blocks_average_beneficial_tail_harm():
    # 80% benefit and 20% strong harm at lambda=1. Mean effect is favorable,
    # while the upper 20% tail is clearly adverse.
    n=50; grid=np.array([0.,.5,1.])
    D=np.zeros((n,3)); D[:40,1]=-.02; D[:40,2]=-.04; D[40:,1]=.04; D[40:,2]=.08
    R=.10+D
    mean=one_se_msw(D,R,grid,alpha=.05,delta=0,B=400,seed=3,lam=0)
    robust=tail_safe_msw(D,R,grid,bootstrap=400,seed=3,tail_fraction=.2,tail_harm_margin=0,one_se_lambda=0)
    assert mean['weight']>0
    assert robust['weight']==0
    assert robust['tail_ucb'][0]==0
