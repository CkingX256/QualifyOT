from __future__ import annotations

"""Training-only retained-influence selection with an upper-tail harm screen."""

import math
import numpy as np

from .weight_selection import simultaneous_ucb


def _cvar_cols(D: np.ndarray, tail_fraction: float) -> np.ndarray:
    n=D.shape[0]; k=max(1,int(math.ceil(float(tail_fraction)*n)))
    return np.partition(D,n-k,axis=0)[n-k:,:].mean(axis=0)


def tail_safe_msw(
    D,
    R,
    grid,
    *,
    alpha_mean: float = 0.05,
    mean_harm_margin: float = 0.0,
    tail_fraction: float = 0.20,
    alpha_tail: float = 0.05,
    tail_harm_margin: float = 0.0,
    bootstrap: int = 5000,
    seed: int = 0,
    one_se_lambda: float = 1.0,
):
    """Choose a retained candidate weight subject to mean and CVaR non-harm.

    ``D[i,j]`` is patient i's loss difference for weight j relative to the
    reference (positive means harm); ``R[i,j]`` is the corresponding absolute
    loss.  Both matrices must come from training-patient cross-fitted
    predictions.  The zero-weight reference is always a deterministic fallback.

    The mean screen uses the existing simultaneous max-t band.  The tail screen
    is a simultaneous patient-bootstrap upper band for CVaR over the complete
    weight grid.  The latter is an empirical robustness device, not an exact
    finite-sample subgroup-safety theorem.
    """
    D=np.asarray(D,float); R=np.asarray(R,float); grid=np.asarray(grid,float)
    if D.ndim!=2 or R.shape!=D.shape or len(grid)!=D.shape[1]:
        raise ValueError("D, R and grid shapes do not match")
    if D.shape[0]<2: raise ValueError("at least two physical patients are required")
    if not np.isfinite(D).all() or not np.isfinite(R).all() or not np.isfinite(grid).all():
        raise ValueError("D, R and grid must be finite")
    if not (0<tail_fraction<=1): raise ValueError("tail_fraction must lie in (0,1]")
    if int(bootstrap)<100: raise ValueError("bootstrap must be >=100")
    zero=np.flatnonzero(np.isclose(grid,0,atol=1e-12))
    if len(zero)!=1: raise ValueError("grid must contain exactly one zero-weight reference")
    zi=int(zero[0])
    mean_ucb,se,q=simultaneous_ucb(D,alpha=alpha_mean,B=bootstrap,seed=seed,zero_index=zi)
    mean_ok=mean_ucb<=float(mean_harm_margin)+1e-12

    cvar=_cvar_cols(D,tail_fraction)
    rng=np.random.default_rng(seed+65537); n=D.shape[0]
    max_dev=np.empty(int(bootstrap),float); batch=100
    for st in range(0,int(bootstrap),batch):
        b=min(batch,int(bootstrap)-st); idx=rng.integers(0,n,size=(b,n))
        for j in range(b):
            cb=_cvar_cols(D[idx[j]],tail_fraction)
            max_dev[st+j]=float(np.max(cb-cvar))
    tq=float(np.quantile(max_dev,1-alpha_tail)); tail_ucb=cvar+tq
    cvar[zi]=0.0; tail_ucb[zi]=0.0
    tail_ok=tail_ucb<=float(tail_harm_margin)+1e-12
    feasible=np.isfinite(mean_ucb)&np.isfinite(tail_ucb)&mean_ok&tail_ok; feasible[zi]=True
    ids=np.flatnonzero(feasible); mr=R.mean(axis=0)
    best=int(ids[np.argmin(mr[ids])]); sebest=float(R[:,best].std(ddof=1)/math.sqrt(n))
    near=ids[mr[ids]<=mr[best]+float(one_se_lambda)*sebest+1e-15]
    # Prefer less retained candidate influence when risks are statistically close.
    chosen=int(near[np.argmin(grid[near])])
    return {'weight':float(grid[chosen]),'best_weight':float(grid[best]),'max_eligible_weight':float(grid[ids[-1]]),
            'se_best':sebest,'max_t_q':float(q),'ucb':mean_ucb,'tail_cvar':cvar,'tail_ucb':tail_ucb,
            'tail_q':tq,'tail_fraction':float(tail_fraction),'risk':mr,'feasible':feasible,
            'mean_harm_margin':float(mean_harm_margin),'tail_harm_margin':float(tail_harm_margin)}
