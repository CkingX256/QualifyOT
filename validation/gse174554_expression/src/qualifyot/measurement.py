from __future__ import annotations
import numpy as np

def dirichlet_posterior_draw(counts,rng=None,prior=.5,size=None):
    """Jeffreys-smoothed multinomial-composition uncertainty draw.

    This is a sensitivity tool, not the unsmoothed primary estimator.
    """
    c=np.asarray(counts,float)
    if (c<0).any() or c.ndim!=1: raise ValueError('counts must be a non-negative 1D vector')
    rng=np.random.default_rng() if rng is None else rng
    return rng.dirichlet(c+float(prior),size=size)

def multinomial_max_se(counts):
    c=np.asarray(counts,float); n=c.sum()
    if n<=0: return np.nan
    p=c/n
    return float(np.sqrt(np.max(p*(1-p)/n)))
