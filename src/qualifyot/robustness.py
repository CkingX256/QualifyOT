from __future__ import annotations
import numpy as np
import pandas as pd
from .model import STATES,mae

def exact_downsample_counts(counts,frac,rng):
    counts=np.asarray(counts,int); n=counts.sum(); draw=int(np.floor(frac*n))
    if draw>=n: return counts.copy()
    return rng.multivariate_hypergeometric(counts,draw)

def downsample_pair_table(pairs,fracs=(1,.75,.5,.25),seeds=100,seed=0):
    # Only runs when exact counts are available. 100% uses raw proportions: no Jeffreys smoothing.
    rows=[]; rng0=np.random.default_rng(seed)
    for frac in fracs:
        for rep in range(seeds):
            rng=np.random.default_rng(int(rng0.integers(0,2**32-1)))
            for _,r in pairs.iterrows():
                ns=r.get('source_cells',np.nan); nt=r.get('target_cells',np.nan)
                if not np.isfinite(ns) or not np.isfinite(nt): continue
                sc=np.rint(np.array([r[f'source__{s}'] for s in STATES])*int(ns)).astype(int); tc=np.rint(np.array([r[f'target__{s}'] for s in STATES])*int(nt)).astype(int)
                # exact sum repair to largest bin
                sc[np.argmax(sc)]+=int(ns)-sc.sum(); tc[np.argmax(tc)]+=int(nt)-tc.sum()
                sd=exact_downsample_counts(sc,frac,rng); td=exact_downsample_counts(tc,frac,rng); sp=sd/sd.sum(); tp=td/td.sum()
                rows.append({'fraction':frac,'seed':rep,'patient_id':r.patient_id,'pair_id':r.pair_id,'measurement_MAE_vs_full_source':float(np.abs(sp-np.array([r[f"source__{s}"] for s in STATES])).mean()),'measurement_MAE_vs_full_target':float(np.abs(tp-np.array([r[f"target__{s}"] for s in STATES])).mean())})
    return pd.DataFrame(rows)
