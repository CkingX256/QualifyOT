import numpy as np
import pandas as pd
from qualifyot.candidates import DirectDeltaRidgeCandidate, TargetRidgeCandidate, RobustBlendCandidate

ST=['A','B','C']
def tiny():
    rows=[]
    for i in range(8):
        s=np.array([.5-.02*i,.3+.01*i,.2+.01*i]); t=np.array([.42-.01*i,.34+.005*i,.24+.005*i])
        d={'patient_id':f'p{i}','pair_id':f'x{i}','source_time':0.0,'target_time':1.0}
        for k,v in zip(ST,s): d[f'source__{k}']=v
        for k,v in zip(ST,t): d[f'target__{k}']=v
        rows.append(d)
    return pd.DataFrame(rows)

def test_latest_candidates_simplex():
    p=tiny()
    for c in [DirectDeltaRidgeCandidate(ST),TargetRidgeCandidate(ST),RobustBlendCandidate(ST)]:
        q=c.fit(p.iloc[:-2]).predict(p.iloc[-2:])
        assert q.shape==(2,3)
        assert np.all(q>=-1e-9)
        assert np.allclose(q.sum(1),1.0)
