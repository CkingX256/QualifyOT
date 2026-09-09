import numpy as np
import pandas as pd

from qualifyot.adaptive_candidates import PatientNormalizedDeltaRidgeCandidate, PatientMultiScaleDeltaCandidate


def _pairs():
    rows=[]
    states=['A','B','C']
    vals={
        'P1':[([.7,.2,.1],[.6,.25,.15]),([.6,.25,.15],[.5,.3,.2])],
        'P2':[([.2,.6,.2],[.25,.5,.25])],
        'P3':[([.3,.2,.5],[.35,.25,.4]),([.35,.25,.4],[.4,.3,.3]),([.4,.3,.3],[.45,.3,.25])],
        'P4':[([.45,.45,.1],[.4,.45,.15])],
    }
    for pid,pairs in vals.items():
        for j,(s,t) in enumerate(pairs):
            r={'patient_id':pid,'pair_id':f'{pid}_{j}','source_time':str(j),'target_time':str(j+1)}
            for k,st in enumerate(states):
                r[f'source__{st}']=s[k]; r[f'target__{st}']=t[k]
            rows.append(r)
    return pd.DataFrame(rows),states


def _duplicate_patient(df,pid,times=4):
    block=df[df.patient_id==pid].copy()
    return pd.concat([df]+[block.copy() for _ in range(times-1)],ignore_index=True)


def test_patient_normalized_ridge_is_exactly_replication_invariant():
    df,states=_pairs(); dup=_duplicate_patient(df,'P3',4)
    a=PatientNormalizedDeltaRidgeCandidate(states,alpha=1.0).fit(df).predict(df)
    b=PatientNormalizedDeltaRidgeCandidate(states,alpha=1.0).fit(dup).predict(df)
    np.testing.assert_allclose(a,b,rtol=0,atol=2e-14)


def test_patient_multiscale_is_exactly_replication_invariant():
    df,states=_pairs(); dup=_duplicate_patient(df,'P3',4)
    a=PatientMultiScaleDeltaCandidate(states).fit(df).predict(df)
    b=PatientMultiScaleDeltaCandidate(states).fit(dup).predict(df)
    np.testing.assert_allclose(a,b,rtol=0,atol=2e-14)
