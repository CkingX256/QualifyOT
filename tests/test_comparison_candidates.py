import numpy as np
import pandas as pd
from qualifyot.comparison_candidates import CompositionEntropicOTCandidate, _sinkhorn_coupling


def _pairs():
    return pd.DataFrame({
        'patient_id':['a','b','c'], 'pair_id':['a0','b0','c0'], 'source_time':[0.,0.,0.], 'target_time':[1.,1.,1.],
        'source__A':[.8,.6,.7], 'source__B':[.2,.4,.3],
        'target__A':[.6,.4,.5], 'target__B':[.4,.6,.5],
    })


def test_sinkhorn_marginals():
    a=np.array([.7,.3]); b=np.array([.4,.6]); g=_sinkhorn_coupling(a,b,.25)
    assert np.allclose(g.sum(1),a,atol=1e-8)
    assert np.allclose(g.sum(0),b,atol=1e-8)


def test_composition_ot_simplex_and_target_free_predict():
    d=_pairs(); c=CompositionEntropicOTCandidate(['A','B']).fit(d)
    test=d.copy(); test[['target__A','target__B']]=np.nan
    p=c.predict(test)
    assert p.shape==(3,2)
    assert np.all(p>=0) and np.allclose(p.sum(1),1)


def test_composition_ot_patient_replication_invariance():
    d=_pairs(); p0=CompositionEntropicOTCandidate(['A','B']).fit(d).predict(d)
    dup=pd.concat([d, d[d.patient_id=='a'], d[d.patient_id=='a'], d[d.patient_id=='a']],ignore_index=True)
    p1=CompositionEntropicOTCandidate(['A','B']).fit(dup).predict(d)
    assert np.allclose(p0,p1,atol=1e-10)
