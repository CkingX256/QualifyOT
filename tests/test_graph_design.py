import numpy as np
import pandas as pd

from qualifyot.graph_design import SafeGraphAveragingCandidate


def _pairs(n=12):
    rng=np.random.default_rng(4)
    states=['A','B','C']
    S=rng.dirichlet([4,3,2],size=n)
    T=S.copy(); f=.2*S[:,0]; T[:,0]-=f; T[:,1]+=f
    T=np.vstack([rng.dirichlet(np.maximum(t*500,.1)) for t in T])
    d={'patient_id':[f'p{i}' for i in range(n)],'pair_id':[f'p{i}:0->1' for i in range(n)],'source_time':0,'target_time':1}
    for j,s in enumerate(states): d[f'source__{s}']=S[:,j]; d[f'target__{s}']=T[:,j]
    return pd.DataFrame(d),states


def test_safe_graph_averaging_candidate_is_training_only_candidate():
    df,states=_pairs()
    graphs=[(('A','B'),), (('B','A'),), (('A','B'),('B','C'))]
    c=SafeGraphAveragingCandidate(states,graphs,frozen_edges=graphs[2],cv_folds=3,grid_step=.5,selector_bootstrap=80,seed=5)
    c.fit(df.iloc[:-2].reset_index(drop=True))
    p=c.predict(df.iloc[-2:].reset_index(drop=True))
    assert p.shape==(2,3)
    assert np.allclose(p.sum(1),1)
    assert np.isclose(c.selected_weights.sum(),1)
