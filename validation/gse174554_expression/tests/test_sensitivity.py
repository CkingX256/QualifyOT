import json
import numpy as np
import pandas as pd
from qualifyot.model import (
    STATES,EDGES,incidence,residual_flow,project_flow,fit_graphflow,
    patient_sample_weights,fit_null,weighted_mean_rows,
)
from qualifyot.sensitivity import all_label_permutations,threshold_stability


def test_residual_graphflow_handles_disconnected_state_change():
    B=incidence(STATES,EDGES)
    s=np.array([.4,.2,.1,.2,.1]); t=np.array([.3,.25,.15,.15,.15])
    f,r,gcf=residual_flow(s,t,Bmat=B)
    assert np.all(f>=-1e-12)
    assert np.allclose(B@f+r,t-s,atol=1e-9)
    assert 0<=gcf<=1
    # Regulatory/Other-T have no incident edges, so their changes must be residual.
    assert abs(r[STATES.index('Regulatory')]-(t[3]-s[3]))<1e-8
    assert abs(r[STATES.index('Other_T')]-(t[4]-s[4]))<1e-8


def test_custom_graph_prediction_stays_in_simplex():
    edges=[('Memory','Exhausted'),('Regulatory','Other_T')]; B=incidence(STATES,edges)
    s=np.array([.5,.1,.1,.2,.1]); f=np.array([.7,.5])
    _,p=project_flow(s,f,B)
    assert p.min()>=-1e-9 and abs(p.sum()-1)<1e-9


def test_patient_balanced_weights_equal_total_per_patient():
    p=np.array(['A','A','A','B','C','C'])
    w=patient_sample_weights(p)
    totals={x:w[p==x].sum() for x in np.unique(p)}
    assert max(totals.values())-min(totals.values())<1e-12
    assert abs(w.mean()-1)<1e-12


def test_patient_balanced_cohort_mean_differs_when_visit_counts_differ():
    source=np.tile(np.array([1.,0,0,0,0]),(4,1))
    target=np.array([[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[0,1,0,0,0]],float)
    pats=np.array(['A','A','A','B'])
    pair_equal=fit_null('CohortMean',source,target,patients=pats,patient_balanced=False)['center']
    patient_equal=fit_null('CohortMean',source,target,patients=pats,patient_balanced=True)['center']
    assert np.allclose(pair_equal,[.75,.25,0,0,0])
    assert np.allclose(patient_equal,[.5,.5,0,0,0])


def test_all_five_state_label_permutations_are_enumerated():
    x=all_label_permutations()
    assert len(x)==120
    assert len({r['graph_key'] for r in x})==60  # two pre-specified isolated nodes create twofold duplicate labelings


def test_threshold_stability_does_not_mutate_primary_values():
    a={'PDR_Hellinger':.6,'PDR_CI2.5':.3,'PUC_CI2.5':.01,'positive_weight_fold_fraction':5/9,'NPI_CI2.5':-.001,'QualifyOT_minus_reference_MAE':-.002}
    before=dict(a); df=threshold_stability(a)
    assert a==before
    assert not df.would_meet_full_positive_criterion.any()
