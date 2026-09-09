import math
import numpy as np
import pandas as pd

from qualifyot.decision import classify_iut_contrast, IUTConfig
from qualifyot.inference import (
    FrozenAnalysisContract,
    honest_confirmation_iut,
    hoeffding_lcb,
    movement_margin_evidence,
    movement_patient_components,
    nearest_grid_risk_regret_bound,
)
from qualifyot.candidate_api import MeanDeltaCandidate
from qualifyot.generic_engine import run_generic_lopo


def test_movement_margin_equivalence_positive_denominator():
    a=np.array([0.02,0.04,0.03,0.05])
    b=np.array([0.20,0.30,0.25,0.35])
    p=np.array(['a','b','c','d'])
    delta=.10
    aa,bb,g=movement_patient_components(a,b,p,movement_margin=delta)
    ratio=aa.mean()/bb.mean()
    assert (ratio > delta) == (g.mean() > 0)


def test_movement_near_zero_denominator_is_not_fabricated():
    a=np.array([1e-10,2e-10,1e-10,2e-10])
    b=np.array([1e-12,2e-12,1e-12,2e-12])
    p=np.array(['a','b','c','d'])
    ev=movement_margin_evidence(a,b,p,movement_margin=.01,denominator_floor=1e-8)
    assert not ev.ratio_identifiable
    assert math.isnan(ev.ratio)
    assert math.isfinite(ev.contrast)
    assert math.isfinite(ev.contrast_lcb)


def test_hoeffding_bound_and_honest_iut():
    x=np.full(200,0.5)
    assert hoeffding_lcb(x,lower=-1,upper=1,alpha=.05) < .5
    out=honest_confirmation_iut(
        np.full(500,.6), np.full(500,.6), np.full(500,.6), movement_margin=.01, alpha=.05
    )
    assert out.qualified


def test_grid_regret_bound():
    assert nearest_grid_risk_regret_bound(grid_step=.01,n_states=5) == .002


def test_frozen_contract_hash_is_deterministic_and_sensitive():
    a=FrozenAnalysisContract('x','nested',('A','B'),'MAE',.01,(0,.5,1),'honest')
    b=FrozenAnalysisContract('x','nested',('A','B'),'MAE',.01,(0,.5,1),'honest')
    c=FrozenAnalysisContract('x','Persistence',('A','B'),'MAE',.01,(0,.5,1),'honest')
    assert a.sha256 == b.sha256
    assert a.sha256 != c.sha256


def test_contrast_classifier_has_same_axis_semantics():
    out=classify_iut_contrast(
        movement_contrast_lcb=.001,utility_lcb=.01,utility_ucb=.02,retention_lcb=.002,
        positive_weight_fraction=.5,cfg=IUTConfig(),
    )
    assert out.qualified and out.label=='Qualified'


def test_generic_engine_emits_movement_margin_diagnostic():
    states=['A','B']
    rows=[]
    for i in range(5):
        s=np.array([.6,.4])
        t=np.array([.55-.01*i,.45+.01*i])
        rows.append({
            'patient_id':f'p{i}','pair_id':f'x{i}','source_time':0,'target_time':1,
            'source__A':s[0],'source__B':s[1],'target__A':t[0],'target__B':t[1],
        })
    df=pd.DataFrame(rows)
    out=run_generic_lopo(
        df,MeanDeltaCandidate(states=states,patient_balanced=True),bootstrap=100,seed=4,
        states=states,orthogonal_profiles=False,return_predictions=False,
    )
    assert 'movement_margin_evidence' in out
    assert out['movement_margin_evidence']['n_patients']==5
    assert 'movement_margin_contribution' in out['patient_influence'].columns


def test_honest_sga_has_exact_frozen_fallback_and_no_harm_screen():
    from qualifyot.safe_averaging import honest_safe_graph_average
    # Model 0 is perfect; model 1 is deliberately worse.  The honest screen
    # must retain an always-feasible frozen fallback.
    target=np.tile(np.array([[.8,.2]]),(20,1))
    p0=target.copy()
    p1=np.tile(np.array([[.2,.8]]),(20,1))
    pred=np.stack([p0,p1],axis=1)
    patients=np.array([f'p{i}' for i in range(20)])
    out=honest_safe_graph_average(target,pred,patients,frozen_index=0,step=.5,alpha=.05,safety_margin=0)
    assert out.fallback_used
    assert out.safety_ucb <= 1e-12


def test_localized_honest_confirmation_is_valid_on_constructed_bounds():
    from qualifyot.inference import honest_confirmation_iut_localized
    n=100
    a=np.full(n,.2)
    gm=np.full(n,.195)  # in [a-.01,a]
    u=np.full(n,.04); cu=np.full(n,.05)
    r=np.full(n,.03); cr=np.full(n,.04)
    out=honest_confirmation_iut_localized(gm,a,u,cu,r,cr,movement_margin=.01,alpha=.05)
    assert out.qualified
    assert out.movement.lower_bound > 0


def test_legacy_ratio_does_not_silently_divide_by_zero():
    import pytest
    from qualifyot.experiment import bootstrap_ratio
    with pytest.raises(ValueError):
        bootstrap_ratio([0,0],[0,0],['a','b'],B=100,seed=1)


def test_honest_confirm_predictions_end_to_end():
    from qualifyot.inference import honest_confirm_predictions
    n=200
    source=np.tile([.7,.3],(n,1)); target=np.tile([.5,.5],(n,1)); ref=source.copy(); cand=target.copy(); retained=cand.copy(); pats=np.array([f'p{i}' for i in range(n)])
    out,detail=honest_confirm_predictions(target,source,ref,cand,retained,pats,movement_margin=.01,alpha=.05)
    assert out.qualified
    assert len(detail['utility'])==n


def test_honest_cohort_conditional_scope_is_explicit():
    from qualifyot.inference import honest_confirmation_iut_cohort_conditional
    n=80
    a=np.full(n,.2); gm=np.full(n,.195)
    u=np.full(n,.04); cu=np.full(n,.05)
    nd=np.full(n,.03); cnd=np.full(n,.04)
    out=honest_confirmation_iut_cohort_conditional(gm,a,u,cu,nd,cnd,movement_margin=.01,alpha=.05)
    assert out.estimand_scope == 'cohort_conditional'
    assert out.mode == 'honest_confirmation_cohort_conditional'


def test_honest_population_scope_uses_k_state_global_support():
    from qualifyot.inference import honest_confirmation_iut_population
    n=500
    out=honest_confirmation_iut_population(
        np.full(n,.6), np.full(n,.3), np.full(n,.3), n_states=5,
        movement_margin=.01, alpha=.05,
    )
    assert out.estimand_scope == 'population'
    assert out.utility.lower_support == -.4
    assert out.utility.upper_support == .4
    assert out.qualified


def test_honest_population_bound_matches_closed_form():
    from qualifyot.inference import honest_confirmation_iut_population
    n=100
    x=.2; k=4; alpha=.05
    out=honest_confirmation_iut_population(
        np.full(n,.5), np.full(n,x), np.full(n,x), n_states=k,
        movement_margin=.01, alpha=alpha,
    )
    width=4.0/k
    expected=x-width*math.sqrt(math.log(1/alpha)/(2*n))
    assert abs(out.utility.lower_bound-expected) < 1e-12


def test_honest_confirm_predictions_can_return_both_scopes():
    from qualifyot.inference import honest_confirm_predictions
    n=200
    source=np.tile([.7,.3],(n,1)); target=np.tile([.5,.5],(n,1))
    ref=source.copy(); cand=target.copy(); retained=cand.copy()
    pats=np.array([f'p{i}' for i in range(n)])
    out,detail=honest_confirm_predictions(
        target,source,ref,cand,retained,pats,movement_margin=.01,alpha=.05,return_both=True
    )
    assert set(out)=={'cohort_conditional','population'}
    assert out['cohort_conditional'].estimand_scope=='cohort_conditional'
    assert out['population'].estimand_scope=='population'
    assert detail['n_states']==2


def test_population_honest_requires_simplex_arrays():
    import pytest
    from qualifyot.inference import honest_confirm_predictions
    n=10
    source=np.tile([.7,.3],(n,1)); target=np.tile([.5,.5],(n,1))
    ref=source.copy(); cand=target.copy(); retained=cand.copy(); retained[0]=[.8,.8]
    pats=np.array([f'p{i}' for i in range(n)])
    with pytest.raises(ValueError, match='simplex|sum'):
        honest_confirm_predictions(target,source,ref,cand,retained,pats,scope='population')


def test_localized_alias_matches_cohort_conditional():
    from qualifyot.inference import honest_confirmation_iut_localized, honest_confirmation_iut_cohort_conditional
    n=50
    a=np.full(n,.2); gm=np.full(n,.195)
    u=np.full(n,.04); cu=np.full(n,.05)
    nd=np.full(n,.03); cnd=np.full(n,.04)
    x=honest_confirmation_iut_localized(gm,a,u,cu,nd,cnd,movement_margin=.01,alpha=.05)
    y=honest_confirmation_iut_cohort_conditional(gm,a,u,cu,nd,cnd,movement_margin=.01,alpha=.05)
    assert x.to_dict()==y.to_dict()


def test_extended_contract_empty_fields_preserve_historical_hash():
    base=FrozenAnalysisContract('x','nested',('A','B'),'MAE',.01,(0,.5,1),'honest')
    explicit=FrozenAnalysisContract(
        'x','nested',('A','B'),'MAE',.01,(0,.5,1),'honest',
        prediction_task='',eligibility_rule='',preprocessing_protocol='',
        hyperparameter_protocol='',split_seed_rule=''
    )
    assert base.sha256==explicit.sha256


def test_extended_contract_fields_are_hash_sensitive():
    a=FrozenAnalysisContract('x','nested',('A','B'),'MAE',.01,(0,.5,1),'honest')
    b=FrozenAnalysisContract(
        'x','nested',('A','B'),'MAE',.01,(0,.5,1),'honest',
        prediction_task='baseline-to-followup composition',
        eligibility_rule='one source and one target per patient',
        preprocessing_protocol='training-only standardization',
        hyperparameter_protocol='fixed grid',split_seed_rule='sha256 60/40'
    )
    assert a.sha256 != b.sha256
    d=b.canonical_dict()
    assert d['prediction_task']=='baseline-to-followup composition'
    assert d['split_seed_rule']=='sha256 60/40'
