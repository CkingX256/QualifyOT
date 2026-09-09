from __future__ import annotations
import numpy as np
import pandas as pd
from .candidate_api import CandidatePredictor
from .model import STATES, feature_matrix, choose_null_loocv, fit_null, predict_null, mae, hellinger
from .weight_selection import patient_curves, one_se_msw
from .utils import patient_equal
from .experiment import bootstrap_contrast, bootstrap_ratio
from .decision import IUTConfig, classify_iut, classify_iut_contrast, retained_risk_improvement
from .inference import movement_margin_evidence, movement_patient_components

DEFAULT_CFG={'pdr_min':.05,'pdr_lcb_min':.01,'puc_lcb_min':0.0,'npi_lcb_min':0.0,
             'positive_weight_fold_fraction_min':.70,'require_final_risk_below_reference':True}

def _resolve_states(candidate,states=None):
    if states is not None: return list(states)
    cs=getattr(candidate,'states',None)
    return list(STATES if cs is None else cs)

def _resolve_reference(kind, S, T, P, patient_balanced=False):
    allowed={"nested","Persistence","CohortMean","MeanDelta"}
    if kind not in allowed:
        raise ValueError(f"reference_rule must be one of {sorted(allowed)}")
    if kind == "nested":
        selected,_=choose_null_loocv(S,T,P,patient_balanced=patient_balanced) if len(np.unique(P))>=4 else ("Persistence",{})
    else:
        selected=kind
    return selected, fit_null(selected,S,T,patients=P,patient_balanced=patient_balanced)


def _inner_oof_generic(train_pairs: pd.DataFrame, candidate: CandidatePredictor, patient_balanced=False,states=None,reference_rule="nested"):

    states=_resolve_states(candidate,states); X,S,T=feature_matrix(train_pairs,states=states); P=train_pairs.patient_id.to_numpy(str)
    prior=np.zeros_like(T); ref=np.zeros_like(T)
    for h in np.unique(P):
        te=P==h; tr=~te
        c=candidate.fresh().fit(train_pairs.loc[tr].reset_index(drop=True)); prior[te]=c.predict(train_pairs.loc[te].reset_index(drop=True))
        nk,nm=_resolve_reference(reference_rule,S[tr],T[tr],P[tr],patient_balanced=patient_balanced); ref[te]=predict_null(nm,S[te])
    return prior,ref

def run_generic_lopo(pairs: pd.DataFrame,candidate: CandidatePredictor,bootstrap=1000,seed=20260823,
                     positive_cfg=None,patient_balanced=False,states=None,iut_cfg:IUTConfig|None=None,reference_rule="nested",
                     orthogonal_profiles=False,profile_draws=5000,return_predictions=False,retention_selector="mean",
                     tail_fraction=0.20,tail_harm_margin=0.0,
                     movement_contrast_method="t",use_movement_contrast_for_core=True):
    """Ontology- and architecture-agnostic patient-level qualification.

    The candidate is refitted inside every outer and inner patient split. The
    formal manuscript paths in ``experiment.run_lopo`` remain frozen; this
    generic engine is the portability layer for external predictors. The
    margin-contrast Movement margin contrast is the public default. Set
    ``use_movement_contrast_for_core=False`` only for explicit historical
    reproduction of the legacy ratio-LCB core.
    """
    pairs=pairs.reset_index(drop=True).copy(); cfg=positive_cfg or DEFAULT_CFG; states=_resolve_states(candidate,states)
    X,S,T=feature_matrix(pairs,states=states); P=pairs.patient_id.to_numpy(str); grid=np.round(np.arange(0,1.00001,.01),2)
    rows=[]; fold_rows=[]
    for fold,h in enumerate(np.unique(P)):
        te=P==h; tr=~te; train=pairs.loc[tr].reset_index(drop=True); test=pairs.loc[te].reset_index(drop=True)
        c=candidate.fresh().fit(train); pr=c.predict(test)
        nk,nm=_resolve_reference(reference_rule,S[tr],T[tr],P[tr],patient_balanced=patient_balanced); nu=predict_null(nm,S[te])
        po,no=_inner_oof_generic(train,candidate,patient_balanced=patient_balanced,states=states,reference_rule=reference_rule)
        _,_,Tin=feature_matrix(train,states=states); Pin=train.patient_id.to_numpy(str)
        _,D,R=patient_curves(Tin,no,po,Pin,grid)
        if retention_selector == 'mean':
            g=one_se_msw(D,R,grid,.05,0,bootstrap,seed+fold*7919,1)
        elif retention_selector == 'tail':
            from .robust_weight_selection import tail_safe_msw
            g=tail_safe_msw(D,R,grid,alpha_mean=.05,mean_harm_margin=0.0,tail_fraction=tail_fraction,
                            alpha_tail=.05,tail_harm_margin=tail_harm_margin,bootstrap=max(100,int(bootstrap)),
                            seed=seed+fold*7919,one_se_lambda=1.0)
        else:
            raise ValueError("retention_selector must be 'mean' or 'tail'")
        final=(1-g['weight'])*nu+g['weight']*pr
        for local,ix in enumerate(np.flatnonzero(te)):
            rows.append(dict(patient_id=P[ix],pair_id=pairs.pair_id.iloc[ix],target=T[ix],source=S[ix],reference=nu[local],candidate=pr[local],qualified=final[local]))
        fd=dict(outer_fold=fold,patient_id=h,mixing_weight=g['weight'],reference_kind=nk)
        # Candidate-internal design diagnostics are collected only after fitting
        # on outer-training patients.  They never alter the outer held-out
        # target score and are serialized as descriptive audit fields.
        if hasattr(c,'selected_graph_weights'):
            try: fd['candidate_graph_weights']=c.selected_graph_weights()
            except RuntimeError: pass
        sel=getattr(c,'selection_result',None)
        if sel is not None and hasattr(sel,'to_dict'):
            fd['candidate_selection']=sel.to_dict()
        if hasattr(c,'edge_weights'):
            try: fd['candidate_edge_weights']=c.edge_weights()
            except RuntimeError: pass
        if hasattr(c,'candidate_diagnostics'):
            try: fd['candidate_diagnostics']=c.candidate_diagnostics()
            except RuntimeError: pass
        fold_rows.append(fd)
    pats=np.array([r['patient_id'] for r in rows],str); Y=np.vstack([r['target'] for r in rows]); S0=np.vstack([r['source'] for r in rows]); N=np.vstack([r['reference'] for r in rows]); C=np.vstack([r['candidate'] for r in rows]); A=np.vstack([r['qualified'] for r in rows])
    movement_num=hellinger(C,S0); movement_den=hellinger(Y,S0)
    pdr=bootstrap_ratio(movement_num,movement_den,pats,bootstrap,seed+700)
    puc=bootstrap_contrast(mae(Y,N)-mae(Y,C),pats,bootstrap,seed+701); npi=bootstrap_contrast(mae(Y,N)-mae(Y,A),pats,bootstrap,seed+703)
    _iut_cfg=iut_cfg or IUTConfig()
    movement_result=movement_margin_evidence(
        movement_num,movement_den,pats,movement_margin=_iut_cfg.movement_margin,
        alpha=_iut_cfg.component_alpha,method=movement_contrast_method,
        bootstrap=max(100,int(bootstrap)),seed=seed+704,
    )
    ref_r=patient_equal(mae(Y,N),pats); q_r=patient_equal(mae(Y,A),pats); fdf=pd.DataFrame(fold_rows); pos=float((fdf.mixing_weight>0).mean())
    gates={'PDR_point':pdr[0]>cfg['pdr_min'],'PDR_LCB':pdr[1]>cfg['pdr_lcb_min'],'PUC_LCB':puc[1]>cfg['puc_lcb_min'],'positive_weight_folds':pos>=cfg['positive_weight_fold_fraction_min'],'NPI_LCB':npi[1]>cfg['npi_lcb_min'],'final_risk':(q_r<ref_r) if cfg['require_final_risk_below_reference'] else True}
    qualified=bool(all(gates.values()))
    if use_movement_contrast_for_core:
        core=classify_iut_contrast(
            movement_contrast_lcb=movement_result.contrast_lcb,
            utility_lcb=puc[1],utility_ucb=puc[2],retention_lcb=npi[1],
            positive_weight_fraction=pos,cfg=_iut_cfg,
        )
    else:
        core=classify_iut(
            pdr_lcb=pdr[1],puc_lcb=puc[1],puc_ucb=puc[2],npi_lcb=npi[1],
            positive_weight_fraction=pos,cfg=_iut_cfg,
        )
    # Patient-equal influence table for transparent diagnostics. Positive NPI contribution
    # means the retained prediction improves on the selected reference for that patient.
    prow=[]
    ref_loss=mae(Y,N); cand_loss=mae(Y,C); ret_loss=mae(Y,A)
    _ma,_mb,_mg=movement_patient_components(movement_num,movement_den,pats,movement_margin=_iut_cfg.movement_margin)
    _up=np.unique(pats)
    for j,pid in enumerate(_up):
        m=pats==pid
        prow.append({'patient_id':str(pid),
                     'reference_loss':float(ref_loss[m].mean()),
                     'candidate_loss':float(cand_loss[m].mean()),
                     'retained_loss':float(ret_loss[m].mean()),
                     'movement_numerator':float(_ma[j]),
                     'movement_denominator':float(_mb[j]),
                     'movement_margin_contribution':float(_mg[j]),
                     'PUC_contribution':float((ref_loss[m]-cand_loss[m]).mean()),
                     'NPI_contribution':float((ref_loss[m]-ret_loss[m]).mean())})
    pdf=pd.DataFrame(prow)
    # Audit the algebraic identity that makes the legacy final-risk gate
    # redundant in the principled core decision.
    retained_risk_improvement(float(npi[0]),float(ref_r),float(q_r),atol=1e-10)
    out={'candidate':candidate.name,'states':'|'.join(states),'patients':len(np.unique(P)),'pairs':len(pairs),'PDR':pdr[0],'PDR_lo':pdr[1],'PDR_hi':pdr[2],'PUC':puc[0],'PUC_lo':puc[1],'PUC_hi':puc[2],'NPI':npi[0],'NPI_lo':npi[1],'NPI_hi':npi[2],'positive_weight_fold_fraction':pos,'reference_risk':ref_r,'qualified_risk':q_r,
            'qualified':qualified,'legacy_qualified':qualified,'gates':gates,
            'core_qualified':core.qualified,'core_evidence_state':core.label,'core_axes':dict(core.axes),
            'evidence_deficit':core.deficits.to_dict(),'retention_stability':pos,'core_rule':core.to_dict(),
            'movement_margin_evidence':movement_result.to_dict(),
            'movement_core_mode':'margin_contrast' if use_movement_contrast_for_core else 'legacy_ratio_lcb',
            'reference_rule':reference_rule,'retention_selector':retention_selector,'folds':fdf,'patient_influence':pdf}
    if return_predictions:
        out['prediction_rows']=pd.DataFrame([{
            'patient_id':r['patient_id'],'pair_id':r['pair_id'],
            'target':np.asarray(r['target'],float),'source':np.asarray(r['source'],float),
            'reference':np.asarray(r['reference'],float),'candidate':np.asarray(r['candidate'],float),
            'retained':np.asarray(r['qualified'],float),
        } for r in rows])
    if orthogonal_profiles:
        from .bayesian_evidence import bayesian_bootstrap_mean
        from .benchmarks import benchmark_patient_effects
        u=pdf['PUC_contribution'].to_numpy(float); r=pdf['NPI_contribution'].to_numpy(float)
        harm=-r
        k=max(1,int(np.ceil(.20*len(harm))))
        out['orthogonal_profiles']={
            'bayesian_bootstrap_utility':bayesian_bootstrap_mean(u,draws=int(profile_draws),seed=seed+901).to_dict(),
            'bayesian_bootstrap_retention':bayesian_bootstrap_mean(r,draws=int(profile_draws),seed=seed+902).to_dict(),
            'utility_benchmarks':benchmark_patient_effects(u,bootstrap=max(200,min(int(bootstrap),2000)),permutations=5000,bayes_draws=int(profile_draws),seed=seed+903).to_dict(),
            'retention_benchmarks':benchmark_patient_effects(r,bootstrap=max(200,min(int(bootstrap),2000)),permutations=5000,bayes_draws=int(profile_draws),seed=seed+904).to_dict(),
            'tail_safety':{
                'tail_fraction':.20,
                'harmed_patient_fraction':float(np.mean(harm>0)),
                'upper_tail_cvar_harm':float(np.sort(harm)[-k:].mean()),
                'worst_patient_harm':float(np.max(harm)),
            },
        }
    return out
