from __future__ import annotations
import numpy as np
import pandas as pd
from .candidate_api import CandidatePredictor
from .model import STATES, feature_matrix, choose_null_loocv, fit_null, predict_null, mae, hellinger
from .weight_selection import patient_curves, one_se_msw
from .utils import patient_equal
from .experiment import bootstrap_contrast, bootstrap_ratio

DEFAULT_CFG={'pdr_min':.05,'pdr_lcb_min':.01,'puc_lcb_min':0.0,'npi_lcb_min':0.0,
             'positive_weight_fold_fraction_min':.70,'require_final_risk_below_reference':True}

def _resolve_states(candidate,states=None):
    if states is not None: return list(states)
    cs=getattr(candidate,'states',None)
    return list(STATES if cs is None else cs)

def _inner_oof_generic(train_pairs: pd.DataFrame, candidate: CandidatePredictor, patient_balanced=False,states=None):
    states=_resolve_states(candidate,states); X,S,T=feature_matrix(train_pairs,states=states); P=train_pairs.patient_id.to_numpy(str)
    prior=np.zeros_like(T); ref=np.zeros_like(T)
    for h in np.unique(P):
        te=P==h; tr=~te
        c=candidate.fresh().fit(train_pairs.loc[tr].reset_index(drop=True)); prior[te]=c.predict(train_pairs.loc[te].reset_index(drop=True))
        nk,_=choose_null_loocv(S[tr],T[tr],P[tr],patient_balanced=patient_balanced) if len(np.unique(P[tr]))>=4 else ('Persistence',{})
        nm=fit_null(nk,S[tr],T[tr],patients=P[tr],patient_balanced=patient_balanced); ref[te]=predict_null(nm,S[te])
    return prior,ref

def run_generic_lopo(pairs: pd.DataFrame,candidate: CandidatePredictor,bootstrap=1000,seed=20260823,
                     positive_cfg=None,patient_balanced=False,states=None):
    """Ontology- and architecture-agnostic patient-level qualification.

    The candidate is refitted inside every outer and inner patient split. The
    formal manuscript paths in ``experiment.run_lopo`` remain frozen; this
    generic engine is the portability layer for external predictors.
    """
    pairs=pairs.reset_index(drop=True).copy(); cfg=positive_cfg or DEFAULT_CFG; states=_resolve_states(candidate,states)
    X,S,T=feature_matrix(pairs,states=states); P=pairs.patient_id.to_numpy(str); grid=np.round(np.arange(0,1.00001,.01),2)
    rows=[]; fold_rows=[]
    for fold,h in enumerate(np.unique(P)):
        te=P==h; tr=~te; train=pairs.loc[tr].reset_index(drop=True); test=pairs.loc[te].reset_index(drop=True)
        c=candidate.fresh().fit(train); pr=c.predict(test)
        nk,_=choose_null_loocv(S[tr],T[tr],P[tr],patient_balanced=patient_balanced)
        nm=fit_null(nk,S[tr],T[tr],patients=P[tr],patient_balanced=patient_balanced); nu=predict_null(nm,S[te])
        po,no=_inner_oof_generic(train,candidate,patient_balanced=patient_balanced,states=states)
        _,_,Tin=feature_matrix(train,states=states); Pin=train.patient_id.to_numpy(str)
        _,D,R=patient_curves(Tin,no,po,Pin,grid); g=one_se_msw(D,R,grid,.05,0,bootstrap,seed+fold*7919,1)
        final=(1-g['weight'])*nu+g['weight']*pr
        for local,ix in enumerate(np.flatnonzero(te)):
            rows.append(dict(patient_id=P[ix],pair_id=pairs.pair_id.iloc[ix],target=T[ix],source=S[ix],reference=nu[local],candidate=pr[local],qualified=final[local]))
        fold_rows.append(dict(outer_fold=fold,patient_id=h,mixing_weight=g['weight'],reference_kind=nk))
    pats=np.array([r['patient_id'] for r in rows],str); Y=np.vstack([r['target'] for r in rows]); S0=np.vstack([r['source'] for r in rows]); N=np.vstack([r['reference'] for r in rows]); C=np.vstack([r['candidate'] for r in rows]); A=np.vstack([r['qualified'] for r in rows])
    pdr=bootstrap_ratio(hellinger(C,S0),hellinger(Y,S0),pats,bootstrap,seed+700); puc=bootstrap_contrast(mae(Y,N)-mae(Y,C),pats,bootstrap,seed+701); npi=bootstrap_contrast(mae(Y,N)-mae(Y,A),pats,bootstrap,seed+703)
    ref_r=patient_equal(mae(Y,N),pats); q_r=patient_equal(mae(Y,A),pats); fdf=pd.DataFrame(fold_rows); pos=float((fdf.mixing_weight>0).mean())
    gates={'PDR_point':pdr[0]>cfg['pdr_min'],'PDR_LCB':pdr[1]>cfg['pdr_lcb_min'],'PUC_LCB':puc[1]>cfg['puc_lcb_min'],'positive_weight_folds':pos>=cfg['positive_weight_fold_fraction_min'],'NPI_LCB':npi[1]>cfg['npi_lcb_min'],'final_risk':(q_r<ref_r) if cfg['require_final_risk_below_reference'] else True}
    qualified=bool(all(gates.values()))
    # Patient-equal influence table for transparent diagnostics. Positive NPI contribution
    # means the retained prediction improves on the selected reference for that patient.
    prow=[]
    ref_loss=mae(Y,N); cand_loss=mae(Y,C); ret_loss=mae(Y,A)
    for pid in np.unique(pats):
        m=pats==pid
        prow.append({'patient_id':str(pid),
                     'reference_loss':float(ref_loss[m].mean()),
                     'candidate_loss':float(cand_loss[m].mean()),
                     'retained_loss':float(ret_loss[m].mean()),
                     'PUC_contribution':float((ref_loss[m]-cand_loss[m]).mean()),
                     'NPI_contribution':float((ref_loss[m]-ret_loss[m]).mean())})
    pdf=pd.DataFrame(prow)
    return {'candidate':candidate.name,'states':'|'.join(states),'patients':len(np.unique(P)),'pairs':len(pairs),'PDR':pdr[0],'PDR_lo':pdr[1],'PDR_hi':pdr[2],'PUC':puc[0],'PUC_lo':puc[1],'PUC_hi':puc[2],'NPI':npi[0],'NPI_lo':npi[1],'NPI_hi':npi[2],'positive_weight_fold_fraction':pos,'reference_risk':ref_r,'qualified_risk':q_r,'qualified':qualified,'gates':gates,'folds':fdf,'patient_influence':pdf}
