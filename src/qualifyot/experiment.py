from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .model import (
    STATES, EDGES, incidence, graph_hash, feature_matrix, fit_graphflow,
    choose_null_loocv, fit_null, predict_null, mae, hellinger, tv, js,
    residual_flow,
)
from .weight_selection import patient_curves, one_se_msw
from .utils import patient_equal, write_json


def bootstrap_contrast(values,patients,B=5000,seed=0):
    pats=np.asarray(patients).astype(str); up=np.unique(pats)
    x=np.array([np.asarray(values)[pats==p].mean() for p in up],float)
    rng=np.random.default_rng(seed); idx=rng.integers(0,len(up),size=(B,len(up))); boot=x[idx].mean(1)
    return float(x.mean()),float(np.quantile(boot,.025)),float(np.quantile(boot,.975))


def bootstrap_ratio(num,den,patients,B=5000,seed=0):
    pats=np.asarray(patients).astype(str); up=np.unique(pats)
    a=np.array([np.asarray(num)[pats==p].mean() for p in up],float)
    b=np.array([np.asarray(den)[pats==p].mean() for p in up],float)
    rng=np.random.default_rng(seed); idx=rng.integers(0,len(up),size=(B,len(up)))
    v=a[idx].mean(1)/np.maximum(b[idx].mean(1),1e-15)
    return float(a.mean()/max(b.mean(),1e-15)),float(np.quantile(v,.025)),float(np.quantile(v,.975))


def _inner_oof(X,S,T,P,boot,seed,flow_labels=None,Bmat=None,edges=None,patient_balanced=False):
    prior=np.zeros_like(T); null=np.zeros_like(T); kinds=[]; Pall=P.astype(str)
    for h in np.unique(Pall):
        te=Pall==h; tr=~te; fl=None if flow_labels is None else np.asarray(flow_labels)[tr]
        gm=fit_graphflow(
            X[tr],S[tr],T[tr],flow_labels=fl,Bmat=Bmat,edges=edges,
            patients=P[tr],patient_balanced=patient_balanced,
        )
        prior[te]=gm.predict(X[te],S[te])
        nk,_=choose_null_loocv(S[tr],T[tr],P[tr],patient_balanced=patient_balanced) if len(np.unique(P[tr]))>=4 else ('Persistence',{})
        nm=fit_null(nk,S[tr],T[tr],patients=P[tr],patient_balanced=patient_balanced)
        null[te]=predict_null(nm,S[te]); kinds.append(nk)
    return prior,null,kinds


def _pred_arrays(pdf,model):
    g=pdf[pdf.model==model].copy()
    return g, np.vstack(g.target_json.map(json.loads)), np.vstack(g.prediction_json.map(json.loads)), g.patient_id.to_numpy(str)


def run_lopo(
    pairs,cohort,outdir:Path,bootstrap=5000,seed=20260810,positive_cfg=None,resume=True,
    edges=None,patient_balanced=False,analysis_label='pair_equal_primary'
):
    """Patient-level LOPO qualification.

    The primary analysis uses pair-equal model fitting and patient-equal
    evaluation. A patient-balanced fitting sensitivity can be run with
    ``patient_balanced=True`` without overwriting the primary result.  Candidate flow labels are residual-GraphFlow labels:
    B f + r = delta p with f>=0.
    """
    if len(pairs)==0: raise ValueError('no eligible longitudinal pairs')
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    edges=list(EDGES if edges is None else [tuple(e) for e in edges]); Bmat=incidence(STATES,edges)
    X,S,T=feature_matrix(pairs); P=pairs.patient_id.to_numpy(str); ids=pairs.pair_id.to_numpy(str); grid=np.round(np.arange(0,1.00001,.01),2)
    flow_labels=np.vstack([residual_flow(s,t,Bmat=Bmat)[0] for s,t in zip(S,T)])
    cache=outdir/'fold_cache'; cache.mkdir(exist_ok=True)
    sig={
        'cohort':cohort,'analysis_label':analysis_label,'bootstrap':int(bootstrap),'seed':int(seed),
        'patients':int(len(np.unique(P))),'pairs':int(len(pairs)),'pair_ids':ids.tolist(),'patient_ids':P.tolist(),
        'states':list(STATES),'edges':[list(e) for e in edges],'graph_hash':graph_hash(STATES,edges),
        'flow_target':'residual_graphflow','patient_balanced_fitting':bool(patient_balanced),
    }
    sig_path=cache/'RUN_SIGNATURE.json'
    if sig_path.exists():
        old=json.loads(sig_path.read_text(encoding='utf-8'))
        if old!=sig:
            for q in cache.glob('fold_*'): q.unlink()
    write_json(sig_path,sig)
    preds=[]; folds=[]; weights=[]; up=np.unique(P)
    for fold,h in enumerate(up):
        p_pred=cache/f'fold_{fold:03d}_predictions.csv'; p_fold=cache/f'fold_{fold:03d}_summary.json'; p_wgt=cache/f'fold_{fold:03d}_weights.csv'
        if resume and p_pred.exists() and p_fold.exists() and p_wgt.exists():
            preds.extend(pd.read_csv(p_pred).to_dict('records')); folds.append(json.loads(p_fold.read_text(encoding='utf-8'))); weights.extend(pd.read_csv(p_wgt).to_dict('records'))
            print(f'[{cohort}|{analysis_label}] fold {fold+1}/{len(up)} patient={h} RESUME',flush=True); continue
        te=P==h; tr=~te
        gm=fit_graphflow(
            X[tr],S[tr],T[tr],flow_labels=flow_labels[tr],Bmat=Bmat,edges=edges,
            patients=P[tr],patient_balanced=patient_balanced,
        ); pr=gm.predict(X[te],S[te])
        null_kind,null_scores=choose_null_loocv(S[tr],T[tr],P[tr],patient_balanced=patient_balanced)
        nm=fit_null(null_kind,S[tr],T[tr],patients=P[tr],patient_balanced=patient_balanced); nu=predict_null(nm,S[te])
        po,no,_=_inner_oof(
            X[tr],S[tr],T[tr],P[tr],bootstrap,seed+fold*11,flow_labels=flow_labels[tr],
            Bmat=Bmat,edges=edges,patient_balanced=patient_balanced,
        )
        _,D,R=patient_curves(T[tr],no,po,P[tr],grid); g=one_se_msw(D,R,grid,.05,0,bootstrap,seed+fold*7919,1)
        final=(1-g['weight'])*nu+g['weight']*pr
        fp=[]
        for j,ix in enumerate(np.flatnonzero(te)):
            for model,pred in [('Persistence',S[ix:ix+1]),('Reference',nu[j:j+1]),('GraphFlow',pr[j:j+1]),('QualifyOT',final[j:j+1])]:
                fp.append({
                    'cohort':cohort,'analysis_label':analysis_label,'outer_fold':fold,'patient_id':P[ix],'pair_id':ids[ix],'model':model,
                    'target_json':json.dumps(T[ix].tolist()),'prediction_json':json.dumps(pred[0].tolist()),'source_json':json.dumps(S[ix].tolist()),
                })
        fs={
            'cohort':cohort,'analysis_label':analysis_label,'outer_fold':fold,'held_out_patient':h,'null_kind':null_kind,'mixing_weight':g['weight'],
            'best_weight':g['best_weight'],'max_eligible_weight':g['max_eligible_weight'],'se_best':g['se_best'],'max_t_q':g['max_t_q'],'test_pairs':int(te.sum()),
            'patient_balanced_fitting':bool(patient_balanced),'graph_hash':sig['graph_hash'],**{f'nullcv_{k}':v for k,v in null_scores.items()},
        }
        fe=[{'cohort':cohort,'analysis_label':analysis_label,'outer_fold':fold,'mixing_weight':w,'risk':g['risk'][j],'ucb':g['ucb'][j]} for j,w in enumerate(grid)]
        pd.DataFrame(fp).to_csv(p_pred,index=False); write_json(p_fold,fs); pd.DataFrame(fe).to_csv(p_wgt,index=False)
        preds.extend(fp); folds.append(fs); weights.extend(fe)
        print(f"[{cohort}|{analysis_label}] fold {fold+1}/{len(up)} patient={h} weight={g['weight']:.2f} reference={null_kind}",flush=True)
    pdf=pd.DataFrame(preds); fdf=pd.DataFrame(folds).sort_values('outer_fold'); edf=pd.DataFrame(weights).sort_values(['outer_fold','mixing_weight'])
    metrics=[]
    for model,gdf in pdf.groupby('model'):
        Y=np.vstack(gdf.target_json.map(json.loads)); Q=np.vstack(gdf.prediction_json.map(json.loads)); pats=gdf.patient_id.to_numpy(str)
        row={'cohort':cohort,'analysis_label':analysis_label,'model':model,'patients':gdf.patient_id.nunique(),'pairs':len(gdf),'patient_balanced_fitting':bool(patient_balanced)}
        for name,fn in [('MAE',mae),('Hellinger',hellinger),('TV',tv),('JS',js)]: row['patient_equal_'+name]=patient_equal(fn(Y,Q),pats)
        metrics.append(row)
    mdf=pd.DataFrame(metrics).sort_values('patient_equal_MAE')

    gg,Y,G,pats=_pred_arrays(pdf,'GraphFlow'); _,_,N,_=_pred_arrays(pdf,'Reference'); _,_,S0,_=_pred_arrays(pdf,'Persistence'); _,_,A,_=_pred_arrays(pdf,'QualifyOT')
    pdr=bootstrap_ratio(hellinger(G,S0),hellinger(Y,S0),pats,bootstrap,seed+700)
    puc=bootstrap_contrast(mae(Y,N)-mae(Y,G),pats,bootstrap,seed+701)
    puc_persistence=bootstrap_contrast(mae(Y,S0)-mae(Y,G),pats,bootstrap,seed+704)
    npi=bootstrap_contrast(mae(Y,N)-mae(Y,A),pats,bootstrap,seed+703)
    final_r=mdf.loc[mdf.model=='QualifyOT','patient_equal_MAE'].iloc[0]; reference_r=mdf.loc[mdf.model=='Reference','patient_equal_MAE'].iloc[0]
    fold_positive=float((fdf.mixing_weight>0).mean())
    cfg=positive_cfg or {'pdr_min':.05,'pdr_lcb_min':.01,'puc_lcb_min':0,'npi_lcb_min':0,'positive_weight_fold_fraction_min':.70,'require_final_risk_below_reference':True}
    expressive=bool(pdr[0]>cfg['pdr_min'] and pdr[1]>cfg['pdr_lcb_min']); forced_beneficial=bool(puc[1]>cfg.get('puc_lcb_min',0.0)); forced_harmful=bool(puc[2]<0.0); retained_beneficial=bool(npi[1]>cfg.get('npi_lcb_min',0.0))
    positive=bool(expressive and forced_beneficial and fold_positive>=cfg['positive_weight_fold_fraction_min'] and retained_beneficial and ((final_r<reference_r) if cfg['require_final_risk_below_reference'] else True))
    if positive: decision='Distinct+QualifiedForRetention'
    elif not expressive: decision='NegligibleDeviationFromReference'
    elif forced_harmful: decision='Distinct+AdversePredictiveValue'
    elif forced_beneficial: decision='Promising+EvidentiallyInsufficient'
    else: decision='EquivocalPredictiveEvidence'
    audit={
        'cohort':cohort,'analysis_label':analysis_label,'flow_target':'residual_graphflow','graph_hash':sig['graph_hash'],'patient_balanced_fitting':bool(patient_balanced),
        'PDR_Hellinger':pdr[0],'PDR_CI2.5':pdr[1],'PDR_CI97.5':pdr[2],
        'PUC_reference_minus_prior':puc[0],'PUC_CI2.5':puc[1],'PUC_CI97.5':puc[2],
        'PUC_persistence_minus_prior':puc_persistence[0],'PUC_P_CI2.5':puc_persistence[1],'PUC_P_CI97.5':puc_persistence[2],
        'NPI_reference_minus_qualified':npi[0],'NPI_CI2.5':npi[1],'NPI_CI97.5':npi[2],
        'positive_weight_fold_fraction':fold_positive,'QualifyOT_minus_reference_MAE':float(final_r-reference_r),
        'qualified_prior_case':positive,'decision':decision,
    }
    gcfs=np.array([residual_flow(s,t,Bmat=Bmat)[2] for s,t in zip(S,T)]); gcf=bootstrap_contrast(gcfs,P,bootstrap,seed+702)
    audit.update({'GCF':gcf[0],'GCF_CI2.5':gcf[1],'GCF_CI97.5':gcf[2]})
    mdf.to_csv(outdir/'model_metrics.csv',index=False); fdf.to_csv(outdir/'outer_folds.csv',index=False); edf.to_csv(outdir/'mixing_weight_curves.csv',index=False); pdf.to_csv(outdir/'predictions.csv',index=False); write_json(outdir/'prior_audit.json',audit)
    return mdf,fdf,audit
