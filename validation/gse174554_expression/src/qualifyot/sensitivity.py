from __future__ import annotations
from pathlib import Path
from itertools import permutations
import json
import numpy as np
import pandas as pd
from .model import STATES,EDGES,mae,hellinger
from .experiment import run_lopo
from .utils import write_json


def all_label_permutations(states=STATES,base_edges=EDGES):
    """Return all 5! state-label permutations of the pre-specified graph.

    Because the pre-specified graph has two isolated vertices, 120 labelings correspond
    to 60 unique edge sets.  Both the labeling index and unique graph key are
    retained so the distinction is transparent rather than hidden.
    """
    states=tuple(states); out=[]
    for i,perm in enumerate(permutations(states)):
        mp=dict(zip(states,perm)); edges=tuple((mp[a],mp[b]) for a,b in base_edges)
        key='|'.join(f'{a}->{b}' for a,b in edges)
        out.append({'permutation_index':i,'mapping':mp,'edges':edges,'graph_key':key})
    return out


def threshold_stability(prior_audit, weight_fractions=(.50,.60,.70,.80,.90), pdr_mins=(.025,.05,.10), pdr_lcb_mins=(0,.01,.025)):
    """Sensitivity map; never changes the pre-specified primary classification."""
    a=dict(prior_audit); rows=[]
    for ef in weight_fractions:
        for pm in pdr_mins:
            for pl in pdr_lcb_mins:
                positive=(a['PDR_Hellinger']>pm and a['PDR_CI2.5']>pl and a['PUC_CI2.5']>0 and a['positive_weight_fold_fraction']>=ef and a['NPI_CI2.5']>0 and a['QualifyOT_minus_reference_MAE']<0)
                rows.append({'positive_weight_fraction_threshold':ef,'pdr_min':pm,'pdr_lcb_min':pl,'would_meet_full_positive_criterion':bool(positive)})
    return pd.DataFrame(rows)


def _decode_predictions(pdf, model):
    g=pdf[pdf.model==model].copy().sort_values(['patient_id','pair_id'])
    Y=np.vstack(g.target_json.map(json.loads)); Q=np.vstack(g.prediction_json.map(json.loads)); S=np.vstack(g.source_json.map(json.loads)); P=g.patient_id.to_numpy(str)
    return g,Y,Q,S,P


def fixed_oof_patient_influence(predictions_csv):
    """Leave-one-patient influence on *fixed OOF predictions* (not a refit)."""
    pdf=pd.read_csv(predictions_csv)
    _,Y,G,S,P=_decode_predictions(pdf,'GraphFlow'); _,_,N,_,_=_decode_predictions(pdf,'Reference'); _,_,A,_,_=_decode_predictions(pdf,'QualifyOT')
    up=np.unique(P); rows=[]
    for omitted in ['NONE']+up.tolist():
        keep=np.ones(len(P),bool) if omitted=='NONE' else P!=omitted
        pk=P[keep]; uk=np.unique(pk)
        def pe(v): return float(np.mean([np.asarray(v)[keep][pk==u].mean() for u in uk]))
        num=hellinger(G[keep],S[keep]); den=hellinger(Y[keep],S[keep])
        pdr=float(np.mean([num[pk==u].mean() for u in uk])/max(np.mean([den[pk==u].mean() for u in uk]),1e-15))
        rows.append({'omitted_patient':omitted,'patients_remaining':len(uk),'PDR_Hellinger':pdr,'PUC_reference_minus_prior':pe(mae(Y,N)-mae(Y,G)),'NPI_reference_minus_qualified':pe(mae(Y,N)-mae(Y,A))})
    return pd.DataFrame(rows)


def run_patient_balanced_sensitivity(pairs,cohort,outdir,bootstrap=5000,seed=20260811,positive_cfg=None,resume=True):
    return run_lopo(pairs,cohort,Path(outdir),bootstrap=bootstrap,seed=seed,positive_cfg=positive_cfg,resume=resume,patient_balanced=True,analysis_label='patient_balanced_fitting_sensitivity')


def run_graph_specificity(pairs,cohort,outdir,bootstrap=5000,seed=20260811,positive_cfg=None,resume=True,max_graphs=None):
    """Run the pre-specified protocol across all state-label permutations.

    This is intentionally expensive and therefore checkpointed graph-by-graph.
    The observed literature graph is recorded separately as permutation='OBS'.
    """
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True); rows=[]
    obs_dir=outdir/'graph_OBS'; _,_,a=run_lopo(pairs,cohort,obs_dir,bootstrap=bootstrap,seed=seed,positive_cfg=positive_cfg,resume=resume,edges=EDGES,analysis_label='graph_specificity_observed')
    rows.append({'labeling':'OBS','unique_graph_key':'|'.join(f'{a}->{b}' for a,b in EDGES),**a})
    specs=all_label_permutations();
    if max_graphs is not None: specs=specs[:int(max_graphs)]
    for j,spec in enumerate(specs):
        gd=outdir/f'graph_perm_{spec["permutation_index"]:03d}'
        _,_,aa=run_lopo(pairs,cohort,gd,bootstrap=bootstrap,seed=seed+1000+spec['permutation_index']*31,positive_cfg=positive_cfg,resume=resume,edges=spec['edges'],analysis_label=f'graph_permutation_{spec["permutation_index"]:03d}')
        rows.append({'labeling':spec['permutation_index'],'unique_graph_key':spec['graph_key'],'mapping_json':json.dumps(spec['mapping'],sort_keys=True),**aa})
        print(f'[{cohort}] graph specificity {j+1}/{len(specs)} complete',flush=True)
    df=pd.DataFrame(rows); df.to_csv(outdir/'graph_specificity_all_labelings.csv',index=False)
    obs_puc=float(df.loc[df.labeling.astype(str)=='OBS','PUC_reference_minus_prior'].iloc[0]); perm=df[df.labeling.astype(str)!='OBS']
    summary={'cohort':cohort,'observed_PUC':obs_puc,'n_label_permutations':len(perm),'n_unique_permuted_graphs':int(perm.unique_graph_key.nunique()),'empirical_tail_PUC_ge_observed':float((1+(perm.PUC_reference_minus_prior>=obs_puc).sum())/(1+len(perm)))}
    write_json(outdir/'graph_specificity_summary.json',summary)
    return df,summary


def cross_cohort_simultaneous_puc(result_dirs,B=10000,seed=20260811,alpha=.05):
    """Max-t simultaneous CIs for cohort-level PUC based on patient OOF contrasts."""
    data=[]
    for d in map(Path,result_dirs):
        pdf=pd.read_csv(d/'predictions.csv'); cohort=str(pdf.cohort.iloc[0])
        _,Y,G,_,P=_decode_predictions(pdf,'GraphFlow'); _,_,N,_,_=_decode_predictions(pdf,'Reference')
        up=np.unique(P); x=np.array([(mae(Y,N)-mae(Y,G))[P==p].mean() for p in up],float)
        data.append((cohort,x))
    rng=np.random.default_rng(seed); stats=[]; boot_means=[]
    for cohort,x in data:
        se=x.std(ddof=1)/np.sqrt(len(x)) if len(x)>1 else np.nan; stats.append((cohort,x.mean(),se,len(x)))
        idx=rng.integers(0,len(x),size=(B,len(x))); boot_means.append(x[idx].mean(1))
    z=[]
    for (_,m,se,_),bm in zip(stats,boot_means):
        z.append(np.abs((bm-m)/max(se,1e-15)))
    q=float(np.quantile(np.max(np.vstack(z),axis=0),1-alpha))
    rows=[]
    for cohort,m,se,n in stats: rows.append({'cohort':cohort,'PUC':m,'SE':se,'simultaneous_q':q,'sim_CI_low':m-q*se,'sim_CI_high':m+q*se,'patients':n})
    return pd.DataFrame(rows)


def resample_patients_with_replacement(pairs,rng):
    pats=np.array(sorted(pairs.patient_id.astype(str).unique())); draw=rng.choice(pats,size=len(pats),replace=True); chunks=[]
    for j,p in enumerate(draw):
        g=pairs[pairs.patient_id.astype(str)==p].copy(); new=f'boot{j:03d}_{p}'; g['patient_id']=new; g['pair_id']=g['pair_id'].astype(str).map(lambda x:f'{new}:{x}'); chunks.append(g)
    return pd.concat(chunks,ignore_index=True)


def full_pipeline_bootstrap(pairs,cohort,outdir,n_boot=200,inner_bootstrap=1000,seed=20260811,positive_cfg=None,patient_balanced=False,resume=True):
    """Computationally intensive full-pipeline patient-resampling stability.

    Each replicate resamples physical patients, assigns unique bootstrap patient
    identifiers, then reruns the complete LOPO qualification pipeline.  This is
    distinct from bootstrapping pre-specified OOF predictions.
    """
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(seed); rows=[]
    for b in range(int(n_boot)):
        jf=outdir/f'replicate_{b:04d}_audit.json'
        if resume and jf.exists(): rows.append(json.loads(jf.read_text(encoding='utf-8'))); continue
        pb=resample_patients_with_replacement(pairs,rng)
        rd=outdir/f'replicate_{b:04d}'
        _,_,a=run_lopo(pb,f'{cohort}_boot{b:04d}',rd,bootstrap=inner_bootstrap,seed=seed+b*1009,positive_cfg=positive_cfg,resume=resume,patient_balanced=patient_balanced,analysis_label='full_pipeline_patient_bootstrap')
        rec={'replicate':b,'PDR_Hellinger':a['PDR_Hellinger'],'PUC':a['PUC_reference_minus_prior'],'NPI':a['NPI_reference_minus_qualified'],'positive_weight_fold_fraction':a['positive_weight_fold_fraction'],'qualified_prior_case':a['qualified_prior_case'],'decision':a['decision']}
        write_json(jf,rec); rows.append(rec)
    df=pd.DataFrame(rows); df.to_csv(outdir/'full_pipeline_bootstrap_summary.csv',index=False)
    return df
