from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd

from qualifyot.candidates import RobustBlendCandidate
from qualifyot.candidate_api import GraphFlowCandidate
from qualifyot.adaptive_candidates import MultiScaleDeltaCandidate, PatientMultiScaleDeltaCandidate
from qualifyot.generic_engine import run_generic_lopo
from qualifyot.inference import one_sided_lcb
from qualifyot.model import mae, hellinger

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'processed_pairs'
OUT=ROOT/'results'/'real_data_robustness'
OUT.mkdir(parents=True,exist_ok=True)

SPECS=[
 ('GSE123813','GSE123813_pairs.csv','robust',None),
 ('GSE272993_9state','GSE272993_9state_pairs.csv','robust',None),
 ('GSE236581','GSE236581_pairs.csv','robust',None),
 ('GSE120575','GSE120575_pairs.csv','robust',None),
 ('GSE179994','GSE179994_pairs.csv','robust',None),
 ('GSE175522','GSE175522_D0_D7_Bcell_pairs.csv','robust',None),
 ('HCC','GSE229772_HCC_primary_pairs.csv','robust',None),
 ('AML','GSE235063_AML_primary_pairs.csv','graph',[('Monocytic','GMP'),('GMP','Progenitor'),('Progenitor','HSC')]),
]
REFS=['nested','Persistence','CohortMean','MeanDelta']

def states_of(df): return [c.split('__',1)[1] for c in df.columns if c.startswith('source__')]
def candidate_of(kind,states,edges):
    if kind=='robust': return RobustBlendCandidate(states=states)
    return GraphFlowCandidate(states=states,edges=edges,alpha=1.0,patient_balanced=False,name='GraphFlow')

def patient_reduce(values,pids):
    pids=np.asarray(pids,str); values=np.asarray(values,float)
    return np.asarray([values[pids==p].mean() for p in np.unique(pids)],float)

def brier(y,p):
    return np.mean((np.asarray(y,float)-np.asarray(p,float))**2,axis=1)

def loss_sensitivity(predrows, movement_pass):
    pids=predrows.patient_id.astype(str).to_numpy()
    Y=np.vstack(predrows.target.to_numpy()); R=np.vstack(predrows.reference.to_numpy())
    C=np.vstack(predrows.candidate.to_numpy()); A=np.vstack(predrows.retained.to_numpy())
    result=[]
    losses={
        'MAE': lambda y,p: mae(y,p),
        'Hellinger': lambda y,p: hellinger(y,p),
        'Brier': brier,
    }
    for lname,fn in losses.items():
        u=patient_reduce(fn(Y,R)-fn(Y,C),pids)
        n=patient_reduce(fn(Y,R)-fn(Y,A),pids)
        ulo=float(one_sided_lcb(u,method='t',alpha=.05)); nlo=float(one_sided_lcb(n,method='t',alpha=.05))
        if not movement_pass: state='Negligible'
        elif ulo>0 and nlo>0: state='Qualified'
        elif ulo>0: state='Promising'
        else: state='Equivocal'
        result.append(dict(loss=lname,utility=float(u.mean()),utility_lcb_t=ulo,retention=float(n.mean()),retention_lcb_t=nlo,state=state))
    return result

def run_reference_matrix(bootstrap,seed,selected=None):
    rows=[]; nested_predictions={}; nested_influence={}
    specs=[x for x in SPECS if selected is None or x[0] in selected]
    for j,(name,fn,kind,edges) in enumerate(specs):
        df=pd.read_csv(DATA/fn); states=states_of(df)
        for k,ref in enumerate(REFS):
            cand=candidate_of(kind,states,edges)
            t=time.time()
            out=run_generic_lopo(df,cand,bootstrap=bootstrap,seed=seed+j*10000+k*1000,states=states,
                                 reference_rule=ref,movement_contrast_method='t',use_movement_contrast_for_core=True,
                                 return_predictions=(ref=='nested'),orthogonal_profiles=False)
            row=dict(dataset=name,reference=ref,candidate=out['candidate'],patients=out['patients'],pairs=out['pairs'],
                     state=out['core_evidence_state'],qualified=bool(out['core_qualified']),
                     movement_contrast=float(out['movement_margin_evidence']['contrast']),
                     movement_contrast_lcb=float(out['movement_margin_evidence']['contrast_lcb']),
                     movement_ratio=float(out['PDR']),
                     utility=float(out['PUC']),utility_lo=float(out['PUC_lo']),utility_hi=float(out['PUC_hi']),
                     retention=float(out['NPI']),retention_lo=float(out['NPI_lo']),retention_hi=float(out['NPI_hi']),
                     retention_stability=float(out['retention_stability']),reference_risk=float(out['reference_risk']),
                     retained_risk=float(out['qualified_risk']),seconds=time.time()-t,bootstrap=bootstrap)
            rows.append(row); print('REF',name,ref,row['state'],row['utility_lo'],row['retention_lo'],flush=True)
            pd.DataFrame(rows).to_csv(OUT/f'R1_reference_contract_matrix_{bootstrap}_partial.csv',index=False)
            if ref=='nested':
                nested_predictions[name]=out['prediction_rows'].copy()
                nested_influence[name]=out['patient_influence'].copy()
                out['patient_influence'].to_csv(OUT/f'{name}__nested_patient_influence.csv',index=False)
    rdf=pd.DataFrame(rows); rdf.to_csv(OUT/f'R1_reference_contract_matrix_{bootstrap}.csv',index=False)

    # Summarize whether post-hoc reference choice can change the locked scientific decision.
    order={'Negligible':0,'Adverse':0,'Equivocal':1,'Promising':2,'Qualified':3}
    summ=[]
    for ds,g in rdf.groupby('dataset'):
        lock=g[g.reference=='nested'].iloc[0]
        alts=g[g.reference!='nested'].copy(); alts['rank']=alts.state.map(order)
        best=alts.sort_values(['rank','retention_lo','utility_lo'],ascending=False).iloc[0]
        summ.append(dict(dataset=ds,locked_state=lock.state,locked_qualified=bool(lock.qualified),
                         any_reference_qualified=bool(g.qualified.any()),qualified_reference_count=int(g.qualified.sum()),
                         best_alternative_reference=best.reference,best_alternative_state=best.state,
                         posthoc_reference_can_create_qualified=bool((not lock.qualified) and g.qualified.any()),
                         state_changes_across_references=bool(g.state.nunique()>1)))
    pd.DataFrame(summ).to_csv(OUT/'R1_reference_contract_summary.csv',index=False)

    # Retention guard + real patient-tail profile under locked nested contracts.
    guard=[]; tail=[]; lossrows=[]
    for ds,g in rdf[rdf.reference=='nested'].groupby('dataset'):
        r=g.iloc[0]
        movement=bool(r.movement_contrast_lcb>0)
        util=bool(r.utility_lo>0); ret=bool(r.retention_lo>0)
        guard.append(dict(dataset=ds,movement_supported=movement,utility_supported=util,retention_supported=ret,
                          utility_only_would_pass=bool(movement and util),full_iut_qualified=bool(movement and util and ret),
                          retention_blocks_direct_utility=bool(movement and util and not ret),state=r.state))
        inf=nested_influence[ds]; harm=-inf.NPI_contribution.to_numpy(float); kk=max(1,int(np.ceil(.2*len(harm))))
        tail.append(dict(dataset=ds,patients=len(harm),mean_harm=float(harm.mean()),fraction_harmed=float(np.mean(harm>0)),
                         cvar20_harm=float(np.sort(harm)[-kk:].mean()),worst_patient_harm=float(harm.max()),state=r.state))
        for lr in loss_sensitivity(nested_predictions[ds],movement):
            lossrows.append(dict(dataset=ds,**lr))
    pd.DataFrame(guard).to_csv(OUT/'R2_retention_guard_locked_real.csv',index=False)
    pd.DataFrame(tail).to_csv(OUT/'R3_tail_safety_locked_real.csv',index=False)
    pd.DataFrame(lossrows).to_csv(OUT/'R4_loss_relativity_fixed_predictions.csv',index=False)
    return rdf

def run_replication_invariance(rep_factor=4):
    rows=[]
    # Multi-pair cohorts only; invariant models should not care how many times an identical patient's rows are copied.
    for name,fn,kind,edges in SPECS:
        df=pd.read_csv(DATA/fn)
        if len(df)==df.patient_id.nunique(): continue
        states=states_of(df); counts=df.groupby('patient_id').size().sort_values(ascending=False); pid=str(counts.index[0])
        extra=pd.concat([df[df.patient_id.astype(str)==pid]]*(rep_factor-1),ignore_index=True)
        dup=pd.concat([df,extra],ignore_index=True)
        candidates=[RobustBlendCandidate(states),MultiScaleDeltaCandidate(states),PatientMultiScaleDeltaCandidate(states)]
        for c in candidates:
            a=c.fresh().fit(df).predict(df)
            b=c.fresh().fit(dup).predict(df)
            rows.append(dict(dataset=name,candidate=c.name,duplicated_patient=pid,original_rows=int((df.patient_id.astype(str)==pid).sum()),
                             replication_factor=rep_factor,max_abs_prediction_shift=float(np.max(np.abs(a-b))),
                             mean_abs_prediction_shift=float(np.mean(np.abs(a-b)))))
            print('INV',name,c.name,rows[-1]['max_abs_prediction_shift'],flush=True)
    out=pd.DataFrame(rows); out.to_csv(OUT/'R5_patient_replication_invariance_real.csv',index=False)
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--bootstrap',type=int,default=1000); ap.add_argument('--seed',type=int,default=20260901); ap.add_argument('--datasets',default='all'); ap.add_argument('--skip-invariance',action='store_true')
    a=ap.parse_args(); selected=None if a.datasets=='all' else set(a.datasets.split(','))
    run_reference_matrix(a.bootstrap,a.seed,selected=selected)
    if not a.skip_invariance: run_replication_invariance()

if __name__=='__main__': main()
