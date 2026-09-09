from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from qualifyot.inference import one_sided_lcb
from qualifyot.model import mae, hellinger

REFS=['nested','Persistence','CohortMean','MeanDelta']
DATASETS=['GSE123813','GSE272993_9state','GSE236581','GSE120575','GSE179994','GSE175522','HCC','AML']
ORDER={'Negligible':0,'Adverse':0,'Equivocal':1,'Promising':2,'Qualified':3}

def patient_reduce(values,pids):
    pids=np.asarray(pids,str); values=np.asarray(values,float)
    return np.asarray([values[pids==p].mean() for p in np.unique(pids)],float)

def brier(y,p):
    return np.mean((np.asarray(y,float)-np.asarray(p,float))**2,axis=1)

def parse_vec_col(s):
    return np.vstack([np.asarray(json.loads(x),float) for x in s])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--indir',required=True); ap.add_argument('--outdir',required=True)
    a=ap.parse_args(); indir=Path(a.indir); outdir=Path(a.outdir); outdir.mkdir(parents=True,exist_ok=True)
    mats=[]
    for ds in DATASETS:
        p=indir/f'{ds}__reference_matrix.csv'
        if not p.exists(): raise FileNotFoundError(p)
        g=pd.read_csv(p)
        miss=set(REFS)-set(g.reference.astype(str))
        if miss: raise RuntimeError(f'{ds}: missing references {sorted(miss)}')
        if len(g)!=4: raise RuntimeError(f'{ds}: expected 4 rows, got {len(g)}')
        mats.append(g)
    rdf=pd.concat(mats,ignore_index=True)
    rdf.to_csv(outdir/'R1_reference_contract_matrix_1000.csv',index=False)

    summ=[]
    for ds,g in rdf.groupby('dataset',sort=False):
        lock=g[g.reference=='nested'].iloc[0]
        alts=g[g.reference!='nested'].copy(); alts['rank']=alts.state.map(ORDER)
        best=alts.sort_values(['rank','retention_lo','utility_lo'],ascending=False).iloc[0]
        summ.append(dict(dataset=ds,locked_state=lock.state,locked_qualified=bool(lock.qualified),
                         any_reference_qualified=bool(g.qualified.any()),qualified_reference_count=int(g.qualified.sum()),
                         best_alternative_reference=best.reference,best_alternative_state=best.state,
                         posthoc_reference_can_create_qualified=bool((not bool(lock.qualified)) and bool(g.qualified.any())),
                         state_changes_across_references=bool(g.state.nunique()>1)))
    sdf=pd.DataFrame(summ); sdf.to_csv(outdir/'R1_reference_contract_summary.csv',index=False)

    guard=[]; tail=[]; lossrows=[]
    for ds in DATASETS:
        r=rdf[(rdf.dataset==ds)&(rdf.reference=='nested')].iloc[0]
        movement=bool(r.movement_contrast_lcb>0); util=bool(r.utility_lo>0); ret=bool(r.retention_lo>0)
        guard.append(dict(dataset=ds,movement_supported=movement,utility_supported=util,retention_supported=ret,
                          utility_only_would_pass=bool(movement and util),full_iut_qualified=bool(movement and util and ret),
                          retention_blocks_direct_utility=bool(movement and util and not ret),state=r.state))

        inf=pd.read_csv(indir/f'{ds}__nested_patient_influence.csv')
        harm=-inf.NPI_contribution.to_numpy(float); kk=max(1,int(np.ceil(.2*len(harm))))
        tail.append(dict(dataset=ds,patients=len(harm),mean_harm=float(harm.mean()),fraction_harmed=float(np.mean(harm>0)),
                         cvar20_harm=float(np.sort(harm)[-kk:].mean()),worst_patient_harm=float(harm.max()),state=r.state))

        pr=pd.read_csv(indir/f'{ds}__nested_prediction_rows.csv')
        pids=pr.patient_id.astype(str).to_numpy(); Y=parse_vec_col(pr.target); R=parse_vec_col(pr.reference); C=parse_vec_col(pr.candidate); A=parse_vec_col(pr.retained)
        losses={'MAE':lambda y,p:mae(y,p),'Hellinger':lambda y,p:hellinger(y,p),'Brier':brier}
        for lname,fn in losses.items():
            u=patient_reduce(fn(Y,R)-fn(Y,C),pids); n=patient_reduce(fn(Y,R)-fn(Y,A),pids)
            ulo=float(one_sided_lcb(u,method='t',alpha=.05)); nlo=float(one_sided_lcb(n,method='t',alpha=.05))
            if not movement: state='Negligible'
            elif ulo>0 and nlo>0: state='Qualified'
            elif ulo>0: state='Promising'
            else: state='Equivocal'
            lossrows.append(dict(dataset=ds,loss=lname,utility=float(u.mean()),utility_lcb_t=ulo,retention=float(n.mean()),retention_lcb_t=nlo,state=state))
    gdf=pd.DataFrame(guard); tdf=pd.DataFrame(tail); ldf=pd.DataFrame(lossrows)
    gdf.to_csv(outdir/'R2_retention_guard_locked_real.csv',index=False)
    tdf.to_csv(outdir/'R3_tail_safety_locked_real.csv',index=False)
    ldf.to_csv(outdir/'R4_loss_relativity_fixed_predictions.csv',index=False)

    summary={
      'datasets':len(DATASETS),'reference_rows':len(rdf),
      'state_changes_across_references':int(sdf.state_changes_across_references.sum()),
      'posthoc_reference_can_create_qualified':int(sdf.posthoc_reference_can_create_qualified.sum()),
      'locked_qualified':int(sdf.locked_qualified.sum()),
      'any_reference_qualified':int(sdf.any_reference_qualified.sum()),
      'movement_plus_utility_supported_locked':int(gdf.utility_only_would_pass.sum()),
      'full_iut_qualified_locked':int(gdf.full_iut_qualified.sum()),
      'retention_blocks_direct_utility_locked':int(gdf.retention_blocks_direct_utility.sum()),
      'loss_state_changes_datasets':int(sum(ldf[ldf.dataset==ds].state.nunique()>1 for ds in DATASETS)),
      'tail_any_harmed_datasets':int(sum(tdf.fraction_harmed>0)),
    }
    (outdir/'real_reference_loss_retention_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=='__main__': main()
